"""Regenerate multi-baseline results and recheck the closed-form likelihood."""
import json
import sys

import numpy as np
from scipy.stats import multivariate_normal

from pf42_multibase import K, OUT, component, valid


DATASETS = {"hammer":"HAMMER","ibims":"iBims-1","diode":"DIODE 실내","diode_outdoor":"DIODE 실외"}


def contrast(a,b):
    keys = sorted(a)
    x,y = np.array([[r[k] for k in keys] for r in (a,b)])
    i = np.random.default_rng(421339).integers(len(keys),size=(20000,len(keys)))
    return 100*(1-x.mean()/y.mean()),np.quantile(100*(1-x[i].mean(1)/y[i].mean(1)),[.025,.975])


def check_evidence():
    d = K*.49+.35
    sigma,tau,eta = .8,.5,.05
    _,_,w = component(d[None],sigma,tau,eta)
    for drop in range(-1,len(K)):
        use = np.arange(len(K)) != drop
        k = K[use]
        cov = sigma**2*np.eye(len(k))+tau**2+1.5**2*np.outer(k,k)
        expected = multivariate_normal.logpdf(d[use],mean=.55*k,cov=cov)
        expected += np.log1p(-eta) if drop == -1 else np.log(eta/len(K)/384)
        np.testing.assert_allclose(w[0,drop+1],expected,atol=2e-9)


def report():
    check_evidence()
    result = json.loads((OUT/"results.json").read_text(encoding="utf8"))
    with np.load(OUT/"held_predictions.npz") as source:
        held = {k:source[k] for k in source.files}
    with np.load(OUT/"held.npz") as source:
        history = {k:source[k] for k in ("y","row","column")}
    mask = valid(held)
    s = result["scores"]
    for key, groups in s.items():
        e = abs(held[key][mask]/held["truth"][mask]-1)
        for g, score in groups.items():
            np.testing.assert_allclose(score,e[held["group"][mask] == g].mean(),atol=1e-12)
    yield f"공통 보류 화소는 {int(mask.sum()):,}개, 장면은 {len(result['held_scenes'])}개, 계열은 {len(s['joint'])}개다."
    yield ("기선 4, 8, 12, 16, 20 cm에서 유효 관측이 있었던 점 수는 "
           + ", ".join(f"{n:,}" for n in result["available_views"])+"개다.")
    yield ("유효 기선 수 1, 2, 3, 4, 5인 점 수는 "
           + ", ".join(f"{n:,}" for n in result["available_baseline_counts"][1:])+"개다.")
    yield "보정 실사에서 맞춘 기선별 공통 픽셀 보정은 "+", ".join(f"{v:+.4f}" for v in result["offsets"])+" px다."
    yield ""
    yield "| 보류 실사 교사 | AbsRel ↓ | 16 cm 미터 중앙값 대비 감소 [95%] |"
    yield "|---|---:|---:|"
    titles = (("continuous_long","기존 16 cm 미터 중앙값"),("continuous_log_long","기존 16 cm 로그 중앙값"),
              ("continuous_all","같은 다기선 전체 미터 중앙값"),("same_origin","같은 우도에서 공통편향 0"),
              ("joint","공통편향을 포함한 다중 기선 교사"))
    for key,label in titles:
        gain,(lo,hi) = contrast(s[key],s["continuous_long"])
        yield f"| {label} | {result['means'][key]:.9f} | {gain:+.3f}% [{lo:+.3f}, {hi:+.3f}] |"
    gain,(lo,hi) = contrast(s["joint"],s["continuous_all"])
    bias_gain,_ = contrast(s["joint"],s["same_origin"])
    yield (f"같은 다섯 기선 중앙값 대비 결합 교사의 감소는 {gain:+.3f}% [{lo:+.3f}, {hi:+.3f}], "
           f"같은 우도에서 공통 편향을 추가한 효과는 {bias_gain:+.5f}%였다.")
    yield (f"기존 16 cm의 AbsRel이 3.5%를 넘은 {result['bad_long_count']}개 점 중 "
           f"다기선 중앙값은 {result['bad_long_all_median_recovered']}개, 결합 교사는 "
           f"{result['bad_long_joint_recovered']}개를 3.5% 이내로 옮겼다.")
    y = np.log(history["y"].astype(float))
    repeat = mask&(np.nanmax(y,axis=1)-np.nanmin(y,axis=1) <= .002)&(
             abs(np.nanmedian(y,axis=1)-np.log(held["truth"])) > .035)
    pass_all = repeat&(abs(held["continuous_all"]/held["truth"]-1) <= .035)
    pass_joint = repeat&(abs(held["joint"]/held["truth"]-1) <= .035)
    i = np.flatnonzero(pass_joint)[0]
    yield (f"기존 반복값이 거의 같았는데 실제 로그 오차가 0.035를 넘은 {int(repeat.sum())}개 점 중 "
           f"다기선 중앙값은 {int(pass_all.sum())}개, 결합 교사는 {int(pass_joint.sum())}개를 복구했다.")
    yield (f"보류 사례 {held['scene'][i]} 좌표 (행 {history['row'][i]}, 열 {history['column'][i]})의 참 깊이는 "
           f"{held['truth'][i]:.4f} m다. 기존 16 cm 중앙값 {held['continuous_long'][i]:.4f} m, "
           f"다기선 중앙값 {held['continuous_all'][i]:.4f} m, 결합 교사 {held['joint'][i]:.4f} m였다.")
    student = json.loads((OUT/"student-results.json").read_text(encoding="utf8"))
    yield ""
    yield "| 외부 단안 평가 | 16 cm 직접 감독 ↓ | 다기선 직접 감독 ↓ | 결합 교사 ↓ | 다기선 직접 대비 감소 [95%] | 16 cm 직접 대비 감소 |"
    yield "|---|---:|---:|---:|---:|---:|"
    for key,label in DATASETS.items():
        d = student["metrics"][key]
        lo,hi = d["interval"]
        yield (f"| {label} | {d['direct_long']:.6f} | {d['direct']:.6f} | {d['proposed']:.6f} | "
               f"{d['relative']:+.3f}% [{lo:+.3f}, {hi:+.3f}] | {d['relative_long']:+.3f}% |")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf8")
    print("\n".join(report()))
