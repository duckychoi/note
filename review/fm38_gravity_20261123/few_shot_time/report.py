"""Generate the Korean experiment note; verify reported means independently with raw engine outcomes."""
import argparse
import json

import numpy as np

from devices import BUDGETS, HERE, N_CAL, N_DEVICES, N_HISTORIES, N_PAIRS, N_TEST, TIMES
from swing import BAND


def verify(report):
    with np.load(HERE/"devices.npz") as raw:
        wins = abs(raw["peak"] - raw["goal"][...,None]) <= BAND
        assert np.unique(raw["start"]).size == raw["start"].size
        assert len(np.unique(raw["damping"])) == 2*N_DEVICES
    logs = []
    for filename, sl in (("calibration",slice(None,N_CAL)),("test",slice(N_CAL,None))):
        with np.load(HERE/f"{filename}.npz") as f:
            hits = np.stack([[truth[np.arange(len(truth))[:,None], actions]
                              for actions in device] for truth,device in zip(wins[:,sl],f["actions"])])
            np.testing.assert_array_equal(hits,f["observed"])
            logs.append(hits)
    practice, test = logs
    with np.load(HERE/"locked_times.npz") as f:
        locked = f["selected"]
    for g, domain in enumerate(("original","strong")):
        devs = range(g*N_DEVICES,(g+1)*N_DEVICES)
        for b, episodes in enumerate(BUDGETS):
            successes = 0
            for d in devs:
                for m, histories in enumerate(practice[d]):
                    for h, pairs in enumerate(histories.reshape(N_HISTORIES,N_PAIRS,2)):
                        votes = pairs[:episodes//2].sum(0)
                        j = 0 if votes[0]>votes[1] else 2
                        assert locked[d,m,h,b] == TIMES[j]
                        successes += test[d,m,:,j].sum()
            denom = N_DEVICES*N_HISTORIES*test.shape[1]*N_TEST
            np.testing.assert_allclose(successes/denom,report["domains"][domain]["budgets"][str(episodes)]["rate"])


def note(r):
    groups = r["domains"]
    pct = lambda x: f"{100*x:.2f}%"
    comp = lambda x: f"{100*x['delta']:+.2f} [{100*x['ci'][0]:+.2f}, {100*x['ci'][1]:+.2f}]"
    rows = ["# #38 · 새 그네에서 4·8·16회 연습해 촬영 시점 고르기",
        "강한 저항에서 고정 0.48초보다 성공률이 높았던 최소 관측 예산은 요청한 세 예산 중 4회, 추가로 잰 2회까지 포함하면 2회였다. 강한 저항에 맞춰 미리 고른 고정 0.32초가 더 좋았고, 두 저항을 반반 섞으면 연습 없는 고정 0.40초가 더 좋았다.",
        f"경사와 감쇠가 고정된 시뮬레이터 장치 {2*N_DEVICES}개를 새로 생성했다. 장치별 시험 {N_TEST}개로 총 {2*N_DEVICES*N_TEST:,}개의 미사용 장면이다. 출발 사진은 공통이고, 각 시험 정책은 탐색 밀기 후 사진을 한 장만 본다. 각 장치에서 시작각과 목표각을 같게 맞춰 0.32초와 0.48초를 한 번씩 시험했다. 각 회에 약한 탐색 밀기와 선택한 본 밀기가 든다. 자신의 성패만 세어 0.32초가 더 많이 성공하면 전환하고 동점이면 0.48초를 유지한다. 모델 3개와 독립 연습 기록 8개는 별도 실험을 반복한 것이고 한 정책에 합쳐 주지 않았다.",
        "## 강한 저항에서 고정 0.48초와 비교",
        "| 연습 횟수 (실제 밀기) | 고정 → 연습 후 성공률 | 차이와 동시 95% 구간 (%p) | 0.32초 선택 빈도 |\n|---|---:|---:|---:|"]
    strong = groups["strong"]
    for n in BUDGETS:
        v = strong["budgets"][str(n)]
        rows[-1] += f"\n| {n}회 ({2*n}번) | {pct(strong['fixed']['0.48'])} → {pct(v['rate'])} | {comp(v['comparisons']['0.48'])} | {pct(v['early'])} |"
    rows += ["## 어떤 기준과 비교하느냐에 따른 판정",
        "| 새 장치 분포 | 고정 0.48초 | 고정 0.40초 | 고정 0.32초 | 16회 연습 |\n|---|---:|---:|---:|---:|"]
    for group, label in (("original","원래 저항"),("strong","강한 저항"),("balanced","두 범위 반반")):
        v = groups[group]
        rows[-1] += f"\n| {label} | {pct(v['fixed']['0.48'])} | {pct(v['fixed']['0.4'])} | {pct(v['fixed']['0.32'])} | {pct(v['budgets']['16']['rate'])} |"
    rows += [
        f"두 분포를 반반 섞었을 때 고정 0.48초 대비 4회·8회·16회 차이(%p)는 "+", ".join(comp(groups['balanced']['budgets'][str(n)]['comparisons']['0.48']) for n in (4,8,16))+"다. 강한 저항에서 16회 보정은 고정 0.32초와 비교해 "+comp(strong["budgets"]["16"]["comparisons"]["0.32"])+"%p였다. 혼합 분포의 16회 보정과 고정 0.40초의 차이는 "+comp(groups["balanced"]["budgets"]["16"]["comparisons"]["0.4"])+"%p였다.",
        f"강한 저항에서 같은 출발·목표의 연습 한 쌍이 서로 다른 성패를 준 비율은 {pct(strong['informative'])}다. 16회 뒤에도 누적 성패가 동점인 비율은 {pct(strong['budgets']['16']['tie'])}였다. 고정 0.48초에 대한 차이가 양수인지와 사전에 정한 +3%p 실용 기준을 구분했다. 강한 저항에서 시험한 예산 중 +3%p에 도달한 경우는 없었다.",
        "## 해석 범위와 재현",
        "실험 때마다 같은 장치로 리셋할 수 있고 각 후보를 같은 시작·목표에서 비교할 수 있다는 가정이다. 0.32초는 이전 강감쇠 대규모 실험에서 골랐다. 총성공률은 모든 새 목표를 포함하며 시험 성공률에 연습 비용을 차감하지 않았다. 구간은 장치 간 변동을 기준으로 36개 비교를 동시 보정했으며, 새로 모델을 다시 학습할 때의 변동까지 다루지 않는다. 이 작은 차이를 모든 센서·물성이나 공개 최고 성능에 일반화할 수 없다.",
        "기존 그네 비교는 공개 Flow-JEPA를 개조한 영상 보정기를 평가했다. 원 논문의 공개 과제와 별개이며, 비슷한 연구인 JEPA-TTT도 새 물성 장면에서 동작하며 예측기를 갱신한다. 두 연구와 동일한 자료·횟수·평가에서 승리한 실험은 없다. [기존 Flow 비교](../damping_gate/README.md) · [Flow-JEPA 원 논문](https://arxiv.org/abs/2608.29029) · [JEPA-TTT 원 논문](https://arxiv.org/abs/2610.00722)",
        "MuJoCo 환경에서 devices.py로 장치를 만든 뒤 기존 CUDA 학습 환경에서 predict.py calibration → choose.py → predict.py test → analyse.py → report.py 순서로 실행한다. report.py --check는 원래 최고점과 고른 힘에서 성패·예산별 선택·시험 성공률을 다시 계산하고 연구노트와 대조한다. [사전 계획](protocol.md) · [탐색 각도·최고점](devices.npz) · [연습 성패](calibration.npz) · [시점 고정](lock.json) · [시험 성패](test.npz) · [전체 평균·구간](results.json)",
    ]
    return "\n\n".join(rows)+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = json.loads((HERE/"results.json").read_text())
    verify(result)
    path = HERE/"README.md"
    if args.check:
        assert path.read_text(encoding="utf8")==note(result)
    else:
        path.write_text(note(result),encoding="utf8")
