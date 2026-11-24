# codex 적대 검수 의뢰 — #33 읽기 예산 관문 1·2 (read-only, 한국어로 답할 것)

폴더: `C:\Users\user\OneDrive\코드\바이브연구\note\review\salvage_20261123\`
읽을 파일: `PREREG_33.md`(관문 1·2 사전등록과 결과), `gate33.py`, `gate33.json`, `gate33.log`, `gate33b.py`, `gate33b.json`, `gate33b.log`. 참고 원본: `..\priority_20261122\feature_budget.py`. Kale et al. 2017 = arXiv 1706.04690.

주장: "읽기 예산 k=12/d=200, 상관 센서(ρ=.5), 지지 교체 τ=480에서, |w| 유지 + 잔차 신용 x_j·err의 UCB 탐색 선택기(resid_ucb)가 검증에서 고른 최강 기준선보다 누적 MSE가 5% 이상 낮다."

반박해 주세요. 특히 다음을 확인해 주세요.
1. 예산 위반이나 정보 누출이 있나. 정답·미선택 좌표를 선택 전에 쓰거나, 참 지지·β를 선택기에 쓰는 경로(oracle_support 참고 팔은 판정 제외).
2. 기준선이 불공정하게 약한가. kale_* 단순화판, signed_corr, random-IPW의 튜닝 격자, 학습기(LMS) 공통화가 신규에 유리한가. Kale 충실판(gate33b.kale_faithful)이 논문 알고리즘을 충실하게 구현했나(Dantzig LP 대신 Lasso, 쌍 IPW 스케일, PSD 투영, 32라운드 갱신, 예측 방식). 버그가 있나.
3. 통계: 시드 단위 짝 부트스트랩, 검증/시험 분리, 사전등록 수정(관문 2 수정 1)의 정당성.
4. 기여 절제(A1/A2/A3) 해석과 회복 지표(가중 재현율, θ 선택 규칙)의 타당성.
5. 결론 문장에서 넘치는 부분, 그리고 고쳐야 할 코드 줄(파일:줄).

출력: 결함 목록(심각도 HIGH/MED/LOW, 파일:줄, 근거, 수정안), 그다음 최종 판정(READY / NOT READY)과 한 문단 요약.
