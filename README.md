# 두 릴스 논문 분석 — MS Rocket 델타 + RSI 서베이(arXiv 2609.11873)

- `docs/ANALYSIS.md` — 정독·코드 수준 분석. Part A: MS Rocket(@aislash.official, #64와 동일 원문, 델타만). Part B: "The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement"(@conanssam) — HCI 수식, 개선 루프 9요소, B0–L5 경계, 단계별 메커니즘, 기법 지도, 산업 사례, 평가 원칙, 비판적 평가, 우리 연구 연결.
- `src/rsi_core.py` — HCI(식 1–4) 구현, 개선 루프 데이터 모델·`run_round`·상속 기록, 단계 판정, HDSO식 쌍대 입장 검사, Library Drift식 스킬 수명관리, RQGM식 에포크 동결 평가기, L3 학습자 조건부 과제 선택, STOP식 자기 적용 개선기 골격, 예산 장부·예산 일치 비교. 서베이에 식이 없는 부분은 `OUR CHOICE:` 표시.
- `tests/test_rsi_core.py` — 11개 테스트. 논문 수치(R_d 98.2/89.6/90.5/86.8 등)로 HCI 식 검산. `python3 tests/test_rsi_core.py`.

주의: 서베이라 새 알고리즘·통제 실험이 없고, §5 산업 수치는 모두 회사 자체 보고다. 코드는 서베이가 서술한 메커니즘을 옮긴 골격이지 원 논문들의 재현이 아니다.
