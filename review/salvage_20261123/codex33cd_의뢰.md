# codex 적대 검수 의뢰 — #33 관문 3·4 (read-only, 한국어로 답할 것, 빠른 검수)

폴더: `C:\Users\user\OneDrive\코드\바이브연구\note\review\salvage_20261123\`
읽을 파일:
- `PREREG_33.md` — 관문 3·4 사전등록과 결과는 파일 뒤쪽에 있다.
- `gate33c.py`, `gate33c.json`, `gate33c.log`
- `gate33d.py`, `gate33d.json`, `gate33d.log`
- 이전 검수 `codex33_최종_sol.txt`(관문 1·2, NOT READY 8건)
- 공용 모듈 `gate33.py`, `gate33b.py`

현재 결론: "#33 선택기 우위 주장 철회(종료)."
- 관문 3: 튜닝한 signed_corr 대비 0.7%로 불통과다.
- 관문 4(불규칙 교체): 7.9%이지만 5% 여유 하한이 음수이고, 회복 RMST도 불통과다. 잔차 신용의 이득은 없다.

확인해 주세요.
1. 이전 지적(HIGH2 Kale 표본추출·IPW, HIGH3 restart 초기화, MED4/5)의 수정이 gate33c.kale_ipw_lasso·simulate에서 올바른가. 새 버그가 있나(파일:줄).
2. 종료 판정을 **뒤집을 만한** 결함이 있나.
   - 신규 계열을 불리하게 만든 버그·격자 문제.
   - 기준선을 부당하게 강하게 만든 누출: signed_corr가 선택 전에 정답을 쓰는가.
   - 관문 4의 불규칙 교체 생성, 회복 지표·절단 처리, θ 규칙, 짝 부트스트랩.
3. 사전등록과 실제 코드의 불일치.

출력: 결함 목록(HIGH/MED/LOW, 파일:줄, 근거, 수정안)과 "종료 판정 유지 / 재검토 필요" 한 줄, 그리고 한 문단 요약.
