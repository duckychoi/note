# Floor Plan → Clash-Free 3D (인터랙티브 데모)

"From Floor Plan to Clash-Free 3D Model with Claude Code"가 가리키는 **워크플로우를 원본 재구성**한 three.js 인터랙티브 3D 데모(단일 HTML).

> ⚠️ 원본 Notion 페이지가 **JavaScript 잠금**이라 본문·영상을 추출하지 못함(제목만 확인). 따라서 이건 **제목이 가리키는 워크플로우를 도메인 지식으로 original 재구성**한 것이며, 특정 자료의 디자인·문구·이미지 복제가 아님. talktrip 랜딩과는 **독립 산출물**(퀄리티 비교용).

## 데모 내용 (5단계)
1. **2D 플로어플랜** — 도면에서 벽 좌표 추출
2. **3D 변환** — 벽 압출(3m) + 바닥·천장 슬래브
3. **구조+MEP** — 구조 보(Beam) + 급수배관·공조덕트 라우팅
4. **충돌 검사** — MEP↔구조 AABB 간섭 탐지 → clash 리스트(빨강 하이라이트)
5. **충돌 해소** — 배관 리라우팅·덕트 이동 → 재검사 → clash-free(초록)

각 단계에 **Claude Code 스타일 명령**을 함께 표시.

## 기술
- three.js r128 + OrbitControls(CDN), 단일 `index.html`, 의존성 로컬 0.
- 충돌검사 = `THREE.Box3.intersectsBox`(AABB). 지오메트리=박스/실린더 자작.
- 인터랙션: 단계 버튼/클릭, 궤도 회전·줌, 실시간 clash 리스트.

## 실행
`index.html`을 브라우저로 열기(인터넷 필요 — three.js CDN).
