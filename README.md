# 최신 학습법 정리 + 축소 재현 (깊이 · 이미지/영상 생성 · LLM · VLM)

- `docs/TRAINING_RECIPES.md` — 25편 남짓의 1차 기술보고서(arXiv TeX)로 정리한 분야별 학습 절차, 수치, arXiv 번호, 논문 간 불일치. ✔ 표시는 원문에서 다시 대조한 수치.
- `docs/REPRO_PLAN.md` — **사전등록** 축소 재현 계획. 티어 A(RTX A6000 ×2) / 티어 B(RTX 5090). 트랙별 데이터·모델·예산·양성 대조·가설·관문·중단조건.
- 코드: `depth/`(전 과정) · `gen/`(rectified-flow DiT 커리큘럼) · `llm/`(사전학습 3단계 + SFT→GRPO→증류) · `vlm/`(LLaVA식 2단계).

## 설치 (GPU 머신)
```bash
pip install torch torchvision timm numpy pillow         # depth, gen, llm 핵심
pip install transformers datasets tiktoken diffusers accelerate   # 데이터 준비·HF 모델 연결
```

## 실행 요약
**깊이 (DA2 + Depth Pro 절차)** — 데이터는 `depth/data.py` 머리말의 폴더 형식으로 변환
```bash
C=depth/configs/tierA.json; R="torchrun --nproc_per_node 2"   # 티어 B: C=depth/configs/tierB.json; R=python
$R depth/train.py --config $C --stage teacher --out runs/depth/teacher                        # 합성 전용 교사 (SSI + 2·gm)
python depth/pseudo_label.py --config $C --ckpt runs/depth/teacher/final.pt --images /data/coco_unlabeled --out /data/pseudo_coco
$R depth/train.py --config $C --stage student --out runs/depth/student                        # 가짜 라벨 학생 (상위10% 무시 + 특징정렬)
$R depth/train.py --config $C --stage metric  --init runs/depth/student/final.pt --out runs/depth/metric   # CSTM f_c=300
$R depth/train.py --config $C --stage sharpen --init runs/depth/student/final.pt --out runs/depth/sharp    # 합성 전용 경계 다듬기
python depth/eval.py --config $C --stage_cfg student --ckpt runs/depth/student/final.pt --data /data/eval/nyu --mode relative --boundary --max_depth 10 --out runs/depth/eval/nyu.json
```
**생성 (Wan 절차 축소)**
```bash
python gen/precompute_latents.py image --src /data/imagenet100/train --res 128 --out /data/lat/img128       # Wan VAE, T=1
python gen/precompute_latents.py video --src /data/UCF-101 --res 128 --frames 17 --label_offset 100 --out /data/lat/ucf128_17f
torchrun --nproc_per_node 2 gen/train.py --config gen/configs/tierA.json --out runs/gen              # 티어 B: python ... tierB.json
```
**LLM**
```bash
python llm/prepare_data.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --out /data/tok/fineweb_edu --max_tokens 7e9
torchrun --nproc_per_node 2 llm/pretrain.py --config llm/configs/tierA.json --out runs/llm/dense    # 비교팔: *_cosine, *_moe, *_moe_auxloss
python llm/run_posttrain.py --model Qwen/Qwen2.5-0.5B --sft_steps 300 --grpo_steps 400 --out runs/post/0.5b
```
**VLM**
```bash
python vlm/llava_mini.py --vision google/siglip-base-patch16-224 --llm Qwen/Qwen2.5-0.5B --stage0_json ... --stage1_json ... --out runs/vlm
```

## 검증 상태 (정직하게)
CPU(torch 2.14 CPU) 스모크 테스트로 **모든 트랙이 끝까지 실행되고 손실이 줄어드는 것**만 확인했다. 성능 판정은 아니다 — 판정은 `REPRO_PLAN.md`대로 GPU에서.
결과 원문: `results/SMOKE_RESULTS.md`.

| 트랙 | 스모크로 확인 | 이 서버에서 미검증(가중치·데이터·GPU 필요) |
|---|---|---|
| depth | 교사→가짜라벨→학생(+특징정렬)→미터→다듬기→평가 전부 실행, 교사 손실 감소, 경계 F1 지표 단위검증 | 사전학습 DINOv2 가중치 로드, DDP, 실제 데이터 변환 |
| gen | 2단계 커리큘럼(이미지→이미지+영상), EMA, CFG 샘플러, 공유 adaLN 파라미터 절감 | `precompute_latents.py`(diffusers VAE), DDP |
| llm | 사전학습 3단계 dense·MoE(aux-free), SFT·GRPO·증류, **GRPO 부호 양성대조**(`diag_grpo_sign.py`) | `prepare_data.py`, `run_posttrain.py`(HF Qwen), DDP |
| vlm | 프로젝터만 → +LLM 2단계, 답 토큰 손실, 동결 확인 | `build_hf`(SigLIP+Qwen), LLaVA json 로더 |

알려진 단순화: 깊이 경계 F1은 원 정의를 단순화(오른쪽·아래 이웃), `tol=2` 변형을 별도 보고. GRPO 생성은 KV 캐시 없는 단순 샘플링(티어 A는 vLLM 권장). VLM은 단일 턴만. 생성 트랙은 VAE·텍스트 인코더를 학습하지 않고 클래스 조건만.
