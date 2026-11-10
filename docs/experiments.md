# 설치와 실험 실행

## 환경 분리와 고정 revision

Light-WAM `db7d05e7`, LIBERO `8f1084e3`, RoboTwin `bf44be51`의 API를 기준으로 작성했습니다.
RoboTwin의 현재 main은 XPolicyLab 기반으로 변경되었습니다. 고정한 revision은 Light-WAM의 `third_party/RoboTwin/README.vendor.md`에 명시된 upstream으로, 현재 main을 기존 qpos 어댑터와 혼합하지 않습니다.

```bash
python scripts/bootstrap_upstreams.py
```

외부 코드는 `.external/`에 보관하며 Git에 포함하지 않습니다. 기존 checkout의 revision이 다르거나 tracked 파일이 수정되어 있으면 스크립트가 멈추고 보존합니다. Git worktree를 만들 필요 없이 현재의 격리된 checkout을 사용합니다.

CPU 개발·테스트 환경은 `bash scripts/setup.sh`로 준비합니다. 실제 LIBERO 환경 검사에는 `bash scripts/setup.sh libero`를 사용합니다. CPU PyTorch가 포함된 `.venv`를 CUDA WAM 환경으로 재사용하지 마세요.

## GPU / Light-WAM 환경

실제 모델 추론은 CUDA GPU가 필요합니다. 필요한 VRAM은 precision, text encoder, action horizon, 입력 크기에 따라 달라집니다. 특정 VRAM 용량에서 실행된다고 보장하지 않습니다.
공식 Light-WAM의 평가용 conda 환경 파일을 기준으로 각각 준비합니다:

```bash
conda env create -f .external/Light-WAM/scripts/libero_env.full.yml
conda activate lightwam-libero-eval
python -m pip install -e .external/Light-WAM --no-deps
python -m pip install -e . --no-deps
```

RoboTwin은 별도로 `scripts/robotwin_env.full.yml`을 사용해 `lightwam-robotwin-eval` 환경을 만듭니다. 이 GPU 설치 경로와 실제 WAM 추론은 CPU 클라우드에서 검증하지 않았습니다. upstream 환경 파일을 덮어쓰거나 CPU lockfile을 GPU 환경에 적용하지 않습니다.

공식 모델 자산 안내:

- Backbone: `Wan-AI/Wan2.1-T2V-1.3B`
- Released checkpoints: `l1ziang/lightwam-checkpoints`
- Dataset/cache: Light-WAM README 참고

벤치마크에 맞는 checkpoint, 원래 training config.yaml, dataset_stats.json 세 파일이 필요합니다. LIBERO 통계·체크포인트를 양팔 모델에 재사용하지 않습니다. Backbone은 Light-WAM이 기대하는 위치에 두고 `DIFFSYNTH_MODEL_BASE_PATH`를 해당 checkpoints 디렉터리로 설정합니다. 다운로드가 끝난 환경에서는 `DIFFSYNTH_SKIP_DOWNLOAD=true`로 누락 파일을 발견할 수 있습니다.

Hugging Face 접근은 `huggingface.co`와 실제 다운로드가 사용하는 CDN/Xet 목적지 허용이 필요할 수 있습니다. 네트워크 정책이 거부하면 환경 설정에서 해당 목적지를 추가해야 합니다. 공개 모델에 토큰이 반드시 필요한 것은 아닙니다. 비밀 값은 저장소에 넣지 않습니다.

## LIBERO 실행

`configs/libero_correction.example.json`을 복사하고 경로를 수정합니다. 저장소 루트에서 실행합니다. 매 재계획에 현재 지시를 인코딩하므로 text encoder가 로드되어 있어야 합니다.

```bash
xenoarm validate configs/libero_correction.example.json
xenoarm run --config configs/libero_correction.example.json --no-corrections --output results/libero-standard
xenoarm run --config configs/libero_correction.example.json --mode drain --output results/libero-drain
xenoarm run --config configs/libero_correction.example.json --mode cancel --output results/libero-cancel
xenoarm compare results/libero-drain/episodes.jsonl results/libero-cancel/episodes.jsonl
```

예시 장면에는 검은 bowl 2개, plate, ramekin이 존재합니다. 원래 목표 `On(akita_black_bowl_1, plate_1)`을 같은 장면의 `On(akita_black_bowl_1, glazed_rim_porcelain_ramekin_1)`으로 변경합니다. 자체 제작 correction episode이며 공식 LIBERO 점수와 구분합니다. 목표 predicate를 설정할 수 있다는 것과 모델이 그 목표를 달성한다는 것은 별도 검증 대상입니다.

초기 상태는 `seed % 초기상태 수`로 고정하고 30개 settle step 뒤의 simulator state hash를 기록합니다. 정책 seed도 episode마다 설정합니다. `max_steps`는 settle을 제외한 제어 step 수입니다. phase 기반 시점은 아직 지원하지 않으므로 step 60을 파지 후라고 가정하지 않습니다.

## RoboTwin 2.0 실행

RoboTwin에는 추가로 objects, embodiments, background textures, motion-planner 의존성, GPU renderer가 필요합니다. `.external/RoboTwin/assets/_download.py`와 공식 설치 안내를 확인하고 해당 checkout에 자산을 준비합니다. 다운로드 전에 공간과 접근 가능 여부를 확인하세요. 모델과 자산은 이 저장소에 포함하지 않습니다.

upstream의 오래된 `script/_install.sh`는 라이브러리를 수정하고 planner의 collision 조건을 제거합니다. 이 스크립트를 자동 실행하지 않습니다. 의존성·경로 설정은 공식 문서를 기준으로 확인하고 충돌 검사나 TLS 검증을 끄는 우회는 사용하지 않습니다.

```bash
# lightwam-robotwin-eval 환경에서, 저장소 루트에서 실행
xenoarm run --config configs/robotwin_correction.example.json --mode drain --output results/robotwin-drain
xenoarm run --config configs/robotwin_correction.example.json --mode cancel --output results/robotwin-cancel
```

`place_a2b_left` 장면을 유지하며 오른쪽 배치 predicate로 전환합니다. 예시의 A/B는 연구용 역할 표현입니다. 실제 물체명 지시가 필요한 모델에서는 episode의 A/B 자산명으로 구체화해야 합니다. 외부 description API나 LLM은 자동 호출하지 않습니다. seed를 그대로 실행하며 expert 성공 seed를 자동 검색하지 않으므로 공식 evaluator의 expert-filtered 점수와 직접 비교하지 않습니다.

어댑터는 native 성공 시 자동 종료를 억제하고 runner가 현재 목표와 연속 성공 유지 횟수로 종료를 결정합니다. 이전 목표를 잠깐 달성해도 이후 정정이 예약되어 있으면 계속합니다. step limit은 실험의 `max_steps`로 설정합니다.

## 압축 비교

동일 config에서 policy 부분만 바꿉니다. `precision`은 bf16, fp16, no(FP32)를 지원하고 `num_inference_steps`는 diffusion 계산 예산 baseline입니다. 실제 pruning/quantization은 `model_transform` hook으로 연결해야 합니다. 새로운 checkpoint에는 맞는 학습 config와 통계를 함께 제공합니다.

일반 작업과 correction 작업을 각각 평가해 벤치마크 내부의 성능 하락폭을 분석합니다. LIBERO와 RoboTwin의 절대 성공률 차이를 팔 수의 인과효과로 해석하지 않습니다. 3/5 seed 예시는 시작용이며 논문용 반복 수나 유의성을 보장하지 않습니다.
