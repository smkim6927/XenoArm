# XenoArm

**Light-WAM의 경량화가 작업 중 언어 정정 반영 능력에 미치는 영향을 측정하는 실험 도구입니다.**
단일 팔 **LIBERO / MuJoCo**와 양팔 **RoboTwin 2.0 / SAPIEN**을 같은 정정·기록 인터페이스로 연결합니다.

핵심 질문은 압축 후 일반 작업 성공률이 유지되어도, 사람이 지시를 바꿀 때 새 의도를 반영하는 능력이 더 크게 저하되는가입니다. 이 저장소는 이를 검증할 실험 기반이며, 해당 가설이나 새로운 압축 방법의 성능을 입증한 결과를 포함하지 않습니다.

## 빠른 실행: CPU / MuJoCo

Python 3.10과 [uv](https://docs.astral.sh/uv/)를 사용합니다. `uv.lock`은 CPU 환경의 의존성을 고정합니다.

```bash
bash scripts/setup.sh
make check test
.venv/bin/xenoarm run --config configs/smoke.json --mode drain --output results/drain
.venv/bin/xenoarm run --config configs/smoke.json --mode cancel --output results/cancel
.venv/bin/xenoarm compare results/drain/episodes.jsonl results/cancel/episodes.jsonl
```

출력 폴더는 새 경로여야 합니다. 기존 결과를 덮어쓰지 않습니다.
이 smoke 실험은 **2축 MuJoCo carriage와 규칙 정책**을 사용해 중간 정정·chunk 취소를 검증합니다.
WAM 추론이나 LIBERO 조작 성공률을 대신하지 않습니다.

## 진행률 · W&B · 시뮬레이터 영상

```bash
.venv/bin/xenoarm run --config configs/smoke.json \
  --output results/visual-run --progress --wandb offline --video
```

tqdm으로 episode/step 진행률과 정정 발생을 표시하고, 다음 파일을 저장합니다:

- `report.html`: 브라우저에서 여는 결과 페이지 — episode별 MP4 재생, 정정 시점·지연 그래프
- `videos/seed-*.mp4`: 실제 simulator RGB, 현재 지시·command/action version·stale 표시
- `plots/seed-*.png`, `summary.json`: 그래프와 집계 지표
- `wandb/offline-run-*`: W&B step/episode 지표와 영상. API 키 없이 로컬 기록

W&B 업로드는 `--wandb online --wandb-project <project>`로 명시적으로 선택합니다. 기존 W&B 로그인 또는 환경 설정의 `WANDB_API_KEY`를 사용합니다. 자세한 설정과 offline 동기화 방법은 [시각화·기록 가이드](docs/observability.md)에 있습니다.

녹화는 기본 off이며 `--video`로 켜면 처음 3개 episode를 녹화합니다. `--video-episodes`, `--video-stride`로 기록량을 조절하고 `--no-progress`로 진행률을 끌 수 있습니다. HTML 리포트는 녹화 여부와 관계없이 생성됩니다.

## 실제 LIBERO 환경 확인

```bash
bash scripts/setup.sh libero
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/libero_smoke.py --steps 5
# 모델 없이 실제 Panda/agentview/wrist 영상을 확인:
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python scripts/libero_smoke.py \
  --steps 30 --video-output results/libero-preview
```

공식 LIBERO 장면·Panda 팔·초기 상태를 로드하고 두 카메라의 RGB 영상, 물리 상태 변화, 변경된 목표 predicate를 검사합니다. 모델 없이 진행하는 환경 검증이므로 작업 성공률 평가는 아닙니다. 렌더링에는 EGL/OpenGL 라이브러리가 필요합니다. CPU PyTorch는 이 환경 검증용이며, GPU WAM 환경과 분리합니다.

## Light-WAM 연구 실행

```bash
.venv/bin/python scripts/bootstrap_upstreams.py
# GPU 평가 환경을 준비하고 체크포인트/학습 config/통계 파일 경로를 채운 뒤:
xenoarm run --config configs/libero_correction.example.json --output results/libero-full-cancel
xenoarm run --config configs/robotwin_correction.example.json --output results/robotwin-full-cancel
```

위 두 파일은 **GPU 및 체크포인트 경로를 설정해야 하는 예시**입니다.
설치·가중치·RoboTwin 자산 준비와 실행 조건은 [실험 가이드](docs/experiments.md)를 확인하세요.
공식 코드의 revision은 [scripts/upstreams.json](scripts/upstreams.json)에 고정합니다.

현재 제공하는 기능:

- 제어 step 단위의 언어 정정 스케줄과 새 목표의 독립적인 성공 판정
- 즉시 chunk 취소(`cancel`)와 기존 chunk 소진(`drain`) 비교
- LIBERO의 최초 prompt 임베딩 재사용을 피하고, 매 재계획에 현재 지시 전달
- RoboTwin에서 장면을 재시작하지 않고 left/right 목표 전환
- seed·초기 상태 hash·정정 시각이 같은 paired 비교와 bootstrap 신뢰구간
- action provenance, 추론/제어 지연, CUDA peak allocated/reserved 메모리 기록
- 커스텀 WAM·압축 변환·성공 판정·물리적 실패 계측을 위한 Python hook
- W&B offline/online 기록, tqdm 진행률, MP4·PNG·로컬 HTML 리포트

INT4/INT8 및 token pruning은 모델에 맞는 `model_transform`을 연결해야 합니다. 실행 횟수 감소를 token pruning이나 양자화로 표기하지 않습니다. 양팔 충돌·낙하·물리적 정지 시간은 해당 센서를 구현하기 전까지 `null`로 기록합니다.

## 연구 자료와 구현 범위

- [Light-WAM 공식 코드](https://github.com/L1ziang/Light-WAM)
- [LIBERO 공식 코드](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [RoboTwin 2.0 공식 코드](https://github.com/RoboTwin-Platform/RoboTwin)
- [실험 설계와 지표 정의](docs/protocol.md)
- [어댑터와 후속 구현 지점](docs/extensions.md)

코드·문서는 MIT 라이선스입니다. 외부 프로젝트, 모델과 데이터는 각자의 라이선스를 따릅니다.
