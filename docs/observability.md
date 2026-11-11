# 실험 진행률과 시뮬레이터 결과 확인

## 기본 실행

```bash
bash scripts/setup.sh libero
.venv/bin/xenoarm run --config configs/smoke.json \
  --output results/visual-run --progress --wandb offline --video
```

`results/visual-run/report.html`을 브라우저로 열면 각 seed의 영상과 그래프를 확인할 수 있습니다. 외부 CDN이나 웹서버를 사용하지 않으며 폴더 전체를 복사해도 상대 경로의 영상·그래프가 유지됩니다. stdout에는 episode별 JSON 요약과 리포트 경로가 출력됩니다.

`--progress`는 tqdm episode/step bar를 강제로 표시합니다. 기본은 터미널에서 자동 표시하고, 파일/CI 로그에서는 생략합니다. `--no-progress`로 끌 수 있습니다. step bar의 총량은 max_steps 예산이며 목표를 일찍 달성하면 예산을 채우지 않고 종료됩니다.

## 영상과 그래프

실제 simulator가 반환한 RGB를 기록합니다:

- MuJoCo smoke: overview 카메라. 규칙 정책의 제어 검증용 장면입니다.
- LIBERO: agentview와 wrist를 공식 Light-WAM 전처리와 같은 방향으로 회전해 나란히 표시합니다.
- RoboTwin: head, left wrist, right wrist. 실제 재생 검증에는 GPU와 RoboTwin 자산이 필요합니다.

영상 상단의 command version은 최신 지시, action version은 실행 action이 생성된 지시를 뜻합니다. 이전 version의 action은 STALE ACTION으로 표시됩니다. 리포트 그래프의 수직 점선은 정정 시점이며 주황 영역은 이전 chunk 실행 구간입니다. step/version은 지시 인과관계 확인용이고 실제 의미적 정정 성공은 별도의 목표 predicate로 평가합니다.

`--video-episodes 2`는 처음 2개만 녹화하고 `--video-stride 4`는 4 step마다 기록합니다. 정정 직후와 마지막 step은 stride와 관계없이 남깁니다. 첫 frame은 reset 상태입니다. 기본 녹화는 off이고, 켜면 처음 3개 episode를 기록합니다. 동영상은 스트리밍 인코딩해 모든 RGB frame을 메모리에 쌓지 않습니다.

기존 JSON config에 다음 설정을 추가할 수 있습니다:

```json
{
  "progress": true,
  "tracking": {"mode": "offline", "project": "xenoarm", "group": "lightwam-libero"},
  "visualization": {
    "enabled": true, "width": 960, "height": 540,
    "fps": 20, "frame_stride": 1, "max_episodes": 3
  }
}
```

폭·높이는 짝수여야 합니다. `fps`는 저장 영상 재생 속도이며 측정된 robot/control wall time이 아닙니다. 녹화와 W&B logging의 step별 비용은 `instrumentation_seconds`에 기록하며 `inference_ms`와 dispatch 전 `control_ms`에서 제외합니다. episode wall time에는 초기 녹화와 step별 instrumentation 비용이 포함되고 최종 파일 flush·리포트 생성은 포함되지 않습니다. 지연 benchmark에는 녹화를 끄고 장치·warmup을 통제하세요.

출력 폴더에는 `episodes.jsonl`, `metadata.json`, `summary.json`, `report.html`, `plots/`, 선택적으로 `videos/`, `wandb/`가 생성됩니다. 기존 결과의 리포트만 다시 만들 때:

```bash
.venv/bin/xenoarm report results/visual-run
```

## W&B

기본값은 disabled로 네트워크나 인증을 요구하지 않습니다. `--wandb offline`은 실제 SDK로 로컬 run을 기록하며 자동 업로드하지 않습니다. control step, command/action version, stale 여부, 추론/제어 지연, 성공 여부, CUDA peak memory, 정정 후 dispatch 지연과 녹화 영상을 기록합니다. 미계측 null을 성공/0으로 바꾸지 않습니다.

online 실행은 계정과 허용된 목적지가 준비된 뒤 선택합니다:

```bash
.venv/bin/xenoarm run --config configs/smoke.json --output results/wandb-online \
  --wandb online --wandb-project xenoarm --wandb-entity YOUR_TEAM --video
```

API 키는 환경 설정의 `WANDB_API_KEY` 또는 기존 `wandb login` 설정으로 공급합니다. config, 명령행 인수, 커밋에 키 값을 넣지 않습니다. 이 클라우드에서는 현재 offline 모드만 검증했습니다. online에는 `api.wandb.ai`와 실제 업로드 경로의 storage 목적지 허용이 필요할 수 있습니다. 네트워크/인증 오류는 실행 실패로 기록합니다.

offline run을 나중에 동기화할 때는 SDK가 출력한 실제 경로로 실행합니다:

```bash
.venv/bin/wandb sync results/visual-run/wandb/offline-run-YYYYMMDD_HHMMSS-RUNID
```

## GUI 없는 클라우드

`--video`는 display가 없을 때 MuJoCo import 전에 EGL backend를 선택합니다. 직접 Python으로 env를 먼저 import하거나 `scripts/libero_smoke.py`를 사용할 때는 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`을 지정하세요. 녹화는 EGL/OpenGL 라이브러리를 요구합니다. 렌더링 실패를 빈 영상으로 대체하지 않습니다.

영상에 한국어 지시를 표시하려면 CJK 글꼴이 필요합니다. 설치된 Noto Sans CJK를 우선 사용하고, `XENOARM_FONT`에 원하는 글꼴 파일 경로를 지정할 수 있습니다. LIBERO 카메라 검증은 모델 없이 `scripts/libero_smoke.py --video-output <새 폴더>`로 실행합니다. 이 영상은 WAM 정책 성능의 증거가 아닙니다.
