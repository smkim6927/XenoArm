# 현재 클라우드 검증 기록

2026-11-10, CPU 클라우드, Python 3.10.22, MuJoCo 3.3.2, PyTorch 2.7.1+cpu.

| 검증 | 결과 |
| --- | --- |
| frozen 설치 스크립트 재실행 | 통과 |
| Ruff | 통과 |
| pytest | 33 passed, 0 skipped |
| MuJoCo correction smoke | cancel/drain 각 3 seed 성공 |
| 동일 seed의 paired 비교 | 통과; 초기 상태 hash 일치 |
| 기존 chunk 출처의 action 비율 | drain 0.375, cancel 0.0 |
| 실제 LIBERO 초기화·반복 reset | 통과; 같은 seed의 simulator state 일치 |
| 실제 LIBERO Panda 움직임 | 30 control step과 agentview/wrist 합성 MP4 확인 |
| 실제 LIBERO 카메라 | agentview와 wrist, 각 256×256 RGB 확인 |
| 변경된 LIBERO BDDL predicate | 실제 장면의 bowl/ramekin에 대해 평가 성공 |
| wheel / sdist | 빌드 통과; MuJoCo XML 포함 |
| W&B offline | 실제 SDK run, step/episode 지표와 MP4 media 파일 저장 확인 |
| tqdm | episode/step 진행률과 정정 표시 확인 |
| MP4 | 실제 MuJoCo RGB 영상 인코딩·디코딩, frame 수와 이동 확인 |
| HTML/PNG report | 로컬 영상 재생 링크, 지시 version/지연 그래프와 집계 생성 |
| Chromium report 확인 | 실제 MP4 로드·seek 확인, page error와 외부 요청 없음 |
| W&B online | 미실행: 계정 인증과 네트워크 접근 필요 |
| Light-WAM 체크포인트 추론 | 미실행: CUDA GPU와 가중치 필요 |
| RoboTwin SAPIEN rollout | 미실행: GPU·자산·planner 환경 필요 |
| INT4/INT8·token pruning | 미구현: 모델별 transform hook을 연결해야 함 |

MuJoCo smoke는 규칙 정책의 제어 검증이며 연구 성능 비교가 아닙니다. 실제 LIBERO 환경 검사는 조작 task 성공을 의미하지 않습니다. Hugging Face 모델 저장소 접근은 현재 proxy에서 HTTP 403으로 거부되었습니다. 전체 모델 평가에는 GPU 환경, 다운로드 접근, 체크포인트·학습 config·통계 파일과 RoboTwin 자산을 추가로 준비해야 합니다.

실행 명령은 README와 docs/experiments.md에 있습니다. 실제 episode 결과와 메타데이터는 ignored 경로 results/에 보관하며 연구 결과로 커밋하지 않습니다.
