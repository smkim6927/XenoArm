# Correction 실험 프로토콜

H1: 경량화가 일반 작업 성공보다 중간 언어 정정 후 성공을 더 크게 손상시킬 수 있다.
H2/H3의 correction-sensitive representation 분석과 보호 방법은 H1 관측 후의 후속 작업입니다.
이 저장소의 smoke 성공은 연구 가설의 증거가 아닙니다.

## 실험 통제

1. 모델·체크포인트·precision·계산 budget·replan horizon을 기록합니다.
2. 같은 simulator seed, 초기 상태, 지시, corrected goal, 정정 step을 유지합니다.
3. 정정 없는 작업, chunk 소진 조건, 즉시 취소 조건을 구분합니다.
4. goal change에는 새 물리적 성공 predicate를 필수로 지정합니다. paraphrase는 목표를 유지합니다.
5. 여러 정정의 최종 correction success는 마지막 목표의 달성을 뜻합니다.

`pairing_key`는 정책을 제외한 장면·목표·스케줄·seed를 포함하며 실제 초기 state hash도 검사합니다. `metadata.json`에는 설정, 설치 버전, checkout SHA/dirty 상태, checkpoint/config/stats SHA-256을 기록합니다. 실패한 실행이나 일부 episode만 있는 결과는 완료된 평가로 읽지 않습니다.

## 지표 정의

| 필드 | 의미 |
| --- | --- |
| original_task_success | 정정 없는 episode의 작업 성공; correction episode는 null |
| correction_success | 모든 예약 정정 후 마지막 목표를 연속 success_hold_steps 충족 |
| original_goal_reached_before_correction | 첫 정정 전 원래 목표가 한 번 이상 참이었는지 |
| stale_action_rate | 첫 정정 이후 action 중 이전 command version chunk에서 나온 비율 |
| correction_to_dispatch_ms | 지시 수신부터 현재 version action의 simulator 전달 직전까지 시간 |
| first_fresh_step | 현재 version의 첫 action을 dispatch한 step |
| inference_p95_ms | CUDA synchronize를 포함한 chunk 예측 시간의 P95; episode별 집계 |
| control_p95_ms | 지시 처리·관측·필요한 추론을 포함한 dispatch 전 제어 시간 P95 |
| gpu_peak_allocated_bytes / reserved | episode별 PyTorch CUDA peak; 전체 GPU 메모리와 다름 |
| collision / object_drop / coordination_failure | diagnostics hook이 전체 step을 계측했을 때 집계; 기본 null |

stale은 action의 출처입니다. 수치적으로 같은 action이 두 지시에 모두 적합할 수 있으므로 의미적 오류율과 같지 않습니다. 지연시간은 장치·warmup·batch·카메라 수를 통제하고 보호 경로 메모리도 예산에 포함합니다.

`correction_to_physical_stop_ms`, `correction_to_physical_action_ms`는 미계측으로 null입니다. 새 action dispatch와 물리적 정지/방향 전환은 다릅니다. API 수준 취소는 양팔 공동 지지의 안전 유지 동작을 구현한 것이 아닙니다.

runner는 동기식 추론과 step 기반 정정을 사용합니다. inference 중 도착한 비동기 입력을 선점하지 않고 simulator는 추론 동안 멈춥니다. 실제 real-time 반응성에는 비동기 입력, wall-clock delivery, device-side 취소/hold 경로가 추가로 필요합니다.

## 초기 범위

goal-change 예시는 LIBERO Spatial 1개와 RoboTwin Place A2B 1개입니다. 다른 장면은 suite/task·지시·predicate를 함께 확장합니다. LIBERO Goal/Spatial 각 4개, RoboTwin Handover Block/Lift Pot은 추가 시나리오·정정 oracle·협응/낙하 센서를 작성하고 검증한 뒤 포함해야 합니다. 정정 step 변경만으로 grasp/contact phase를 통제했다고 주장하지 않습니다.

INT4 kernel, token-pruned Light-WAM, 제안한 보호 adapter와 실제 양팔 안전 제어는 이 버전의 구현 범위에 포함하지 않습니다.
