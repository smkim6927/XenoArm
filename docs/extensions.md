# 확장 인터페이스

## 사용자 정책

```json
{"backend": "plugin", "factory": "my_policy:build", "checkpoint": "/path/to/model.pt"}
```

factory는 config를 받아 아래 API의 객체를 반환합니다. 사용자 모듈의 코드가 실행됩니다.

```python
class Policy:
    def reset(self, seed): ...
    def predict(self, observation, instruction): ...  # nonempty float [T, action_dim]
    def on_instruction(self, instruction): ...       # optional: invalidate language/history caches
    def synchronize(self): ...                      # optional: synchronize accelerator
    def resources(self): ...                        # optional: measured resource fields
```

LIBERO는 7차원 normalized OSC action, RoboTwin은 해당 embodiment의 qpos vector입니다. 정책의 정규화·행동 공간을 일치시켜야 합니다. 컨트롤러가 chunk queue를 소유하므로 정책 안에 중복 queue를 두지 않습니다.

## 압축 hook

```json
{"model_transform": "my_compression:apply", "transform_options": {"keep_ratio": 0.5}}
```

`apply(model, options)`는 실제로 변환된 모델을 반환합니다. `infer_action`, dtype/device, action shape 계약을 유지하고 모델 고유의 attention mask와 cache를 함께 수정해야 합니다. correction-sensitive 보호 token/adapter의 후속 구현 위치입니다. 모델 로드 후 한 번 호출되며 실제 지원되는 GPU kernel과 총 메모리를 측정해야 합니다.

## 목표와 실패 센서

새 목표 없는 텍스트 변경을 correction 성공으로 집계하면 원래 작업의 성공을 오인할 수 있습니다. LIBERO는 BDDL predicate를, RoboTwin Place A2B는 방향·평면거리·양쪽 gripper release를 검사합니다.

다른 작업의 goal에는 `{"kind": "plugin", "callable": "my_oracles:success", ...}`를 지정합니다. callback `(raw_libero_env_or_robotwin_task, goal)`은 bool을 반환합니다. Handover/Lift Pot은 목표 변경의 물리적 정의와 안전 유지 조건을 별도로 구현해야 합니다.

`environment.diagnostics_hook`에는 `"my_oracles:diagnostics"`를 지정합니다. callback은 XenoArm env를 받아 simulator contact·물체 높이·파지 상태를 검사하고 측정한 값만 반환합니다:

```python
def diagnostics(env):
    return {"collision": None, "object_drop": None, "coordination_failure": None}
```

미계측 None을 False로 바꾸어 실패율 0을 만들지 않습니다. 정상 접촉과 실패 충돌의 body-pair 필터, drop 기준 높이, 양팔 협응 허용 오차를 연구 프로토콜에 명시해야 합니다.
