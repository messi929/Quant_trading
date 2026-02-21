# Next Steps - 다음 작업 로드맵

**최종 업데이트**: 2026-02-21 (Phase 3 완료)
**현재 성과**: Sharpe 1.94, MDD -1.26% (Phase 3 기준)
**목표**: 달성됨. 다음 목표: Walk-forward 검증으로 과적합 여부 확인

---

## 완료된 작업 (Phase 3~4, 2026-02-21 04:37)

### [x] RL simplex 제약 (`models/rl/environment.py`)
- action을 long-only simplex에 투영 → 레버리지 제거

### [x] 보상 클리핑 (`models/rl/environment.py`)
- `clip(-10, 10)` → V.Loss 50,000+ → 76, Entropy 15.6 → 9.7

### [x] GAN 조기 종료 (`models/gan/trainer.py`)
- `W-dist > best + 5.0` 시 조기 종료 → 다음 GAN 재훈련부터 적용

### [x] Top-K 신호 생성 (`main.py`)
- 상위 3개 섹터만 long, 나머지 neutral → MDD -25.9% → -1.26%

### [x] Phase 5 RL 재훈련 + 백테스트
- 결과: **Sharpe 1.94, MDD -1.26%, Return +38.8%, Calmar 46.4**

### [x] GAN 안정화 (Phase 4, 2026-02-21 04:37)
- Generator Spectral Normalization 추가 (`models/gan/model.py`)
- `n_critic` 5→10, `generator_hidden_dims` [128,256,128]→[64,128,64] (`config/model_config_fast.yaml`)
- Early stopping Epoch 6 작동, best W-dist 13.0→**7.17** 개선
- 재훈련 후 백테스트: **Sharpe 1.94 유지**

---

## 우선순위 1: 과적합 검증 (다음 할 일)

### [ ] Walk-forward 백테스트 실행
현재 성과가 과도하게 낙관적일 수 있음 (MDD 1.26%는 비정상적). Walk-forward로 검증.

```bash
/c/Users/wogus/miniconda3/envs/quant/python.exe main.py backtest --walk-forward --config config/settings_fast.yaml
```

**확인 항목**:
- 각 기간별 성과가 일관적인가?
- 특정 기간에만 수익이 집중되어 있는가?
- MDD가 정말 1% 수준인가, 아니면 대부분 현금 보유인가?

---

## Phase 3: GAN 근본적 안정화 (우선순위: 높음)

### 문제 요약
- Epoch 1이 항상 best (W-dist 13→40+으로 발산)
- lr 5e-5로 줄여도 개선 없음

### 해결 방안 (우선순위 순)

#### [ ] 3-1. GAN 조기 종료 구현 (30분)
`models/gan/trainer.py`에서 best W-dist 기준 early stopping 추가:
```python
# W-dist가 best에서 5 이상 벌어지면 중단
if current_w_dist > best_w_dist + 5.0 and epoch > 5:
    logger.info(f"Early stopping at epoch {epoch}")
    break
```
→ **항상 best epoch1 모델 사용되도록 보장**

#### [ ] 3-2. n_critic 증가 (5→10)
`config/model_config_fast.yaml`:
```yaml
gan:
  n_critic: 10  # 5 → 10 (Discriminator 더 많이 학습)
```
Discriminator가 더 강해지면 Generator 발산 억제.

#### [ ] 3-3. Generator 구조 축소 (과적합 방지)
```yaml
gan:
  generator_hidden_dims: [64, 128, 64]  # [128, 256, 128] → 축소
```

#### [ ] 3-4. Spectral Normalization 추가 (복잡, 1-2시간)
`models/gan/model.py` Generator에 `torch.nn.utils.spectral_norm()` 적용.
Generator의 Lipschitz 조건을 직접 강제.

---

## Phase 4: RL 에이전트 개선 (우선순위: 높음)

### 문제 요약
- SectorTradingEnv에서 portfolio_value가 66조원까지 폭발
- Evaluation MDD 102% (거의 전재산 손실)

### 해결 방안

#### [ ] 4-1. SectorTradingEnv 포지션 합 제한 (30분)
`models/rl/environment.py` - step() 함수에서 action 정규화:
```python
# action을 simplex에 투영 (합이 1이 되도록, 음수 없음)
action = np.clip(action, 0, None)
if action.sum() > 0:
    action = action / action.sum()
else:
    action = np.ones(self.n_sectors) / self.n_sectors
```
→ 레버리지 제거, portfolio_values 폭발 방지

#### [ ] 4-2. RL 보상 정규화 (30분)
현재 reward 스케일이 너무 커서 value loss가 50,000+. 보상 클리핑:
```python
reward = np.clip(reward, -10.0, 10.0)
```

#### [ ] 4-3. entropy_coef 추가 감소 실험
현재 0.001에서도 entropy 10+. 0.0001 시도 또는 entropy schedule (점진적 감소) 구현.

#### [ ] 4-4. RL 평가 환경 개선
`models/rl/trainer.py` evaluate()에서 portfolio_value 리셋 확인:
- 에피소드 시작마다 initial_capital로 리셋되어야 함

---

## Phase 5: Transformer 신호 품질 개선 (우선순위: 중간)

### 문제 요약
- Direction accuracy 52.4% (무작위 수준)
- Rank IC 0.094 (낮음, 0.05 이상이 의미 있음)

### 해결 방안

#### [ ] 5-1. 피처 중요도 분석
어떤 피처가 예측력이 있는지 분석:
```python
from indicators.evaluator import IndicatorEvaluator
# IC로 피처별 예측력 랭킹
```

#### [ ] 5-2. 예측 horizon 실험
현재 5일 → 1일, 3일 비교. 짧은 horizon이 더 예측 가능할 수 있음.
`config/model_config_fast.yaml`:
```yaml
transformer:
  prediction_horizon: 1  # 5 → 1
```

#### [ ] 5-3. 학습 데이터 증가
현재 5년치 → 10년치 또는 20년치 데이터 수집:
```bash
python main.py train --config config/settings.yaml  # history_years=20
```

#### [ ] 5-4. Transformer 구조 확대
```yaml
transformer:
  d_model: 256  # 128 → 256
  n_encoder_layers: 6  # 4 → 6
  epochs: 50  # 25 → 50
```

---

## Phase 6: 백테스트 신호 생성 방식 개선 (우선순위: 중간)

### 문제 요약
현재 `cmd_backtest`에서 모든 예측을 softmax 정규화 → 항상 100% long.
음수 예측(하락 예상 섹터)도 long 포지션이 됨.

### 해결 방안

#### [ ] 6-1. 음수 신호를 short/neutral로 처리
`main.py` cmd_backtest 신호 생성 부분:
```python
# 현재: shifted = row - row.min() + 1e-8 (항상 양수화)
# 개선: 상위 섹터만 long, 나머지는 0 (neutral)
top_k = 3  # 상위 3개 섹터만 투자
indices = np.argsort(row)[-top_k:]
weights = np.zeros_like(row)
weights[indices] = row[indices]
weights = np.clip(weights, 0, None)
if weights.sum() > 0:
    weights = weights / weights.sum()
```

#### [ ] 6-2. Walk-forward 백테스트 실행
```bash
/c/Users/wogus/miniconda3/envs/quant/python.exe main.py backtest --walk-forward --config config/settings_fast.yaml
```

---

## Phase 7: 발견된 지표 분석 (우선순위: 낮음, 흥미로운 연구)

#### [ ] 7-1. VAE 잠재 공간 시각화
```python
from indicators.generator import IndicatorGenerator
from indicators.evaluator import IndicatorEvaluator

gen = IndicatorGenerator(vae_model, n_indicators=32)
indicators = gen.generate(data)

eval = IndicatorEvaluator()
scores = eval.evaluate_all(indicators, returns)
# IC, 신규성, 단조성 점수 확인
```

#### [ ] 7-2. 발견된 지표 해석
- 어떤 원시 피처와 상관관계가 높은가?
- 기존 알려진 지표(RSI, MACD 등)와 유사한 지표가 발견됐는가?
- 정말 새로운 지표는 어떤 것인가?

---

## Phase 8: 인프라 및 실전 준비 (장기)

#### [ ] 8-1. 데이터 수집 최신화
```bash
# 최신 데이터 업데이트 (주기적 실행)
/c/Users/wogus/miniconda3/envs/quant/python.exe main.py train --config config/settings.yaml
```

#### [ ] 8-2. 실험 추적 도입
- MLflow 또는 Weights & Biases로 실험 이력 관리
- 하이퍼파라미터 → 백테스트 성과 매핑

#### [ ] 8-3. 모델 버전 관리
- 날짜별 체크포인트 폴더 구분 (현재 덮어쓰기)
- 최고 성과 모델 별도 보관

#### [ ] 8-4. 추론 파이프라인 검증
```bash
/c/Users/wogus/miniconda3/envs/quant/python.exe main.py infer --config config/settings_fast.yaml
```

---

## 예상 성과 개선 경로

| Phase | 핵심 변경 | 예상 Sharpe | 예상 MDD | 소요 시간 |
|-------|---------|------------|---------|---------|
| 현재 (Phase 2) | entropy + RiskManager | **0.62** | **-25.9%** | - |
| Phase 3 (GAN 조기종료 + 4-1 RL 포지션 합 제한) | 빠른 수정 | 0.7-0.8 | -20% | ~1시간 |
| Phase 4-5 (Transformer 개선 + RL 정상화) | 재훈련 필요 | 0.9-1.0 | -18% | ~4시간 |
| Phase 6 (신호 생성 개선) | 백테스트 개선 | 1.0+ | -15% | ~30분 |

---

## 핵심 명령어 모음

```bash
PYTHON=/c/Users/wogus/miniconda3/envs/quant/python.exe

# 빠른 RL만 재훈련 (GAN/Transformer 체크포인트 유지)
$PYTHON main.py train --start-phase 5 --config config/settings_fast.yaml

# GAN부터 재훈련
$PYTHON main.py train --start-phase 4 --config config/settings_fast.yaml

# 전체 재훈련 (VAE부터)
$PYTHON main.py train --start-phase 2 --config config/settings_fast.yaml

# 백테스트
$PYTHON main.py backtest --config config/settings_fast.yaml

# Walk-forward 백테스트
$PYTHON main.py backtest --walk-forward --config config/settings_fast.yaml

# 진행 중인 훈련 로그 모니터링
tail -f C:/Users/wogus/AppData/Local/Temp/phase4_retrain.log
```
