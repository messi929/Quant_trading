# Quant Trading System — CORE

본 문서는 Vol Expansion Trader **V3.2.1**(2026-05-08 기준)의 전략 설계,
수식·모델 디테일, 코드 아키텍처, 성과·한계를 한 곳에 모은 **토론용 reference**다.
운영 규칙(Claude가 따라야 할 행동 규약)은 `CLAUDE.md`,
변경 이력은 `docs/CHANGELOG.md`,
보류된 후속 과제는 `docs/FOLLOW_UPS.md`에 분리.

이 문서를 읽는 사람이 "이 전략이 합리적인가? 어디를 어떻게 손볼 수 있는가?"
판단할 수 있도록 thesis → 수식 → 코드 → 데이터를 순서대로 펼친다.

---

## 1. 30초 요약

- **무엇을 베팅하나**: NASDAQ 종목의 "다음 5거래일 변동성 팽창 + 방향 정렬"이
  거래비용을 충분히 상회하는 케이스만 진입.
- **왜 작동하리라 보나**: 변동성 팽창은 IC 0.70(테스트 OOS 0.7486)로 **강한
  교차단면 신호**가 존재. 변동성은 방향 정보가 없어 risk model이지만,
  방향 알파(trend, reversion)와 곱해주면 진입 후보를 좁히는 강력한 confidence
  multiplier가 된다.
- **어떻게 거래하나**: 단일 수식 `opportunity = direction × conviction`이
  비용·게이트(`cost × 1.75`)를 넘으면 진입. 월 5회 이내, 1~3종목 집중.
  bear regime에선 100% 현금.
- **현재 상태**: 백테스트 Sharpe 1.65 / Return +38% / MDD 4%. NASDAQ paper
  4/11~5/3 실측 +1.39%(승률 71%, 손익비 12:1). 진입 빈도 부족이 1차 병목으로
  Phase 25.2(2026-05-07)에서 sizer floor 0.05 → 0.15 인상.

---

## 2. 투자 철학

> **"확신 있을 때만, 크게, 빠르게"**
> — 작은 edge × 반복 × Kelly sizing = 수익

세 원칙이 상호 강화한다.

### 원칙 1. 확신 없으면 안 산다 (Conviction-or-Cash)

- 모델 신뢰도 × 알파 방향 × 거래비용 게이트가 모두 통과해야 진입.
- 0~100% 현금 허용. "항상 투자해야 한다"는 전제를 거부.
- Two Sigma의 "sigma-based amplification"을 적용: 신뢰도에 비례해서만 사이징.

### 원칙 2. 집중 투자한다 (Concentration over Diversification)

- 1억원으로 9종목에 분산하면 종목당 1,100만원 → 2~3% 수익이 비용·세금에
  잠식돼 의미 없음.
- 확신 있는 1~3개 종목에 집중 → 종목당 30%+ 비중 → 의미 있는 절대 수익.
- `max_single_weight = 0.40`, `max_positions = 3` (paper 검증 단계는 1).

### 원칙 3. 빠르게 청산한다 — 단, "시간"이 아닌 "thesis"

- V3 재해석: "빠르게" = "thesis 깨지면 빠져나오기"이지 "시간 됐으니 팔기"가
  아니다.
- 리스크 기반 청산(`vol_contraction`, `dynamic_stop_mae`, `portfolio_stop`)이
  실질적 구현체. 시간 기반(`profit_take`, `max_hold`)은 opportunity 재평가
  트리거로 격하.
- 회전율 자체가 수익 공식이 아니다: positive expectancy × 반복.

### 수익 목표

- **월 평균 일 +1%** = 월 +20%, 거래일 기준. **매일** 달성이 아닌 **월 누적**.
- 승률 55%+ × 손익비 1.5:1 이상 → 위 목표 가능.
- paper 4/11~5/3 실측: 승률 71%, 손익비 12:1 (페르소나 통과), 사이즈 부족이
  병목.

---

## 3. 전략 thesis — 왜 변동성 팽창인가

### V2 폐기 이유 (2026-04 post-mortem)

V2는 OHLCV 기반 일간 수익률 순위를 예측했다. 결과:

| 항목 | V2 측정값 | 의미 |
|------|-----------|------|
| 모델 IC (수익률 예측) | 0.04 | 거의 노이즈와 구분 안 됨 |
| KRX 왕복 비용 | 1.0% | 매일 거래 시 연 240% |
| 7일 paper 결과 | -6.91% | 모든 전략 마이너스 |

**핵심 진단**: "수익률 순위 예측"은 IC가 너무 낮아 어떤 모델 아키텍처를
써도 거래비용을 못 이긴다. 문제는 모델이 아니라 **예측 대상**이었다.

### V3 전환 — 4가지 전제 변경

| 전제 | V2 | V3 | 효과 |
|------|-----|-----|------|
| 무엇을 예측 | 수익률 순위 (IC=0.04) | **변동성 팽창** (IC=0.70) | 17.5× 강한 신호 |
| 언제 거래 | 매일 (연 240% 비용) | **월 5회 이하** (연 6%) | 비용 40× 감소 |
| 어디서 거래 | KRX (왕복 1%) | **NASDAQ** (왕복 0.1%) | 비용 10× 저렴 |
| 무엇을 보고 | OHLCV 가격만 | OHLCV + vol 구조 + macro regime + 수급 | 다차원 |

### 변동성 + 방향 분리 (Two Sigma 컨벤션)

핵심 통찰: 변동성 자체는 방향 정보가 없다.
- VolTransformer dir IC ≈ 0 (signed return과 무관)
- 직접 수익률 예측에 부적합
- 하지만 **다른 알파의 confidence를 modulate**하면 강력함

이를 두 축으로 분리한 것이 Phase 26 재설계의 핵심:

| 축 | 역할 | 출력 범위 | 현재 구성 |
|----|------|----------|----------|
| **DirectionalAlpha** | 수익률 예측 (signed) | [-0.1, 0.1] | `trend`, `reversion` |
| **ConvictionSource** | 확신도 (unsigned) | [0, 1] | `vol` (VolTransformer) |

**원칙**: VolTransformer는 risk model이지 alpha model이 아니다. 변동성 팽창은
"크기 예측"이지 "방향 예측"이 아니므로 signed return과 IC ≈ 0. 직접 수익률
예측에 합치 않고 다른 알파의 확신도를 modulate한다.

---

## 4. 핵심 수식 — Opportunity Gate

V3.2의 모든 진입 결정은 단일 수식으로 표현된다.

```
direction(t)   = Σ_a  w_a(regime) · α_a(t)        ∈ [-0.1, 0.1]
conviction(t)  = Π_c  c_s(t)                       ∈ [0, 1]
opportunity(t) = direction · conviction            ∈ [-0.1, 0.1]

enter_if  opportunity > cost × k       # cost = 0.001 (NASDAQ 왕복)
                                       # k = 1.75 (gate multiplier)
                                       # gate = 0.00175
```

구현: `v3/strategy/opportunity.py:62` (`OpportunityScorer.score`).

### 수식 해석

- `direction`: 회귀형 가중합 — 알파별 점수의 regime-conditional 가중평균.
  Two Sigma의 "다중 알파 선형 결합" 컨벤션.
- `conviction`: 곱(product) — 모든 confidence source가 동시에 높아야 큰
  값. 한 source가 0이면 전체 0 → "위험 신호 하나라도 있으면 안 한다".
- `cost × 1.75`: 비용 회수 + 안전 여유. 1.75는 Kelly 1/2 + slippage
  variance를 고려한 휴리스틱.

### 단일 게이트의 의의 (Phase 2 원칙)

V3.1까지는 10개 조건(C1~C10)이 흩뿌려져 있었다:

```
C1 vol_score ≥ 0.05
C2 confidence ≥ 0.30
C3 direction_clarity ≥ 0.20
C4 direction == "long"
C5 monthly_trades < dynamic_max
C6 current_positions < max_positions
C7 circuit_breaker OFF
C8 expected_move > cost × 1.75
C9 ticker_volume ≥ min_volume
C10 sector_concentration ≤ 2
```

V3.2에서 알파 관련(C1/C2/C3/C4/C8)을 `opportunity > cost × k` 단일 수식에
통합. 운영 제약(C5/C6/C7/C9/C10)만 EntryFilter에 분리.

**효과**: regime이 threshold를 동적 수정하던 뮤테이션이 사라지고, regime은
오직 `alpha_weights`만 변경 (Phase 26 원칙 3 "Regime은 알파 가중치 선택자").

---

## 5. Regime 시스템

### 5단계 분류 + 연속 score

`v3/strategy/regime_v2.py:41-47`:

```python
REGIME_THRESHOLDS = [
    (0.75, "strong_bull"),
    (0.55, "bull"),
    (0.40, "neutral"),
    (0.25, "caution"),
    (0.00, "bear"),
]
```

Composite score는 8개 macro 피처의 가중 percentile rank:

| Feature | Weight | Sign | 의미 |
|---------|--------|------|------|
| `hy_level` | 0.249 | -1 | HY credit spread 수준 (낮을수록 risk-on) |
| `vix_ratio` | 0.222 | -1 | VIX / VIX 평균 (낮을수록 risk-on) |
| `breadth` | 0.159 | +1 | 시장 breadth (높을수록 risk-on) |
| `dxy_mom_60d` | 0.155 | -1 | 달러 모멘텀 (약달러 = risk-on) |
| `hyg_tlt_mom_60d` | 0.151 | +1 | HY 대 장기국채 상대강도 |
| `yc_slope` | 0.036 | +1 | 수익률곡선 기울기 |
| `hy_change_60d` | 0.020 | +1 | HY spread 변동 |
| `gold_spy_mom_60d` | 0.008 | -1 | Gold 대 SPY 상대강도 |

가중치는 alpha_weight_trainer.py가 stand-alone IC로 학습 (`alpha_weights.json`
의 `regime_composite.feature_weights`).

### Position Scale Curve — discrete 게이트, continuous 사이징

```
POSITION_SCALE_CURVE (piecewise linear):
  score 0.00 → 0.00  (bear = 100% CASH)
  score 0.25 → 0.30
  score 0.40 → 0.60
  score 0.55 → 0.90
  score 0.75 → 1.10
  score 1.00 → 1.20
```

**Bridgewater discrete + Medallion continuous 하이브리드**:
이름은 5단계 discrete bucket이지만 실제 size scaling은 score 기준 piecewise
linear. regime 경계(0.54 ↔ 0.56)에서 포지션 쇼크 없음.

### Hysteresis — 깜빡거림 방지

`hysteresis_days = 2`. raw regime이 바뀌어도 2거래일 연속 확인된 후에만
confirmed regime이 바뀐다. transition 중에는 confidence가 0.7~1.0 사이로
감쇠.

### Regime별 진입 — 게이트 고정, 가중치만 변경

`alpha_weights.json` (2026-04-18 학습, 3년 lookback, panel 14817 row):

| Regime | trend | reversion | 학습 근거 (conditional IC) |
|--------|-------|-----------|---------------------------|
| strong_bull | 0.0 | 1.0 | reversion IC=0.113 (유일 양성) |
| bull | 0.5 | 0.5 | 둘 다 IC < 0.02 → uniform fallback |
| neutral | 1.0 | 0.0 | trend IC=0.028 (유일 유의) |
| caution | 0.5 | 0.5 | uniform |
| bear | 0.5 | 0.5 | (CASH로 우회되어 사용 안 됨) |

**vanilla IC는 둘 다 게이트 미달**(trend 0.014, reversion -0.010). 그래도
조건부 IC는 strong_bull/neutral에서 유의미 → regime split이 정보를 보존.

---

## 6. 알파 소스 — Directional + Conviction

구현: `v3/strategy/alpha_sources.py`.

### AlphaTrend (Directional)

다중 lookback 모멘텀(5/20/60일) 평균 → 교차단면 z-score → tanh → ±0.1 스케일.

```python
returns_t = mean([close[-1]/close[-p-1] - 1 for p in (5, 20, 60)])
z_t       = (returns_t - cs_mean) / cs_std
α_trend   = tanh(z_t / 2.0) × 0.10   # ∈ [-0.1, 0.1]
```

### AlphaReversion (Directional)

20일 SMA 대비 z-score, **부호 반전** (overbought → 음의 알파).

```python
z = ((close - SMA20) / SMA20) / std_pct20
α_reversion = -tanh(z / 2.0) × 0.10
```

### VolConviction (Conviction)

VolTransformer가 출력한 `vol_score`(예측 5일 vol expansion ratio)의
**교차단면 percentile rank** → [0, 1].

```python
conviction_vol = rank(vol_scores, pct=True)   # ∈ [0, 1]
```

VolTransformer 자체 학습 IC: vol expansion IC=0.7185 (Val), 0.6998 (Test),
Dir Acc 80.3% / 78.2%. **알파 모델이 아니라 risk/conviction model**.

### 확장 패턴

새 알파 추가는 `AlphaSource`(signed) 또는 `ConvictionSource`(unsigned)
abstract base만 구현하면 된다. 기존 코드 수정 없이 `DEFAULT_DIRECTIONAL` /
`DEFAULT_CONVICTION` 튜플에 추가.

후보(미구현, IC 게이트 통과 시 추가):
- `AlphaFlow` — 수급(외국인·기관 순매수)
- `AlphaSentiment` — 뉴스/SNS 감성
- `AlphaEvent` — 어닝, 가이던스, 거시 이벤트

---

## 7. 포지션 사이징 — VolTargetSizer

구현: `v3/strategy/sizing.py:16` (`VolTargetSizer`).

### 4단계 합성

```
1. Inverse-vol baseline:   w_i ∝ 1 / σ_i
2. Portfolio vol target:   scale = target_vol(0.15) / port_vol
3. Half-Kelly tilt:         edge = max((conf - 0.5) × 2, 0)
                            kelly_adj = 0.3 + 0.7 × min(0.5 × edge, 1.0)
4. Correlation drag:         drag = max(0.5, 1 - avg_corr × 0.3)
```

이후 `[min_position_weight, max_single_weight] = [0.15, 0.40]`로 클램프
하고, 합계 0.95 초과 시 비례 축소.

### Phase 25.2 사이저 floor 인상 (2026-05-07)

`min_position_weight 0.05 → 0.15` 인상. 이유:

- 4/27~5/7 13세션 연속 entries=0의 1차 단속점이 cap이 아니라 **floor**였음.
- caution regime 최저 scale 0.35에서도 5M 통과 보장(0.15 × 0.35 = 5.25M >
  `min_order_amount_krw` 5M).
- bull 1종목 13.5%, 3종목 균등 40.5% — 페르소나 "1~3종목 집중" 부합.

### 유동성 제약

```python
position_shares = (capital × weight) / price
max_shares = daily_volume × volume_participation_max(0.05)
if position_shares > max_shares: 비례 축소
```

Renaissance 5% 참여율 룰. 시장 충격 관리.

### Correlation은 Ledoit-Wolf shrinkage

```python
shrunk = (1 - 0.3) × raw_corr + 0.3 × I
```

소표본 correlation의 분산을 줄여 portfolio vol 추정을 안정화.

---

## 8. 청산 규칙 — V3.2.1 Conditional Veto

구현: `v3/rules/exit.py:28` (`ExitRules`).

### 5가지 조건 (우선순위 순)

| # | 이름 | 조건 | Veto 여부 |
|---|------|------|-----------|
| P1 | `portfolio_stop` | 일간 PnL ≤ -dynamic_limit | 무조건 |
| P2 | `dynamic_stop_mae` | MAE -3% trigger 후 종가 -2.5% | 무조건 |
| P3 | `profit_take` | 시간감쇠 TP 달성 | **Conditional** |
| P4 | `max_hold` | 보유 5일 만료 | **Conditional** |
| P5 | `vol_contraction` | vol 진입 대비 70% 이하 (3일 지속) | 무조건 |

### 시간감쇠 Profit Take

```python
TIME_DECAY_TARGETS = {
    0: 0.050,   # Day 1: +5.0%
    1: 0.040,   # Day 2: +4.0%
    2: 0.030,   # Day 3: +3.0%
    3: 0.020,   # Day 4: +2.0%
    4: 0.015,   # Day 5: +1.5%
}
```

신뢰도·vol regime에 따라 ±30% 추가 조정:
- conf > 0.75 → 0.7배 (빠르게 차익실현)
- conf < 0.45 → 1.3배 (충분한 보상 요구)
- vol 팽창 중(ratio > 1.2) → 0.85배
- vol 수축 중(ratio < 0.8) → 1.1배

### Conditional Veto 정책 (V3.2.1)

`profit_take`/`max_hold`가 트리거되면 **opportunity 재계산** 후:
- `opportunity > cost × 1.75` → **유지** (veto + 로그)
- 그 외 → 청산

이유: 시간 기반은 본질적으로 thesis 자체가 깨졌다는 신호가 아니다.
Day 5에 +1.5% 도달했어도 알파가 살아있으면 자르지 말자는 것이 V3 "빠르게"의
재해석.

**리스크 기반(P1, P2, P5)은 veto 금지** — 무조건 체결.

### Dynamic Daily Limit

배포 비례 stop:

```python
deployed > 0.90 → 1.0%
deployed > 0.70 → 1.2%
deployed > 0.30 → 기본 1.5%
deployed < 0.30 → 2.0%
```

deploy 클수록 타이트. paper 검증 중에는 `daily_loss_limit = 0.005`로
더 보수적으로 (v3.1-safe).

### MAE 스탑은 V2.2~V3 일관된 결론으로 디폴트 OFF

`use_mae_stop = false` (config). 백테스트 스윕 결과:

| 조합 | Return | Sharpe | MDD | Win |
|------|--------|--------|-----|-----|
| MAE -3% (baseline) | +5.2% | 0.65 | 4.4% | 61% |
| MAE -4% | +10.7% | 1.06 | 4.4% | 68% |
| **MAE 제거 + 진입완화** | **+38.3%** | **1.65** | **4.0%** | **64%** |

MAE 제거 시 Sharpe 0.65 → 1.65 (2.5×). vol 팽창 환경에서 tight stop은
노이즈 매도다.

(코드에는 P2가 살아있지만 config로 비활성. 향후 사용 가능성 보존.)

---

## 9. VolTransformer — 모델 아키텍처

구현: `v3/model/vol_transformer.py`. 2.26M 파라미터.

```
Input (B, 60, F)
  ↓ Linear(F → 192) + Dropout
  ↓ PositionalEncoding (sin/cos)
  ↓ TransformerEncoder × 5 layers
       (heads=8, d_ff=768, pre-norm, batch_first)
  ↓ Mean-pool over time          ← V2 lesson: better than last-token
  ↓ Linear(192 → 96) → GELU → Linear(96 → 1)   = prediction
  ↓ Linear(192 → 48) → GELU → Linear(48 → 1) → Sigmoid = confidence
```

### 학습 설정

```yaml
batch_size: 256
lr: 3e-5
weight_decay: 1e-5
scheduler: cosine_warmup (warmup_steps: 1000)
epochs: 60 (early stop patience 20)
loss: 0.4 × Huber + 0.6 × ListMLE (ranking 강화)
       + 0.1 × confidence_loss
```

### 배포 게이트

```yaml
min_vol_ic: 0.30
min_vol_rank_ic: 0.20
min_dir_acc: 0.55
min_backtest_sharpe: 1.0
```

게이트 미달 시 라이브 배포 금지. **게이트 완화하지 않는다** (V2.2 교훈:
완화한 게이트로 7일 만에 -6.91%).

### 현재 측정값 (NASDAQ-100, 5년 학습)

| 지표 | Val (Best E19) | Test (OOS) |
|------|----------------|------------|
| Vol IC | 0.7185 | 0.6998 |
| Vol Rank IC | 0.7896 | 0.7486 |
| Dir Accuracy | 80.3% | 78.2% |
| High Conf Acc | - | 85.4% |

---

## 10. 코드 아키텍처

### 디렉토리 구조

```
v3/
├── config/
│   ├── v3_config.yaml          # 단일 설정 파일 (백테스트/라이브 공유)
│   ├── schema.py                # V3Config 데이터클래스
│   ├── alpha_weights.json       # S2 학습 산출물 (regime별 가중치)
│   └── alpha_weights_history/   # 월별 버전
├── data/
│   ├── collector.py             # OHLCV (yfinance / pykrx)
│   ├── macro_collector.py       # FRED API + yfinance ETF
│   ├── feature_engineer.py      # vol/return 피처 71→35개
│   ├── macro_features.py        # 8개 macro 피처
│   ├── normalizer.py            # 학습셋 통계 z-score
│   └── universe.py              # NASDAQ-100 / KOSPI200
├── model/
│   ├── vol_transformer.py       # 2.26M params
│   ├── trainer.py               # ListMLE + Huber
│   └── inference.py             # VolInference (live + backtest)
├── strategy/
│   ├── alpha_sources.py         # Directional/Conviction protocol
│   ├── opportunity.py           # OpportunityScorer (단일 게이트)
│   ├── regime_v2.py             # RegimeDetectorV2 + POSITION_SCALE_CURVE
│   ├── sizing.py                # VolTargetSizer (vol-target + Kelly + corr)
│   ├── signal.py                # SignalGenerator (오케스트레이터)
│   ├── risk.py                  # RiskManager (서킷 브레이커)
│   └── _legacy/                 # Phase 1 단일자산 regime, cross-asset 패치
├── rules/
│   ├── direction.py             # (Phase 1 잔존, Phase 2에선 알파가 대체)
│   ├── entry.py                 # EntryFilter (운영 제약만)
│   └── exit.py                  # ExitRules (TP/MAE/contraction)
├── execution/
│   ├── broker.py                # KIS REST 추상화
│   ├── paper_broker.py          # yfinance 실시간 가격 시뮬
│   ├── position_manager.py      # 포지션 영속화 + 고스트 정리
│   └── executor.py              # 주문 라이프사이클
├── backtest/
│   ├── engine.py                # SignalGenerator 재사용 이벤트 드리븐
│   ├── walk_forward.py          # fold별 재학습 (미실행)
│   ├── alpha_weight_trainer.py  # S2: 3-step bootstrap
│   ├── metrics.py               # Sharpe/MDD/PF/win
│   └── _legacy/
├── pipeline/
│   ├── data_pipeline.py
│   ├── train_pipeline.py
│   └── live_pipeline.py         # E2E 라이브 (Phase 2 단일 경로)
├── scripts/                     # CLI entry points
└── tests/
    └── test_regression.py       # 회귀 invariant
```

### 데이터 플로우 — 단일 캐노니컬 경로

```
                    ┌─────────────────────────────┐
                    │  yfinance / pykrx / FRED    │
                    │  (OHLCV, macro ETF, 매크로)  │
                    └──────────────┬──────────────┘
                                   ↓
                    ┌─────────────────────────────┐
                    │  VolFeatureEngineer (35 col) │
                    │  MacroFeatureEngineer (8)    │
                    │  FeatureNormalizer (z-score) │
                    └──────────────┬──────────────┘
                                   ↓
        ┌──────────────────────────┼──────────────────────────┐
        ↓                          ↓                          ↓
  VolTransformer            MacroPercentile             OHLCV (raw)
  inference                 rolling 5y                  for alphas
        │                          │                          │
        ↓                          ↓                          ↓
   vol_scores             RegimeDetectorV2           AlphaTrend +
   (vol_score,            .detect()                  AlphaReversion
    confidence)                    │                          │
        │                          │                          ↓
        ↓                          ↓                  directional DF
   VolConviction         Regime(name, score,         (ticker × alpha)
   (rank pct)            alpha_weights,
        │                position_scale)                      │
        └──────┐         ┌─────────┴──────────────────────────┘
               ↓         ↓
           ┌───────────────────────┐
           │  OpportunityScorer    │  opportunity = direction × conviction
           │  .score()             │  passes if  > cost × 1.75
           └──────────┬────────────┘
                      ↓
           ┌───────────────────────┐
           │  EntryFilter (5 ops)  │  positions, monthly cap, CB,
           │                       │  liquidity, sector concentration
           └──────────┬────────────┘
                      ↓
           ┌───────────────────────┐
           │  VolTargetSizer       │  inverse-vol × Kelly × corr drag
           │  + position_scale     │  + clamp + normalize
           └──────────┬────────────┘
                      ↓
              TradeSignal (frozen)
                      ↓
        ┌─────────────┴──────────────┐
        ↓                            ↓
   Backtest engine             Live executor
   (replay)                    (KIS / PaperBroker)
                                     │
                                     ↓
                          PositionManager (영속화)
                                     ↓
                          ExitRules (15min monitor)
```

### 단일 SignalGenerator (live-backtest parity)

`v3/strategy/signal.py:56` (`SignalGenerator`)는 backtest engine과
live pipeline이 **동일 인스턴스**를 호출한다.

```python
# v3/backtest/engine.py:109
self.signal_gen = SignalGenerator(opportunity_scorer=scorer, ...)

# v3/pipeline/live_pipeline.py: 동일 인스턴스 구성
self.signal_gen = SignalGenerator(opportunity_scorer=scorer, ...)
```

차이는 오직 **데이터 공급 방식**:
- backtest: `df_up_to_t` slice + 사전 계산된 `vol_predictions`
- live: 매일 OHLCV collect + VolInference 호출

**금지**: top_k, max_position, threshold 등 어떤 파라미터도 분기에 따라
달라지면 안 됨. 위반은 V2의 가장 큰 실패 원인.

### Immutable data flow

핵심 도메인 객체는 모두 `@dataclass(frozen=True)`:

- `Regime` (regime_v2.py:71)
- `OpportunityRow`, `OpportunityReport` (opportunity.py)
- `EntryCandidate`, `OperationalState`, `EntryDecision` (entry.py)
- `TradeSignal` (signal.py:40)
- `ExitDecision` (exit.py)

뮤테이션 없음. 모든 변환은 새 인스턴스 생성.

### Pure functions

```
compute_directional(ohlcv, sources) → DataFrame  [no state]
compute_conviction(ohlcv, vol_scores, sources) → DataFrame
OpportunityScorer.score(...) → OpportunityReport
EntryFilter.check(candidate, state) → EntryDecision
VolTargetSizer.size_portfolio(vols, confs, corr) → dict
```

테스트 가능성과 백테스트 재현성을 보장.

---

## 11. 백테스트 ↔ 라이브 정합성

### 절대 규칙

- 백테스트와 라이브 모두 **동일한 SignalGenerator** 호출
- 동일한 `OpportunityScorer`, `EntryFilter`, `RegimeDetectorV2`, `VolTargetSizer`
- 차이는 데이터 공급 방식(live=실시간, backtest=과거 replay)
- 매수+매도 양방향 슬리피지 적용

### 비용 파라미터

```yaml
NASDAQ:
  roundtrip:  0.001            # 0.1% (수수료 0.01% × 2 + 슬리피지 0.1% × 2)
  monthly:    5회 × 0.1% = 0.5%
  annual:     6%

KRX (제외):
  roundtrip:  0.010            # 1.0% — vol 전략에 부적합 확인
```

### Realistic slippage (variance)

```python
# 백테스트 엔진에서
slippage = base + ticker_vol × multiplier   # 시장·종목별
entry_price = open × (1 + slippage)
```

평균치만 빼는 게 아니라 vol 비례로 가산해 conservative 추정.

### Walk-forward 모드 (구현됨, 미실행)

`v3/backtest/walk_forward.py`: 252일 학습 → 63일 테스트 → 63일 step.
fold별 재학습 모듈 완성. test 외 fold에서 거래 0건이 발견된 후 fold별
재학습으로 수정 — 다만 cost 큰 작업이라 아직 실행하지 않음.

---

## 12. 자동화 — 월 재학습 + systemd

### 알파 가중치 재학습

- **주기**: 매월 1일 06:00 KST
- **Lookback**: 3년 rolling
- **자동화**: `alpha-retrain.timer` (Phase 25.2, 2026-05-07 적용)
- **다음 실행**: 2026-06-01 06:00 KST
- **로그**: `/var/log/alpha-retrain.log`
- **Artifact**: `v3/config/alpha_weights_history/alpha_weights_YYYY-MM.json`

### 라이브 데몬

- `quant-trading-v3.service` (systemd)
- KR 세션: 06:00 collect → 06:15 inference → 09:30 execute → 15:20 close
- US 세션: 22:00 inference → 23:40 execute → 04:30 close
- Monitor: 15분 간격 (TP/MAE/contraction 체크)

### 3-step bootstrap (alpha_weight_trainer)

```
A. Vanilla IC — 각 directional alpha의 전체 기간 IC
   · MIN_VANILLA_IC = 0.02
   · pass=false라도 저장 (정직)

B. Regime 분류 — 8개 macro feature stand-alone IC로 composite weights
   · feature_signs (+1 risk-on, -1 risk-off)
   · 5단계 quantile thresholds

C. Conditional IC — 각 regime 내 alpha × return IC
   · max(IC - 0.02, 0) shrinkage
   · 음수/소수 → uniform fallback
```

---

## 13. 성과

### 백테스트 (NASDAQ-100, 189거래일)

| 지표 | V2 최고 (KRX) | **V3.1 (NASDAQ)** |
|------|---------------|-------------------|
| Return | -53.2% | **+38.3%** |
| Sharpe | -9.86 | **1.65** |
| MDD | 53.6% | **4.0%** |
| Win Rate | 26.6% | **64.4%** |
| Profit Factor | - | **3.93** |
| 거래 수 | 200+ | **45** (월 6회) |
| 비용 합계 | ~50% | **0.5%** |

### Paper Trading 실측 (4/11 ~ 5/3, NASDAQ sandbox)

- 거래: BUY 7회, SELL 7회
- 실현 PnL: +1.39M KRW (+1.39% / 1억 base)
- 승률: 71.4% (5승 2패)
- 손익비: ~12:1 (큰 winner FANG/AMZN/ADI 영향)
- Deployed capital: 평균 32% (cash 68%) — **사이즈 부족이 누적 수익 병목**

### 페르소나 점수 (2026-04-21 기준)

| 원칙 | 점수 | 상태 |
|------|------|------|
| 1. 확신 있을 때만 | 5/10 | 관찰 중 |
| 2. 크게 | 3/10 | **가장 심각** → Phase 25.2에서 처방 |
| 3. 빠르게 | 8/10 | 현재 정책 적절 |

원칙 2 처방: sizer floor 0.05 → 0.15 (2026-05-07). 검증 기간 5/8~5/14.

---

## 14. 알려진 한계 + 보류 항목

### Hard limits (구조적 제약)

1. **소형 자본 제약** — 1억 → 종목당 30%가 의미 있는 단위. 10억+ 되면
   집중도 낮춰야 함 (1종목 10%로 분산).
2. **NASDAQ 의존** — KRX 비용 1%로는 vol 팽창 edge가 잠식됨. 한국 종목
   직접 거래는 현재 전략으로 불가.
3. **VolTransformer 학습 비용** — 5년 데이터 × 5 layers × 2.26M params.
   GPU 없이는 재학습 어려움. 현재 RTX 4060 Ti 8GB.
4. **paper sandbox 제약** — KIS sandbox는 NASDAQ 미지원. yfinance 실시간
   가격 + 시뮬 매칭 사용. 실제 호가 충격은 라이브에서만 검증 가능.

### Soft limits (개선 여지, 보류 중)

1. **Conditional Veto가 작동하지 않음** (FOLLOW_UPS 1순위)
   - 8h staleness threshold < KR↔US 14h 간격
   - 4/11~5/3 paper 5회 TP 청산 모두 "fire unconditionally"
   - 4/21 ADI +9.02%, AMZN +3.30% 같은 winner도 무차별 청산
   - 후보: threshold 16h, 세션 시작 즉시 generate, 정책 폐기

2. **opp gate가 "확신 != 비용 회수"**
   - `cost × 1.75 = 0.00175`은 "비용의 1.75배 회수" — 진짜 conviction 측정과
     다름
   - 진입 gate와 유지 veto gate가 동일 수식인 점도 검토 필요
   - 후보: 유지는 `cost × 3.0`, regime multiplier (caution × 1.5)

3. **Weekly monitor 주말 stale warning** — cosmetic. weekday 가드 부재.

4. **변수명 정합성** — `monthly_trades`가 옵션 C 이후 의미적으로
   `monthly_unique_tickers`. 호출처 7곳 일괄 변경 필요 (저우선).

### 페르소나 정합성 자가진단

- **원칙 1 (확신)**: opp gate 통과 = "비용 회수"인데 conviction 정의로
  불충분. 진입과 유지의 conviction 요구 차이 미반영.
- **원칙 2 (크게)**: Phase 25.2 처방 후 측정 중. 5/14 결과 대기.
- **원칙 3 (빠르게)**: 리스크 기반 청산이 실질 구현 — 적절. 시간 기반은
  trigger로 격하한 게 맞다고 판단.

---

## 15. 토론 포인트 — Claude와 합의할 의제

이 문서를 들고 가서 묻고 싶은 질문 모음. (하나씩, 동시 다발 수정 금지
원칙 준수.)

### Q1. 알파 다양성 — trend/reversion 만으로 충분한가?

- 현재 vanilla IC: trend 0.014, reversion -0.010. **둘 다 게이트 미달**.
- conditional IC는 strong_bull/neutral에서만 유의 (regime split이 정보를
  보존).
- 새 알파 후보(flow, sentiment, event) 중 무엇부터 IC 측정?
- 또는 기존 두 알파를 "다중 horizon" / "다중 변형"으로 확장?

### Q2. Conditional Veto 수정 방향

- (a) staleness 8h → 16h: 가장 단순. KR↔US 14h + 여유.
- (b) 세션 시작 즉시 generate_signal: cache freshness 보장. 비용 큼.
- (c) veto 폐기, 무조건 청산: V2 회귀 위험. 이미 Phase 25.2로 진입 발생
  여건 마련했으니 데이터부터 모으는 게 맞나?

### Q3. opp gate 이원화

- 진입: `cost × 1.75`, 유지: `cost × 3.0`을 분리한 이중 gate가 Phase 2의
  "Single source of truth" 원칙과 충돌하는가, 아니면 정련(refinement)인가?
- regime multiplier(caution × 1.5)는 Phase 26에서 제거한 "regime이
  threshold 조작"의 부활 아닌가?

### Q4. 사이징 철학 — Half-Kelly + vol target은 진짜로 "크게"인가?

- 현재 deploy 32%, 1억 기준 종목당 5~21M.
- Phase 25.2 floor 0.15 인상으로 caution 1종목 5.25M 보장 — 충분한가?
- target_annual_vol 0.15 자체를 0.20~0.25로 올리는 건 어떤가? (vol budget
  확대) MDD 4%가 너무 보수적일 수 있다.

### Q5. Universe 확장

- NASDAQ-100 → NASDAQ-500 / Russell 2000? 학습셋 14817 row → 50000+
  panel 가능. conditional IC 통계 안정.
- 하지만 모델 재학습 비용 + 유동성 분포 변화 + sector concentration 의미
  변화. 비용/효과는?

### Q6. VolTransformer 후속 — 2.26M으로 천장인가?

- d_model 192 → 384, layers 5 → 8 확장의 OOS IC 증분?
- 또는 multi-task: vol expansion + regime classification 동시 학습?
- 현재 IC 0.70으로 이미 매우 높아 추가 증분의 한계효용은 작을 가능성.

### Q7. 거래 빈도 — "월 5회 이하"가 정말 최적인가?

- V2 매일 → V3 월 5회는 비용 차원의 결정.
- NASDAQ 0.1% 비용 환경에선 월 10회도 가능.
- 더 많은 진입 기회 vs. edge 보존의 균형점?
- Phase 25.1 옵션 C(unique-ticker 카운트)의 실제 효과 측정이 prerequisite.

### Q8. 백테스트 → 라이브 transfer gap

- 백테스트 Sharpe 1.65 vs paper Sharpe(짧아 의미 없지만 +1.4% / 23일 ≈
  연 14% 페이스).
- 어디서 새고 있나? (a) 사이즈 부족 — Phase 25.2 처방 중. (b) Conditional
  Veto bug — winner 자르기. (c) opp gate 정의 — 진입 자체가 적음.
- 어느 가설이 가장 설명력 있는가?

---

## 16. 참고

| 문서 | 내용 |
|------|------|
| `CLAUDE.md` | 운영 규칙 + 개발 워크플로우 (Generator/Evaluator) |
| `docs/CHANGELOG.md` | Phase 22~25.1 narrative |
| `docs/FOLLOW_UPS.md` | 보류 항목 + 페르소나 점수 + 관찰 포인트 |
| `memory/v3_architecture.md` | V3 초기 아키텍처 메모 |
| `memory/phase2_plan_detailed.md` | Phase 26 (V3.2) 재설계 상세 |
| `v3/config/v3_config.yaml` | 모든 운영 파라미터 |
| `v3/config/alpha_weights.json` | 학습된 regime별 가중치 |
| `v3/strategy/opportunity.py` | 단일 게이트 수식 구현 |
| `v3/strategy/signal.py` | 백테스트·라이브 공유 오케스트레이터 |
| `v3/tests/test_regression.py` | 회귀 invariant |
