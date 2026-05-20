# Quant Trading System — CHANGELOG

CLAUDE.md에서 분리된 Phase별 이력. 운영 지침 자체는 CLAUDE.md, 잔존 후속
과제는 docs/FOLLOW_UPS.md 참조.

목차

- [Phase 22~23 — V2 재구축 + 서버 배포](#phase-2223--v2-재구축--서버-배포-2026-04-01)
- [V2 실거래 성과 (2026-04-01 ~ 04-08)](#v2-실거래-성과-2026-04-01--04-08)
- [V2.2 — 개별 스탑 제거, 3일 보유 (2026-04-03)](#v22--개별-스탑-제거-3일-보유-2026-04-03)
- [V2.3 — 모델/비용/인프라 10가지 개선 (2026-04-09)](#v23--모델비용인프라-10가지-개선-2026-04-09)
- [Phase 24 — 알파 개선 실험 + 근본 원인 진단 (2026-04-11)](#phase-24--알파-개선-실험--근본-원인-진단-2026-04-11)
- [Phase 25 — V3 Vol Expansion Trader 도입 (2026-04-11)](#phase-25--v3-vol-expansion-trader-도입-2026-04-11)
- [Phase 26 (V3.2) — Regime/Alpha 재설계 (2026-04-18)](#phase-26-v32--regimealpha-재설계-2026-04-18)
- [V3.2.1 — PaperBroker + Conditional TP (2026-04-20)](#v321--paperbroker--conditional-tp-2026-04-20)
- [V3.2.1 핫픽스 — hold_days 달력 일수 (2026-04-21)](#v321-핫픽스--hold_days-달력-일수-2026-04-21)
- [Phase 25.1 — Monthly Cap 옵션 C (2026-05-03)](#phase-251--monthly-cap-옵션-c-2026-05-03)
- [Phase 25.2 — Sizer floor 0.05→0.15 (2026-05-07)](#phase-252--sizer-floor-005015-2026-05-07)
- [Phase 26 (V3.3) — Profitability Engine (2026-05-09)](#phase-26-v33--profitability-engine-2026-05-09)
- [V3.3 전체 활성화 (2026-05-10)](#v33-전체-활성화-2026-05-10)

---

## Phase 22~23 — V2 재구축 + 서버 배포 (2026-04-01)

V1 (VAE→Transformer→GAN→RL 4단계) 폐기. V2 단일 모델로 축소.

### 완료 항목

- Feature 정규화 (z-score per column, 학습셋 통계)
- AlphaTransformer 단일 모델 (d_model=192, 5 layers, mean pooling)
- Confidence head (방향 정확도 확률 출력)
- 백테스트/라이브 파라미터 완전 통일 (system_config.yaml 1개)
- Conviction threshold (0~3종목, 0~100% 현금)
- Profit taking (+2.5% 절반, +5% 전량) — V2.2에서 +5% 전량만으로 단순화
- Stop loss (-2% 전량) — V2.2에서 제거
- 시간 기반 청산 (2시간 내 +1% 미달) — V2.2에서 제거
- 거래 세션 간소화 (7세션 → 2세션)
- US 신호 22:00 재생성
- 데이터 수집 파이프라인 연결
- 서버 배포 + E2E 검증

### 모델 성능 이력

| 버전 | Dir Acc | Val IC | Test IC | 피처수 | Loss | 게이트 |
|------|---------|--------|---------|--------|------|--------|
| V2.2 | 52.51% | 0.1044 | 0.0529 | 71 | Pairwise+Huber(0.5) | Val PASS / Test FAIL (완화) |
| V2.3 | 53.07% | 0.0748 | 0.0383 | 35 | ListMLE+Huber(1.5) | 모두 FAIL (정직) |

---

## V2 실거래 성과 (2026-04-01 ~ 04-08)

### 포트폴리오 추이 (sandbox, 초기 1억)

| 날짜 | 포트폴리오 | 일간 | 누적 | 비고 |
|------|-----------|------|------|------|
| 4/1 | ~100M | ~0% | 0% | V2 배포, KR=CASH |
| 4/2 | 96.5M | -3.37% | -3.5% | V2.0: 043260 SL -3.46%, 001527 SL -3.08% |
| 4/3 | 95.5M | -1.02% | -4.5% | V2.1 배포. 043260 SL -3.32% |
| 4/6 | 93.5M | -2.08% | -6.5% | V2.2: 포트폴리오 리스크 -1.65% 발동 |
| 4/7 | 92.0M | -1.62% | -8.0% | 포트폴리오 리스크 -1.55% 발동 |
| 4/8 | 93.1M | +1.16% | **-6.91%** | 반등. MDD -9.97% |

### 핵심 패인 분석

- 043260.KQ: 7일간 4번 진입, 모두 손실 (-3.46%, -3.32%, -6.43%, -5.88%)
- 001527.KS: 고스트 포지션 (매도 40+회 실패, 5분마다 반복)
- 원인: 모델 Rank IC 0.053으로 edge 부족 + 동일 종목 반복 진입
- **교훈**: 게이트를 완화해서 배포하면 안 된다 (V2.2의 7일간 -6.91% 직접 원인)

### Exit Strategy 비교 백테스트 (120일, 2025-09~2026-03)

| 전략 | 수익률 | Sharpe | 승률 |
|------|--------|--------|------|
| **V2.2 (3일, 스탑없음)** | **-7.65%** | **-1.38** | 39.9% |
| 3일+SL2% | -18.43% | -6.52 | 25.0% |
| 당일청산 | -14.48% | -3.76 | 31.1% |
| 당일+SL2% (V2.0) | -17.26% | -7.43 | 25.3% |

**결론**: 스탑 제거+3일 보유가 상대적으로 최적이지만, 모든 전략이 마이너스.

---

## V2.2 — 개별 스탑 제거, 3일 보유 (2026-04-03)

실거래 2일(4/2~4/3) 분석 결과, 전략 구조 문제 확인:
- 기대값 -1.14%/거래 (승률 37.5%, 손익비 0.23:1, Kelly 음수)
- 근본 원인: prediction_horizon=3일 vs 실제 보유 2시간 (time_exit)
- 043260.KQ: 스탑 -3.32% → 종가 -0.11% (V자 반등에서 바닥 매도)

### 변경 사항

- 개별 스탑로스 제거 → 포트폴리오 daily_loss_limit -1.5% (미실현 포함)
- time_exit 제거 → 3일 보유 만료까지 유지
- 이익실현: +5% 전량만 (부분 청산 제거)
- 진입 09:10 → 09:30 (스프레드 안정화)
- session_close: 만료 포지션만 청산, 오버나이트 보유
- 서킷 브레이커: MDD 5/10/20/30% 단계별 스케일링
- 매도 실패 시 KIS 잔고 확인 + 포지션 동기화
- score_history JSON 영속화

---

## V2.3 — 모델/비용/인프라 10가지 개선 (2026-04-09)

V2.2 실거래 7일(4/1~4/8) 분석: -6.91%, MDD -9.97%.
세계적 퀀트 트레이더 관점에서 전면 재진단 → 10가지 개선.

### 모델/학습 변경

- PairwiseRankingLoss → ListMLE (listwise 순위 학습)
- Huber delta 0.5 → 1.5 (아웃라이어 로버스트)
- 피처 71 → 35개 (상관관계 기반 중복 제거, max_corr=0.85)
- Walk-forward 학습 파이프라인 구현 (4 fold)
- 리턴 클리핑 [-1, 10] → [-0.3, 0.3] (KRX 일일 제한가 현실화)

### 비용/게이트 변경

- 거래비용 0.6% → 1.0% (세금+슬리피지 현실화)
- 수수료 0.02% → 0.05% (한투 실제 비용)
- 배포 게이트: dir_acc 52% → 54.5%, rank_ic 0.05 → 0.10
- bear_threshold: -3% → -8% (과민 반응 방지)
- ranking_loss_weight: 0.5 → 0.6 (ranking이 핵심)

### 실행 인프라 변경

- 고스트 포지션 강제 제거 (매도 3회 실패 → 삭제)
- 포지션 디스크 영속화 (open_positions.json)
- 멀티데이 냉각기 (5일 내 2패 → 진입 차단)
- 연속 진입 제한 (동일 종목 3일 연속 → 차단)
- Kelly → 변동성 기반 + 균등 + confidence tilt 사이징

### 백테스트 정합성

- day_trade_default: True → False (3일 보유 반영)
- 매도 슬리피지 추가 (기존: 매수만 적용)
- 백테스트 exit strategy 비교 스크립트 (7가지 전략)

### V2.3 비용 파라미터 (현실화)

```yaml
transaction_cost_rate: 0.010   # 왕복 1.0% (V2.2: 0.6%)
commission_rate: 0.0005        # 편도 0.05% (V2.2: 0.02%)
slippage_by_market:
  KOSPI: 0.003                 # 매수+매도 각각 적용
  KOSDAQ: 0.005
  NASDAQ: 0.001
```

**결과**: 게이트 미통과 (Val IC 0.0748, Test IC 0.0383). 배포 불가.
**교훈**: OHLCV만으로는 Rank IC 0.10 달성 어려움. 피처 품질이 병목.

---

## Phase 24 — 알파 개선 실험 + 근본 원인 진단 (2026-04-11)

### Phase A: 피처 추가 실험

| 실험 | 추가 피처 | Test IC 변화 | 결과 |
|------|----------|-------------|------|
| V2.3 (기준) | OHLCV 35개 | 0.038 | FAIL |
| V2.4 (+수급) | +flow 1개 | 0.057 (+0.019) | FAIL |
| V2.5 (+섹터+베타) | +sector 3, beta 3 | 0.041 (-0.016) | FAIL |

- 수급: 네이버 금융 크롤링 (`data/flow_data.py`), 175종목 2년치 수집
- 외국인 순매수 IC=+0.025 (175종목), 대형주만(61종목)에서는 IC≈0
- 섹터/베타 피처 추가 시 오히려 IC 하락 (과적합 또는 노이즈)

### Phase B: 예측 대상 실험

| 타겟 | 최고 피처 IC | 비고 |
|------|------------|------|
| 1일 상대수익률 | -0.040 (return_1d) | 3일보다 약간 강함 |
| 3일 상대수익률 (현행) | -0.036 (volatility_20d) | 현행 |
| **변동성 예측** | **+0.487 (volatility_20d)** | **13배 강한 신호** |

### 백테스트 최종 검증 (V2.5, 120일)

| 전략 | 수익률 | Sharpe | 승률 | Cash일 |
|------|--------|--------|------|--------|
| 3일 보유 | **-53.2%** | -9.86 | 26.6% | **0일** |
| 3일+SL2% | -59.4% | -14.86 | 24.3% | 0일 |
| 당일청산 | -54.6% | -16.49 | 16.1% | 0일 |
| 5일 보유 | -56.4% | -7.39 | 28.1% | 0일 |

**결론**: 모든 전략 -50% 이상 손실. 피처 추가로 해결 불가.

### 근본 원인 진단

V1→V2→V2.5 전체를 관통하는 **불변 전제 4가지**가 문제:

```
1. 입력: 공개 OHLCV 가격 데이터 (모두가 접근 가능 → edge 없음)
2. 목표: 3일 수익률 순위 예측 (가장 어려운 예측 문제 중 하나)
3. 실행: 매일 거래 (비용 240%/년 → IC 0.30+ 필요, 현재 0.04)
4. 시장: KRX 왕복 1% (US 0.1%의 10배 비용)
```

V1→V2 "재설계"는 모델 간소화였을 뿐, 위 전제는 동일했다.
**모델 아키텍처가 아니라 전제 자체를 바꿔야 한다.**

---

## Phase 25 — V3 Vol Expansion Trader 도입 (2026-04-11)

> **"변동성 팽창 예측 → 방향은 규칙 → 조건부 진입"**

V2의 4가지 불변 전제를 모두 변경한 전면 재설계.

### 전제 변경 내역

| 전제 | V2 | V3 | 결과 |
|------|-----|-----|------|
| 무엇을 예측 | 수익률 순위 (IC=0.04) | **변동성 팽창** (IC=0.70) | **17.5× 강한 신호** |
| 언제 거래 | 매일 (연 240% 비용) | **월 5회 이하** (연 6% 비용) | **비용 40× 감소** |
| 어디서 거래 | KRX (왕복 1%) | **NASDAQ** (왕복 0.1%) | **비용 10× 저렴** |
| 무엇을 보고 | OHLCV 가격만 | OHLCV + vol 구조 + 수급 + 이벤트 | 다차원 신호 |

### V3.1 백테스트 결과

**모델 성능 (NASDAQ-100, 5년 학습):**

| 지표 | Val (Best E19) | Test (OOS) | 게이트 |
|------|----------------|------------|--------|
| Vol IC | 0.7185 | 0.6998 | >0.30 PASS |
| Vol Rank IC | 0.7896 | 0.7486 | >0.20 PASS |
| Dir Accuracy | 80.3% | 78.2% | >55% PASS |
| High Conf Acc | - | 85.4% | 확신 시 극대화 |

**백테스트 (189일, NASDAQ-100, 최적 파라미터):**

| 지표 | V2 최고 | V3.1 |
|------|---------|------|
| Return | -53.2% | **+38.3%** |
| Sharpe | -9.86 | **1.65** |
| MDD | 53.6% | **4.0%** |
| Win Rate | 26.6% | **64.4%** |
| Profit Factor | - | **3.93** |
| 거래 수 | 200+ | **45** (월 6회) |
| 비용 합계 | ~50% | **0.5%** |

**파라미터 스윕 결과 (9개 조합):**

| 조합 | Return | Sharpe | MDD | Win | PF |
|------|--------|--------|-----|-----|-----|
| Baseline (MAE -3%) | +5.2% | 0.65 | 4.4% | 61% | 1.61 |
| MAE -4% | +10.7% | 1.06 | 4.4% | 68% | 2.13 |
| No MAE + 진입완화 | **+38.3%** | **1.65** | **4.0%** | **64%** | **3.93** |

### V3 핵심 교훈

```
1. 예측 대상이 전부다
   - 수익률 예측 IC=0.04 → 변동성 예측 IC=0.70
   - "무엇을 예측할 것인가"가 모델 아키텍처보다 100배 중요

2. 비용이 전략을 결정한다
   - KRX 1% 비용 → vol 팽창 edge(3~5%)를 잠식 → FAIL
   - NASDAQ 0.1% 비용 → 동일 edge로 Sharpe 1.65 → PASS
   - 동일 모델, 동일 전략, 시장만 다르면 결과가 정반대

3. 개별 스탑은 해롭다 (V2.2~V3 일관된 결론)
   - MAE -3% 스탑: Sharpe 0.65
   - MAE 제거: Sharpe 1.65 (2.5× 개선)
   - vol 팽창 환경에서 tight stop = 노이즈 매도

4. 진입 빈도를 줄이면 수익이 늘어난다
   - V2 매일 거래: 연 비용 240%, 모든 전략 마이너스
   - V3 월 5회: 연 비용 6%, edge 보존

5. 확신 없으면 현금이 최고의 포지션
   - 99종목 중 2~3종목만 진입, 나머지 87% 현금
   - 이것이 MDD 4%의 비결
```

---

## Phase 26 (V3.2) — Regime/Alpha 재설계 (2026-04-18)

> **"Regime은 임계값 조작자가 아니라 알파 가중치 선택자"**

Phase 1의 `threshold_multiplier`, `engine toggle`, 상속 변이 등 원칙
위배 요소를 전면 제거하고, Two Sigma/AQR 컨벤션에 맞춰 재설계.

### 재설계 동기

V3.1 운영 7일(4/11~4/18) 관찰 결과, 연속 `volatile` 레짐 + CASH 지속으로
QQQ +6% 급등장을 완전히 놓침. 원인 진단에서 발견된 **7가지 원칙 위배**:

| # | 위치 | 문제 |
|---|------|------|
| P1 | `signal.py` `_apply_threshold_multiplier` | entry filter 필드 런타임 뮤테이션 |
| P2 | `CrossAssetRegimeState(RegimeState)` | Liskov 위반, 필드 추가 상속 |
| P3 | `RegimeConfig.engine` 토글 | 한 시스템 두 철학 공존 |
| P4 | `live_pipeline._detect_regime` if/else | engine별 분기 |
| P5 | `backtest/engine.py` 단독 regime | live-backtest 코드 불일치 |
| P6 | vol 단일 알파 | "vol 팽창 없는 bull 랠리" 원천 차단 |
| P7 | Phase 1 가중치 임의 지정 (0.20, 0.15) | 데이터 증거 없음 |

### Phase 2 설계 7대 원칙

1. **Single source of truth** — 진입은 `opportunity > cost × k` 단일 수식
2. **직교·가산적 알파** — 여러 독립 알파 소스, 회귀에서 가중합만 다름
3. **Regime = 알파 가중치 선택자** — threshold 조작 금지, 가중치 분포만 변경
4. **Immutable data flow** — frozen dataclass, pure functions, 뮤테이션 제거
5. **단순한 Regime 출력** — name, score, alpha_weights, position_scale만
6. **Legacy 제거** — engine 토글 없음, 단일 경로
7. **백테스트가 설계를 증명** — 모든 가중치/threshold는 과거 IC 학습 결과

### 매수 정책 변경

#### 이전 (V3.1, Phase 1) — 10개 조건 게이트

```
C1: vol_score ≥ min_vol_expansion (0.05)
C2: confidence ≥ min_confidence (0.30)
C3: direction_clarity ≥ min_direction_clarity (0.20)
C4: direction == "long"
C5: monthly_trades < dynamic_max
C6: current_positions < max_positions
C7: circuit_breaker OFF
C8: expected_move > cost × 1.75 (vol-adjusted)
C9: ticker_volume ≥ min_volume
C10: sector_concentration ≤ max_sector_conc
```

#### 이후 (V3.2, Phase 2) — 단일 알파 게이트 + 운영 제약

```
[알파 게이트] — OpportunityScorer (단일)
  opportunity(ticker) > cost × 1.75

[운영 제약] — EntryFilter (C5~C10을 재분류)
  ✓ 포지션 한도 (max_positions)
  ✓ 월 거래 한도 (dynamic, win_rate/mdd 반영)
  ✓ Circuit breaker halt
  ✓ 유동성 (daily_volume ≥ 5억)
  ✓ 섹터 집중 (≤ 2 per sector)
```

**변경 본질**: 알파 관련 게이트(C1/C2/C3/C4/C8)를 **opportunity 단일 수식**에
통합. 임계값 여러 개에 흩뿌려진 로직을 `edge > cost × k` 하나로.

### Regime별 진입 방식 차이

**이전**: regime이 `threshold_multiplier`로 entry filter의 `min_vol_expansion`
등을 동적 수정 (뮤테이션) → 원칙 위배.

**이후**: regime은 **알파 가중치**만 결정. 게이트 자체는 고정:
- strong_bull: alpha_weights 학습 결과 (예: reversion 1.0) → reversion 알파가
  주도하는 opportunity → 그 기준 진입
- bull/neutral/caution/bear: 각 regime이 자기 가중치로 opportunity 계산
- bear: `position_scale=0 → CASH` 단락

### Position scale 연속화

Phase 1: discrete scale (bull 1.2, neutral 1.0, volatile 0.6, bear 0.0)
Phase 2: **연속 score → smooth scale** (Bridgewater discrete + Medallion continuous 하이브리드)

```
POSITION_SCALE_CURVE (piecewise linear):
  score 0.00 → 0.00  (bear = CASH)
  score 0.25 → 0.30
  score 0.40 → 0.60
  score 0.55 → 0.90
  score 0.75 → 1.10
  score 1.00 → 1.20
```

**효과**: regime 전환 경계(0.54 ↔ 0.56)에서 포지션 쇼크 없음. 매끄러운 조절.

### Alpha weights (regime별)

`v3/config/alpha_weights.json`에 저장 (S2 학습 결과). 최신 (2026-04-18, 3년 학습):

```
strong_bull: {trend: 0.0, reversion: 1.0}   # reversion IC=0.113 유일 양수
bull:        {trend: 0.5, reversion: 0.5}   # IC 모두 게이트 미달 → uniform
neutral:     {trend: 1.0, reversion: 0.0}   # trend IC=0.028 유일 유의미
caution:     {trend: 0.5, reversion: 0.5}   # uniform
bear:        {trend: 0.5, reversion: 0.5}   # uniform (CASH로 우회)
```

### 알파 가중치 학습 정책

#### 월 재학습 + 3년 rolling

- **주기**: 매월 1일 재학습 (분기는 regime shift 반영 느림)
- **Lookback**: 3년 (더 길면 구체제 편향, 짧으면 noise)
- **파일**: `v3/config/alpha_weights.json` (latest) +
  `v3/config/alpha_weights_history/alpha_weights_YYYY-MM.json` (versioned)
- **수동 실행**: `python v3/backtest/alpha_weight_trainer.py --lookback-years 3`

#### 3-step bootstrap (원칙 7 구현)

```
A. Vanilla IC — 각 directional alpha의 전체 기간 IC 측정 (baseline edge)
   · MIN_VANILLA_IC = 0.02  (노이즈 vs 신호 구분)
   · vanilla_ic_pass = false 여도 저장 (정직 기록)

B. Regime 분류 — 7개 macro feature의 stand-alone IC로 composite weights 산출
   · feature vs 시장 forward return Spearman 측정
   · max(|IC|, 0) / total 정규화 → feature_weights
   · sign(IC) 저장 → feature_signs

C. Conditional IC — 각 regime 내에서 alpha × return IC 재측정
   · 최소 40 샘플 미달 regime → uniform fallback
   · weight = max(IC - 0.02, 0) / total  (shrinkage로 노이즈 차단)
   · 전부 게이트 미달 → uniform weights
```

#### Conviction accuracy (별도 측정, 가중치 학습 아님)

```
expansion_ic = Spearman(vol_conviction, realized_5d_vol / past_20d_vol - 1)
             = 0.1899  (VolTransformer 훈련 타겟과 일치 측정)

level_ic     = Spearman(vol_conviction, realized_5d_vol)
             = 0.0358  (참고값, 사용 안 함)
```

### 학습된 feature_signs (3년 NASDAQ 2023~2026, contrarian 패턴)

```
+1 (percentile 그대로 기여):    vix_ratio, hy_level, gold_spy_mom_60d
-1 (1 - percentile 반전):       yc_slope, hy_change_60d, dxy_mom_60d,
                                hyg_tlt_mom_60d, breadth
```

**해석**: 최근 3년은 NASDAQ 상승장 → "vix 높고 HY 넓으면 반등"(contrarian),
"breadth 높으면 과열"(mean-revert) 패턴이 학습됨. 데이터 기반 결과이므로 존중
하되, 시장 국면 변화 시 월 재학습에서 자동 수정됨.

### 백테스트-라이브 정합성 강화

**이전 (V3.1) — 다른 코드 경로**
- `live_pipeline` → `RegimeDetector` + `DirectionEngine` + `EntryFilter`
- `backtest/engine.py` → 동일하지만 재구현 (불일치 리스크)

**이후 (V3.2) — 단일 코드 경로**
- 양쪽 모두 동일한 `SignalGenerator`(v3/strategy/signal.py) 호출
- 동일한 `OpportunityScorer`, 동일한 `EntryFilter`, 동일한 `RegimeDetectorV2`
- 차이점은 오직 "데이터 공급 방식"(live=실시간, backtest=과거 replay)

### 게이트 수정 사항

| 게이트 | 이전 (V3.1) | 이후 (V3.2) |
|--------|------------|-------------|
| 알파 진입 임계값 | `min_vol_expansion=0.05, min_confidence=0.30, min_direction_clarity=0.20` | **제거**. `opportunity > cost × 1.75` 하나 |
| Vanilla IC | 없음 | `MIN_VANILLA_IC = 0.02` (게이트 통과 실패도 기록) |
| Conditional IC shrinkage | 없음 | `max(IC - 0.02, 0)` (노이즈 레벨 기각) |
| Regime 샘플 최소 | 없음 | 40개 미달 regime → uniform fallback |
| Feature signs | 하드코딩 (Phase 1 patch) | **학습 기반** (artifact에 저장) |

### 코드 구조 변경 일람

#### 신규 파일 (4개)

```
v3/strategy/alpha_sources.py       — AlphaSource, ConvictionSource ABC + 구현
v3/strategy/opportunity.py         — OpportunityScorer (stateless pure function)
v3/strategy/regime_v2.py           — @dataclass(frozen=True) Regime + detector
v3/backtest/alpha_weight_trainer.py — 월 재학습 트레이너
```

#### 신규 설정/데이터 (2개)

```
v3/config/alpha_weights.json                                (latest)
v3/config/alpha_weights_history/alpha_weights_YYYY-MM.json  (monthly version)
```

#### 재작성 파일 (4개)

```
v3/strategy/signal.py        — SignalGenerator 재작성, _apply_threshold_multiplier 제거
v3/rules/entry.py            — EntryFilter 재작성 (운영 제약 5개만)
v3/pipeline/live_pipeline.py — engine toggle 제거, 단일 경로
v3/backtest/engine.py        — SignalGenerator 재사용 (live parity)
```

#### 아카이브 (_legacy/)

```
v3/strategy/_legacy/regime_single_asset.py   (이전: regime.py)
v3/strategy/_legacy/regime_cross_asset.py    (Phase 1 patch)
v3/pipeline/_legacy/inference_pipeline.py    (미사용, 정리)
v3/backtest/_legacy/engine_phase1.py         (이전 engine.py)
```

#### Config 정리

```
v3/config/schema.py     — RegimeConfig.engine 토글 필드 제거
                          legacy single_asset 필드 (bull_threshold 등) 제거
                          hysteresis_days + macro_history_years만 유지
v3/config/v3_config.yaml — 위와 동일, yaml에서도 정리
```

### 검증 결과 (2026-04-18)

#### S2 학습 결과
- Panel: 14,817 rows (150 dates × 99 tickers, weekly × 3y)
- Regime counts: neutral 7411, caution 4736, bull 2175, bear 297, strong_bull 198
- Vanilla IC: trend 0.014, reversion -0.010 (둘 다 게이트 미달, 정직 기록)
- Conditional IC: strong_bull reversion **+0.113** (유일 유의미), neutral trend +0.028
- Conviction accuracy: vol expansion_ic **0.19**

#### 실데이터 검증 (로컬·서버 동일)
- Regime: `caution`, score=0.33, scale=0.47, weights uniform
- 99 tickers → opportunity gate **10개 통과** (gate=0.00175)
- FANG 선정: direction=0.035, conviction=0.657, **opportunity=0.023** (gate의 13배)

### 제거된 요소 (Phase 1)

```
[제거] signal.py::_apply_threshold_multiplier (뮤테이션)
[제거] signal.py::_base_thresholds 캐시 (몽키패치)
[제거] CrossAssetRegimeState(RegimeState) 상속 (Liskov 위반)
[제거] RegimeConfig.engine 토글 필드 (두 철학 공존)
[제거] live_pipeline._detect_regime() if/else 분기 (단일 경로로)
[제거] schema.py::apply_threshold_multiplier 플래그
[제거] v3_config.yaml legacy regime 필드 (bull/bear_threshold 등)
[아카이브] v3/strategy/regime.py → _legacy/
[아카이브] v3/strategy/regime_cross_asset.py → _legacy/
[아카이브] v3/pipeline/inference_pipeline.py → _legacy/
[아카이브] v3/backtest/engine.py (phase1) → _legacy/
```

### Phase 2 커밋 체인

```
9ebff71  feat: S1 — alpha_sources.py (3 independent alphas, Phase 2)
d4f5276  feat: S1+S2 revised — alpha/conviction separation, regime-conditional IC
5ddbd75  feat: S3 — regime_v2.py frozen Regime detector + artifact export
843cbc4  feat: S4 — opportunity scorer + signal/entry rewrite + live_pipeline
bfea53e  feat: S5 — backtest/engine.py Phase 2 rewrite + legacy archive
```

### V3.2 추가 교훈

```
6. 알파와 리스크를 혼동하지 말라
   - VolTransformer는 vol 예측 (risk model), 수익률 예측 아님
   - Alpha model ≠ Risk model (Two Sigma 컨벤션)
   - conviction으로 modulate, 직접 가중합 금지

7. 임계값은 수학적 안전계수만, 정책 레버 아님
   - k=1.75 (cost multiplier)는 고정 수학값 (보수적 기대수익 요구)
   - Regime은 threshold 조작 금지, 가중치만 조절
   - "매개변수 튜닝"으로 기회를 만들지 말 것

8. Shrinkage는 과적합 방어의 핵심
   - Conditional IC 0.03 → 100% weight (과적합)
   - max(IC - 0.02, 0) 적용 → uniform fallback (보수적)
   - 노이즈 레벨 edge로 포지션 잡지 말기

9. 백테스트와 라이브는 동일 함수를 호출해야 한다
   - 두 세계 다른 코드 = 결과 불일치 = 라이브 손실
   - `SignalGenerator` 하나를 양쪽에서 재사용

10. 설계 원칙 위배는 나중에 비싸게 돌려받는다
    - Phase 1 threshold_multiplier 패치 → 1주간 CASH, 급등장 놓침
    - 패치 누적 금지, 원칙 훼손 발견 즉시 재설계
```

---

## V3.2.1 — PaperBroker + Conditional TP (2026-04-20)

Phase 2 재설계 후 첫 TRADE 신호 발생(2026-04-20 09:30 KST, regime=caution,
FANG opp=0.023) 과정에서 **두 가지 실행 레벨 결함**을 발견·수정.

### 개선 1: PaperBroker 와이어링 (sandbox NASDAQ 체결 복구)

**배경**
- V3 universe는 NASDAQ-only. KIS sandbox는 **해외주식 API 미지원**
  (`openapivts...:9443/.../overseas-stock/.../price` → 404).
- `v3/config/v3_config.yaml`에 `broker.paper_trading: true`가 설정되어 있었으나
  `TradingExecutor`가 이를 **읽지 않고** 항상 `KISApi`로 라우팅.
- 결과: 09:30 세션에서 FANG 신호 발생했으나 `Buy order failed FANG: 404`로
  **Entries=0**. Paper 검증 루프 자체가 성립하지 않음.

**수정 내용** (`v3/execution/executor.py`)
```python
# __init__
self.paper: PaperBroker | None = (
    PaperBroker(initial_capital=cfg.backtest.initial_capital)
    if cfg.broker.paper_trading else None
)

# 해외 티커 라우팅 헬퍼
def _use_paper(self, ticker: str) -> bool:
    return self.paper is not None and not is_domestic(ticker)
```

- `_place_buy_order`, `_place_sell_order`, `_get_price`,
  `_get_portfolio_value`, `_get_holding_qty` 5개 메서드에 paper 분기 추가
- 해외 티커(non-domestic) → PaperBroker(yfinance 실시간/전일종가)
- 국내 티커 → 기존 KIS 경로 유지 (NASDAQ-only 환경이라 실제 해당 없음)
- 포지션 dict에 `qty` 저장 (매도 시 필요 — 기존 누락 버그 동시 수정)

**검증 결과** (22:16 KST 수동 세션)
```
PAPER BUY FANG: 82주 @ $180.45 = 20,715,691 KRW (수수료 2,072)
Entries: 1
```

### 개선 2: Conditional TP (opportunity 재평가 후 유지/청산)

**배경**
- V3 exit 규칙은 시간감쇠 TP 도달 시 **무조건 청산**
  (Day1 +5%, Day2 +4%, …, Day5 +1.5%).
- 사용자 정책: "+5% 됐다고 무조건 청산하는 게 아니라 그 때의 시그널을 보고
  또 판단하자"
- 승자를 조기에 자르는 문제. 모델이 여전히 "진입할 만하다"(opportunity > gate)
  판단하는 종목을 TP 도달만으로 매도하면 edge 손실.

**정책** (2026-04-21 확장: max_hold 포함)
- **시간 기반 청산**(`profit_take`, `max_hold`) 트리거 시:
  - `opportunity(ticker) > cost × 1.75` 성립 → **보유 유지** (veto + 로그)
  - 그 외 → 청산
- **리스크 기반 청산**(`vol_contraction`, `dynamic_stop_mae`, `portfolio_stop`)
  은 **veto 금지** (무조건 체결). 리스크 신호는 시간 경과와 무관.
- Phase 2의 진입 수식(`opportunity > cost × k`)을 **유지 판단에도 재사용**
  → 단일 진입/유지 기준 (설계 통일)
- 원칙: "시간이 다 됐다"는 청산 사유가 아니라 **재평가 트리거**일 뿐.
  모델이 여전히 진입할 만하다면 보유 유지하는 것이 edge 보존.

**구현** (3파일)

```python
# v3/strategy/signal.py — TradeSignal 필드 추가
opportunity_map: dict[str, float]  # 전 티커 opportunity
opportunity_gate: float             # cost × gate_multiplier

# v3/pipeline/live_pipeline.py — generate_signal 직후 캐시
self._opportunity_map = dict(signal.opportunity_map)
self._opportunity_gate = signal.opportunity_gate

# executor에 전달
self.executor.monitor_positions(
    today,
    opportunity_map=self._opportunity_map,
    opportunity_gate=self._opportunity_gate,
)

# v3/execution/executor.py — monitor_positions veto 로직
if exit_decision.should_exit and exit_decision.reason == "profit_take":
    opp = opportunity_map.get(ticker)
    if opp is not None and opp > opportunity_gate > 0:
        logger.info(f"TP HOLD {ticker}: ret={ret:+.2%} target hit, "
                    f"but opportunity={opp:.5f} > gate={opportunity_gate:.5f} — hold")
        continue
```

### 매도 정책 변경

```
기존 (v3.2):
  1. 시간감쇠 이익실현: Day1 +5% ~ Day5 +1.5% → 무조건 청산
  2. Vol 수축 (진입 vol의 70% 이하 3일 지속) → 청산
  3. 보유 만기 (5일) → 청산
  4. 포트폴리오 일간 -1.0~2.0% → 전량 청산

개정 (v3.2.1):
  1. 시간감쇠 이익실현: Day1 +5% ~ Day5 +1.5%
     → opportunity(ticker) > cost × 1.75 이면 **유지** (veto)
     → 그 외 청산
  2~4. 변경 없음 (무조건 청산)
```

### 검증 (2026-04-20 22:16~22:22 KST)

```
opportunity_map cached: 99 tickers, gate=0.00175
  FANG: opp=0.02303  passes=True     (gate의 13배)
  BKR:  opp=0.01691  passes=True
  EXC:  opp=0.01144  passes=True
  TMUS: opp=0.00730  passes=True
  NFLX: opp=0.00547  passes=True

PAPER BUY FANG: 82주 @ $180.45
Entries: 1, Paper total: 100,679,152 KRW
```

### V3.2.1 추가 교훈

```
11. 설정 필드는 반드시 코드가 소비해야 한다
    - broker.paper_trading=true가 설정만 되고 코드 미소비 → 1주간 모든
      NASDAQ 신호가 KIS 404로 체결 실패. "dead config"는 즉각 수정 대상.

12. TP는 신호의 일부지, 신호 위에 올라앉은 규칙이 아니다
    - +5% 도달을 "확정 청산"이 아닌 "재평가 트리거"로 재정의
    - 진입과 유지를 동일 수식(opportunity > cost × k)으로 통일
    - 승자 자르기 방지 = 손절만큼 중요한 edge 보존 메커니즘
```

---

## V3.2.1 핫픽스 — hold_days 달력 일수 (2026-04-21)

**증상**: FANG 22:16 진입 → 23:08 매도(+1.65%) → 23:40 재진입. 1시간 만에 churn.

**원인**: `monitor_positions`에서 `hold_days = pos["hold_days"] + 1` — monitor
호출마다 +1 증가. 15분 간격 monitor 4회면 hold_days=4 → `TIME_DECAY_TARGETS[4]
= 0.015` (Day 5 TP 1.5%) → +1.65% ≥ 1.5% → profit_take 트리거.

**수정** (`v3/execution/executor.py`)

```python
from datetime import date
entry_date_str = str(pos.get("entry_date", current_date))[:10]
entry_d = date.fromisoformat(entry_date_str)
today_d = date.fromisoformat(current_date)
hold_days = max(0, (today_d - entry_d).days)   # 달력 일수
```

**교훈 13**: 시간 관련 카운터는 **호출 빈도가 아니라 도메인 시간축**을 기준으로
측정해야 한다. 15분 tick이 "1일"이 되면 Day5 TP가 1시간 만에 도달.

---

## Phase 25.1 — Monthly Cap 옵션 C (2026-05-03)

원칙 1 ("확신 있을 때만") implementation 결함 수정. 4/11~5/3 paper 데이터
관찰 결과 monthly cap이 churn에 의해 잠식되어 진짜 결정자가 conviction이
아니라 임의 cap이 된 상태. 단기 응급조치로 옵션 C 채택.

### 관찰 사실 (4/11~5/3 데이터)

- 4/24 23:40 세션 31종목 opp gate 통과 → `rejections={'monthly_trades': 31}` 전부 차단
- 4월 BUY 7회 중 **5회 FANG churn** (22:00 진입 → 09:30 청산 반복)
- 4/27~4/30 8세션 100% CASH (opportunity=31 일관) — QQQ +1.55% 놓침

### 진단 — 페르소나 misalignment

- 페르소나 본질은 "Conviction-or-Cash" (원칙 1)이지 "월 N회 이하"가 아님
- monthly cap이 의도치 않게 conviction filter 역할 (opp gate 0.00175 너무 낮음)
- 결과: 진짜 결정자가 conviction이 아니라 **임의 cap**이 됨

### 구조적 결함 3가지 (기록 보존)

1. monthly cap이 "건수" 카운트 → FANG 5회 = 신규 5회와 동등 취급
2. 동일 종목 재진입과 신규 진입을 구분 못함
3. 8h staleness가 conditional veto 무효화 → churn 누적 → 월 budget 조기 소모

### 대안 비교 (당시 평가)

| 옵션 | 본질 | 평가 |
|------|------|------|
| A. 현행 유지 | 건수 cap | 페르소나 misaligned |
| B. cap 폐지 + opp gate 상향 (cost×3~5) | conviction | 중기 보류 (gate 튜닝 리스크) |
| **C. cap 유지 + unique-ticker 카운트** | **종목 다양성** | **✅ 단기 채택** |
| D. portfolio turnover 카운트 | 회전율 | 장기 보류 (구현 복잡) |

### 적용 내역 (2026-05-03)

- `position_manager.py::monthly_trade_count` — set 기반 unique 카운트
- `backtest/engine.py` — `monthly_unique_tickers: set[str]`로 parity 유지
- `executor.py::execute_entry` — skip 사유 진단 로깅 추가 (정책 무변경)
- `tests/test_regression.py` — `TestMonthlyUniqueTickerCount` 4개 테스트 추가
- 변수명 `state.monthly_trades` 유지 (의미만 변경; 향후 정리는 후속 과제)
- Hook 자동 회귀: 28/28 PASS

### 적용 후 검증 포인트 (5/4 영업일부터 1~2주 관찰)

- monthly cap 도달까지 unique ticker 수
- FANG 같은 churn 패턴이 다시 나오는지 (그때는 카운트되지 않음)
- 5/1 entries=0 케이스 재현 시 executor 진단 로그로 사유 확정
  (consecutive_entry / min_order_amount / _place_buy_order None 중 식별)

### 커밋

```
9b2614c  feat: Phase 25.1 옵션 C — monthly cap을 unique-ticker 카운트로 재정의
```

---

## Phase 25.2 — Observation Tools + Size Floor Adjustment (2026-05-07)

옵션 C 적용 후 4영업일(5/4~5/7) 8세션 연속 entries=0 관찰. 원인 진단 결과
페르소나 정합성의 1차 단속점을 옵션 C가 아니라 **사이저 floor**라고 재확인.
같은 세션에서 두 가지 부수 발견(자동화 부재, regime 추세) 동시 처리.

### 관찰 결과 (5/4~5/7)

| 기간 | Regime | Score | Opp Pass | 진입 |
|------|--------|-------|---------|------|
| 4/27~5/1 (5일) | **neutral** | 0.41~0.45 | 31 | 0건 |
| 5/4~5/7 (4일) | **caution** | 0.27~0.31 | 29 | 0건 |

핵심: regime 무관 진입 0. neutral(score 0.43)에서도 0건. 이전 진단
("caution이라 사이즈 작아짐")은 부분적으로만 맞음 — 양 regime에서 동일하게
사이저 출력이 1종목 weight ~0.02로 일관, min_order_amount 5M 미달.

### 진단 — 진짜 단속점

```
1차 단속점: 사이저 floor 0.05 → caution scale 0.42 곱 후 0.021 weight
            → 1억 × 0.021 = 2.1M < 5M (min_order_amount) → 모든 종목 SKIP

2차 단속점: 매월 1일 재학습 자동화 부재 (정책 명시되었으나 수동 only)

옵션 C 효과: cap 1차 단속점 아니었으므로 측정 거리 없음 (`rejections={}` 확인)
```

수학적 증명:
```
필요: weight × scale × balance ≥ 5,000,000
     weight ≥ 5M / 1억 / 최저 scale = 0.0494 / scale_min

caution scale 0.35 (관측 최저) → weight ≥ 0.141 필요
caution scale 0.42 (관측 평균) → weight ≥ 0.119 필요
```

### 변경 1: 매월 alpha_weights 재학습 자동화

배경: `alpha_weights.json`이 4/18 19:40 freeze 후 미갱신.
`alpha_weights_history/`에 `alpha_weights_2026-04.json`만 존재. 매월 1일
재학습 정책이 cron/timer 없이 수동 실행 only.

systemd unit 2개 신규:

```ini
# /etc/systemd/system/alpha-retrain.service
[Unit]
Description=V3 Alpha Weights Monthly Retraining
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=root
WorkingDirectory=/opt/quant
Environment=PYTHONPATH=/opt/quant
EnvironmentFile=/opt/quant/.env
ExecStart=/opt/quant/venv/bin/python v3/backtest/alpha_weight_trainer.py --lookback-years 3
StandardOutput=append:/var/log/alpha-retrain.log
StandardError=append:/var/log/alpha-retrain.log
TimeoutStartSec=30min
```

```ini
# /etc/systemd/system/alpha-retrain.timer
[Unit]
Description=V3 Alpha Weights Monthly Retraining Timer

[Timer]
OnCalendar=*-*-01 06:00:00
Persistent=true
Unit=alpha-retrain.service

[Install]
WantedBy=timers.target
```

다음 자동 실행: 2026-06-01 06:00 KST. 5/1 catch-up은 미실행 (3년 lookback이라
20일 차이 영향 미미).

### 변경 2: Recommendation log (관찰 인프라)

목적: 매 세션 모델 출력 분포 추적 → "매일 같은 종목" 패턴이 시장 횡보인지
모델 둔감인지 판정 + 사이즈 변경 후 효과 측정.

`v3/pipeline/live_pipeline.py` 변경:

```python
# imports 추가
import json
from pathlib import Path

# __init__
self._last_signal = None

# generate_signal 끝에서 캐시
self._last_signal = signal

# run_session 끝에서 호출
self._log_recommendation_snapshot(summary, entries, exits)

# 새 메서드 — 정책 영향 0, append-only JSONL
def _log_recommendation_snapshot(self, summary, entries, exits) -> None:
    """Per-session snapshot to v3/saved_models/recommendation_log.jsonl.
    Top-10 opportunities, sized positions, rejections, entries/exits."""
    ...
    f.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")
```

발견 + 수정: numpy.float32 직렬화 리스크 (VolTransformer 출력 round() 후에도
numpy 타입 유지 → json.dumps TypeError). Evaluator가 발견. 대응:
1. `default=float` 추가 (numpy float → Python float 자동 변환)
2. except 범위 확장: `(OSError, TypeError, ValueError)` (silent failure 방지)
3. 회귀 테스트 3개 추가 (`TestRecommendationLogResilience`)

### 변경 3: Sizer floor 0.05 → 0.15 (2단계)

#### 1차: 0.05 → 0.12

수학적 의도: caution scale 0.42 기준 0.12 × 0.42 = 0.0504 = 5.04M (5M 통과).

`v3/strategy/sizing.py:23` default 변경 1줄. 백테스트-라이브 parity 자동
유지 (양쪽 모두 default 사용, custom 인자 전달 없음).

회귀 테스트 5개 추가 (`TestSizerFloorPassesMinOrder`):
- caution scale에서 5M 통과
- neutral scale에서 5M 통과
- max_single_weight 0.40 위반 없음
- size_portfolio 결과가 floor 준수
- vol_target 2× 이내 (sanity)

Latent landmine 1건 식별 + 테스트로 문서화:

```python
def test_size_portfolio_n8_normalize_violates_floor(self):
    """n=8 + floor 0.12 → np.clip → 0.95 normalize → 0.1187 (floor 미만).
    현재 max_positions=1이라 unreachable. 미래 8+ 변경 시 수정 필요.
    이 테스트는 known violation 문서화 — pass 조건은 violation 발생."""
```

#### 2차: 0.12 → 0.15 (boundary 보정)

5/7 23:40 US 세션 첫 검증 결과: PYPL weight=0.0493 → order=4,997,999 KRW
→ **1,001 KRW 차이로 SKIP**.

원인: caution scale이 0.42가 아닌 **0.411**. caution 영역 scale은 5/4~5/7
관측 0.35~0.42 범위로 변동. 0.42를 anchor로 잡은 0.12 floor는 최저 scale에서
통과 보장 못함.

수정:
- `v3/strategy/sizing.py:23` 0.12 → 0.15
- 회귀 테스트 강화: `test_caution_floor_clears_min_order` anchor scale
  0.42 → **0.35** (관측 최저). docstring에 5/7 production 실패 1,001 KRW 차이 기록.

검증 시뮬:

| Scale (regime) | Weight | Order | 결과 |
|---------------|--------|-------|------|
| 0.35 (caution 최저) | 0.0525 | 5.32M | PASS |
| 0.41 (5/7 관측) | 0.0615 | 6.23M | PASS |
| 0.65 (neutral) | 0.0975 | 9.88M | PASS |
| 0.90 (bull) | 0.135 | 13.69M | PASS |

부작용 검토:
- bull 1종목 deploy 13.5%, 3종목 균등 40.5% — 페르소나 "1~3종목 집중" 부합
- vol-target 15% × 1.5 = 22.5% 이내로 유지 (sanity test 통과)
- max_single_weight 0.40 cap 그대로

### 회귀 테스트 추가 일람

| 테스트 클래스 | 테스트 수 | 목적 |
|--------------|---------|------|
| TestRecommendationLogResilience | 3 | numpy/serialize/disk failure 처리 |
| TestSizerFloorPassesMinOrder | 5 | caution/neutral 5M 통과, vol_target, max_weight |

전체 회귀: 28 → **37 tests, 모두 PASS**.

### 검증 결과 (5/7 23:40 KST 첫 적용 세션)

```
2026-05-07 23:40:18 | Regime caution (score=0.3055, scale=0.411)
                      Opportunity: 99 tickers, 25 pass gate
                      Signal: TRADE — 1 position
                      → PYPL weight 0.0493 = 4,997,999 KRW < 5M → SKIP
```

floor 0.12 적용 시: weight 2.5× 인상은 확인되었으나 boundary 미달.
floor 0.15 재적용 후 다음 세션부터 통과 예상 (수학적으로 증명, 5/8 09:30 KR 첫 검증).

첫 recommendation_log.jsonl 레코드 정상 기록 (top10 opp, selected_positions,
rejections 등 모든 필드 capture).

### Phase 25.2 부수 발견 (옵션 C 효과 재해석)

이전 진단 (Phase 25.1): "옵션 C 효과 측정 데이터 0건이라 측정 불가"

수정된 진단:
- 4/27~5/1 (옵션 C 이전): monthly_trades cap 도달 → 진입 0
- 5/1 새 달 자연 리셋 후에도 진입 0 → cap이 1차 단속점 아니었음
- 5/3 옵션 C 배포 → 패턴 변화 없음 (예상대로)
- **결론**: 옵션 C는 정책으로서 옳고 cap은 unstick됐으나, 진짜 1차
  단속점은 처음부터 사이저 floor였음. 옵션 C 효과 측정 거리 없음.

이 재해석은 페르소나 점수에도 영향:
- 원칙 1 (확신): opp gate 통과 25~31 종목으로 충분 → 모델은 conviction 표현 중
- 원칙 2 (크게): floor 0.05 → 0.15로 인상되어 **부분 해소**. paper로 검증 중.

### 운영 모드 — 1주 관찰

5/8~5/14 매일 데이터 누적. FOLLOW_UPS.md의 6개 관찰 포인트에 처음 데이터
입력 시작 (이전엔 거래 0이라 0개 채워짐).

다음 의사결정 시점: 5/14 또는 진입 5건 누적 시점 중 빠른 쪽.

### Phase 25.2 커밋

```
TBD  feat: Phase 25.2 — observation tools + sizer floor 0.05→0.15
       - alpha-retrain systemd timer (monthly auto, next 6/1)
       - recommendation_log JSONL (per-session snapshot)
       - sizer floor 0.05 → 0.15 (caution 5M passing)
       - 9 new regression tests (28 → 37, all pass)
```

---

## Phase 25.2 — Sizer floor 0.05→0.15 (2026-05-07)

페르소나 §2 "크게" (3/10) 직접 처방. 4/27~5/7 13세션 entries=0의 1차
단속점이 sizer floor 5M 미달임을 확인 후 인상.

### 변경
- `v3/strategy/sizing.py`: `min_position_weight` 0.05 → 0.15
  - caution scale 0.35 하한에서도 0.15 × 0.35 = 5.25M ≥ 5M 통과 보장
- `v3/pipeline/live_pipeline.py`: `recommendation_log.jsonl` per-session 누적
- `v3/tests/test_regression.py`: floor 0.15 회귀 (caution 0.35 anchor)
- `alpha-retrain.timer` systemd: 매월 1일 06:00 KST 자동 재학습

### 1주 검증 (5/8~5/14 진행 중, 5/9 시점)
- bull regime 1종목 가중치 13.5%, 3종목 균등 40.5%
- caution 최저 사이즈 5.25M (이전 1.75M에서 3배)

---

## Phase 26 (V3.3) — Profitability Engine (2026-05-09)

> **"Vol Expansion Trader" → "기대수익 기반 자본 재배분 엔진"**

V3.2.1 위에 13개 신규 모듈 + BookOptimizer orchestrator 추가.
**features.* OFF default → V3.2.1 동작 100% 보존.**
실 활성화는 5/14 검증 종료 후 paper promotion 주차별.

### 24+ commits, 450 tests, ~14,000 line

### Phase 1 (5 PR) — Diagnostics + Research foundation

| PR | 모듈 |
|----|------|
| 1.0 | V3.3 skeleton + features section + Protocol + frozen types (8 dataclass) |
| 1.1 | NoTradeReasonLogger + RejectReason enum (15) |
| 1.2 | TransferCoefficientMonitor + Spearman/rank pure-python |
| 1.3 | ExecutionQualityMonitor + FillRecord |
| 1.4 | build_edge_dataset.py + lookahead-safe forward outcomes |

### Phase 2 (4 PR) — Calibration & Edge Engine

| PR | 모듈 |
|----|------|
| 2.1 | EdgeCalibrator + calibrate_edge.py + validate_edge.py |
| 2.2 | EdgeEngine + cost decomposition + threshold derivation |
| 2.3 | EdgeTierSystem + 분위수 기반 임계값 도출 |
| 2.4 | BookOptimizer skeleton (parity standalone) |

### Phase 3 (5 PR) — Exit policies (FOLLOW_UPS 1순위 해결)

| PR | 모듈 |
|----|------|
| 3.1 | Conditional Veto 정상화 (max_signal_staleness 8h → 16h) |
| 3.2 | ExitThesisEngine 본체 (HOLD/REDUCE/ROTATE/EXIT) |
| 3.3 | SignalDecayEngine (alpha별 holding profile) |
| 3.4 | PartialExitEngine (winner protection + Phase 25.2 floor 동기화) |
| 3.5 | Phase 3 통합 테스트 + 회귀 시나리오 (4/21 ADI) |

### Phase 4 (4 PR) — Capital Expansion

| PR | 모듈 |
|----|------|
| 4.1 | AllocationEngine (LEVERAGE_CAP 1.00 영구 거부, net_edge / risk sizing) |
| 4.2 | PyramidPolicyEngine (winner-only, averaging-down 절대 금지) |
| 4.3 | CapitalRotationEngine (switching cost + 월간 cap 4) |
| 4.4 | BookOptimizer 완성 (모든 정책 wiring) |

### Phase 5 (2 PR) — Ablation

| PR | 모듈 |
|----|------|
| 5.1 | Ablation infrastructure (16 specs + runner + report) |
| 5.2 | Promotion plan + sweep CLI |

### 통합 (3 PR) — Production wiring

| PR | 작업 |
|----|------|
| 2.5 | BookOptimizer ↔ BacktestEngine + LivePipeline (DecisionContext) |
| 2.6 | Ablation sweep production wiring (BacktestEngine 호출) |
| 2.7 | CLAUDE.md / CHANGELOG.md V3.3 narrative (이 commit) |

### 페르소나 점수 변화 예상

- 원칙 1 (확신): 5/10 → **7~8/10** (calibration → 진짜 expected_return)
- 원칙 2 (크게): 3/10 → **6~7/10** (allocation + pyramid + Phase 25.2)
- 원칙 3 (빠르게): 8/10 → **9/10** (Conditional Veto 정상화 + ExitThesis)

### FOLLOW_UPS 1순위 해결

V3.2.1 Conditional Veto bug:
- max_signal_staleness 8h < KR↔US 14h gap → 항상 stale → unconditional fire
- 4/11~5/3 paper TP 5건 모두 stale 청산 (4/21 ADI +9.02% 자르기)

V3.3 fix:
- max_signal_staleness 16h
- refresh_at_session_start
- evaluate_conditional_veto 명시적 분기 (risk vs time)
- 회귀 테스트 ADI 시나리오 KEEP 검증

### 합격 기준 (V3.3_AB_PLAN §6)

| 측정 | V3.2.1 baseline (BT) | V3.3 Full 목표 |
|------|---------------------|---------------|
| Sharpe | 1.65 | ≥ 1.50 |
| MDD | 4% | ≤ 8% |
| Profit Factor | 3.93 | ≥ 2.0 |
| Avg Deployed | ~60% | ≥ 75% |
| TC | (미측정) | ≥ 0.30 |
| LEVERAGE_CAP 위반 | 0 | 0 (영구 invariant) |

### 다음 단계

```
5/14: Phase 25.2 검증 종료
5/15: feature/v3.3-phase1 → main 머지
5/15+: server에서 build_edge_dataset → calibrate_edge → validate_edge
5/15+: Paper Week 0 — 진단 3개 활성화
6월~7월: 주차별 정책 활성화 (rollback 자동)
```

> **NOTE**: 위 일정은 페르소나 정합성 일정. 사용자가 5/10 "페르소나 무시,
> 즉시 활성" 결정으로 12개 features 한 번에 ON 됨. 다음 섹션 참조.

---

## V3.3 전체 활성화 (2026-05-10)

> 사용자 결정 — *"일단 이번에는 페르소나 무시하고 즉시 활성화하자"* (옵션:
> "전체 (Edge layer 포함, 가장 느림)"). Phase 26 ROADMAP §5의 주차별
> promotion (Week 0~8) 건너뛰고 12개 features 동시 ON. 동시 다발 수정
> 금지 원칙 위배 — paper trading에서 효과 측정 분리 불가능. 자동 rollback
> (`v33-rollback-check.timer`, 1주 PnL -2% 시 OFF) 안전망 유지.

### A. LivePipeline F3 풀 통합 (commit 09cc8f2)

V3.3 도입 시 BookOptimizer는 `BacktestEngine`만 wiring 됐었음 (5ee271f F3
backtest 통합). Live pipeline은 `ctx.signal`만 사용 → V3.2.1 path. Live에서
도 ctx.actions 소비 가능하게 통합.

**TradingExecutor.execute_actions(actions, current_date)** — 단일 신규
entrypoint. 5종 핸들러:

| ActionType | 처리 | 신규 vs 재사용 |
|------------|------|----------------|
| EXIT | `_place_sell_order` + `record_loss` | 재사용 |
| TRIM | paper.sell(qty × trim_frac) + weight 갱신 | 신규 (paper 한정) |
| ROTATE | sell(old) + buy(new at target_weight) | 신규 |
| ADD_TO_WINNER | buy(add_amount) + 가중평균 entry 갱신 | 신규 |
| ADD_NEW | execute_entry 동일 gating + place_buy | 재사용 |
| KEEP/NO_ACTION/BLOCKED | skip | — |

KIS API + PaperBroker 동시 지원. Circuit breaker / monthly cap / position cap
/ cooldown 모두 V3.2.1 동등 적용. TRIM은 paper 한정 (KIS partial sell 미구현).

**LivePipeline 통합** (`v3/pipeline/live_pipeline.py`):

- `_use_v33_routing()` — 6개 decision-affecting flag (`exit_thesis` /
  `partial_exit` / `signal_decay` / `allocation` / `pyramid` / `rotation`)
  중 하나라도 ON → V3.3 path. Diagnostic-only 3개는 routing 미영향.
- `_convert_to_position_states(positions, ohlcv)` — V3.2.1 dict positions →
  V3.3 PositionState (latest close per ticker, pnl 계산).
- `_check_triggers_v33(positions, ohlcv)` — ExitRules.check 결과를
  `{ticker: trigger_name}` 매핑. BookOptimizer.exit_triggers 인자.
- `_signal_age_hours()` — `_opportunity_at` 기준 시간 차.
- `_refresh_rotations_counter(today)` — 캘린더 월 경계마다 0 리셋.
- `generate_signal()` — BookOptimizer.decide_with_context()에 positions /
  exit_triggers / signal_age_hours / rotations_this_month / adds_per_position
  / initial_weights 전달. `self._last_ctx` 저장.
- `execute()` — `_use_v33_routing()` True면 `executor.execute_actions(
  ctx.actions)` 호출. 결과에서 entered/rotated/added 집계 + pyramid/rotation
  cross-session state 갱신. False면 V3.2.1 signal path (parity).

**Monitor 루프 한계**: V3.2.1 `ExitRules` + executor.py inline conditional
veto 그대로 유지. `features.exit_thesis` 효과는 generate_signal 시점
(KR 09:30 / US 23:40)에만 적용. 15분 monitor 루프에서는 V3.2.1 동작.
ExitThesis V3.3 staleness fix (16h)는 ctx.actions 시점에만. 향후 별도 PR
에서 monitor 통합 가능.

### B. v3_config.yaml features 12개 모두 ON

```yaml
features:
  # Phase 1 (read-only diagnostics)
  no_trade_logger: true        # → no_trade_logs/
  tc_monitor: true             # → tc_history.jsonl
  execution_quality: true      # → execution_quality.jsonl
  # Phase 2 (Edge layer — calibration_table.json 없으면 graceful no-op)
  edge_calibrator: true
  edge_engine: true
  edge_tier: true
  allocation: true
  # Phase 3 (exit policies)
  exit_thesis: true
  partial_exit: true
  signal_decay: true
  # Phase 4 (capital expansion)
  pyramid: true
  rotation: true
```

`test_load_config_has_features` 의 all-OFF 단정 제거 (schema integrity 만
검증). 547/547 통과.

### C. Edge layer 활성용 데이터 빌드

`prep_calibration_inputs.py` 신규 (b1135d9 + d818be7 + 98308d4):

- 입력: `v3/data/raw/ohlcv_raw.parquet` + `v3/data/raw/macro.parquet` +
  `vol_transformer_epoch019.pt` + `feature_config.json`
- 출력: `data/research/{ohlcv_panel,macro_pctl,vol_predictions}.parquet`
- per-date 배칭으로 1274 dates × 99 tickers VolTransformer 추론.
  서버 CPU ~11분 (배칭 없으면 ~100분).
- searchsorted dtype 충돌 fix (int64 ns 변환).

서버 실행:
```
PYTHONPATH=/opt/quant /opt/quant/venv/bin/python \
    v3/scripts/prep_calibration_inputs.py
```

### D. Calibration pipeline 실행 (2026-05-10 01:35~01:52)

```
prep_calibration_inputs        # ~11min  → 3 parquets
run_calibration_pipeline       # ~17min  → 211 buckets
  build_edge_dataset           # 118,079 rows × 1205 dates × 99 tickers
  calibrate_panel              # train 107K / OOS 10.8K, 211 buckets
  validate_oos                 # FAIL (아래)
  archive                      # data/research/history/
  publish_latest               # 차단 (validation FAIL)
```

**Validation FAIL** (research/reports/validation_2026-05.md):

| 기준 | 값 | 통과 |
|------|----|------|
| Decile monotonicity (Spearman) > 0.5 | 0.515 | ✓ |
| Top decile mean > 0 | +0.0072 | ✓ |
| Top - Bottom decile > 0 | -0.0001 | **✗** |

원인: Decile 0 (가장 음의 opportunity) fwd_5d mean +0.0072 = top decile과
동일. → V3.2.1 alphas의 cross-sectional discrimination이 OOS 6개월에서
약함. NASDAQ 99종목 표본 한계 + 알파 자체 weakness.

**수동 publish 결정** — 사용자 "전체 활성화" 의도 + Edge layer 는
enrichment-only (BookOptimizer doc §SignalGenerator entry decision remains
canonical). FAIL은 net_edge 정확도 신호 약함이지 시스템 결함 아님.

```
ssh root@77.42.78.9 \
    "cp /opt/quant/data/research/edge_calibration_2026-05.json \
        /opt/quant/v3/config/edge_calibration.json && \
     systemctl restart quant-trading-v3"
```

확인 로그:
```
2026-05-10 07:07:49 INFO V3.3 Edge layer loaded from calibration_table.json
2026-05-10 07:07:49 INFO feature_tracker: 12 new activation(s) recorded
```

### E. F2 hook 활성 이력 기록

`feature_activations.jsonl` 12 entries (2026-05-10 00:31:56 deploy 시점).
`feature_state_snapshot.json` 갱신. Rollback 시 reverse delta 추적 가능.

### F. 자동 갱신

기존 timer 그대로 유지:
- `calibration-retrain.timer` — 매월 1일 07:00 KST. 6/1 첫 자동 실행.
- `v33-daily-report.timer` — 매일 16:00 KST.
- `v33-rollback-check.timer` — 매일 16:30 KST. 1주 PnL -2% 자동 OFF.
- `alpha-retrain.timer` — 매월 1일 06:00 KST.

### G. 위험 + 관찰 사항

**전체 활성 시 잠재 위험**:

1. **사이즈 정책 충돌** — V3.2.1 sizer (Phase 25.2 floor 0.15) + V3.3
   AllocationEngine 둘 다 활성. BookOptimizer가 SignalGenerator positions
   ADD_NEW로 변환할 때 어느 weight가 canonical인지 코드 경로 검증 필요.
2. **Pyramid 재현 빈도** — `pyramid_winner_only` invariant는 단위 테스트
   통과했으나 live 운용에서 winner 정의 (residual_edge > threshold)가 V3.2.1
   alpha와 정렬되는지 미검증.
3. **Rotation 월 cap** — `_rotations_this_month` 캘린더 리셋 외 별도 cap
   없음. RotationPolicy 기본값 의존.
4. **Calibration FAIL 영향** — Decile 0 anomaly가 EdgeTier 분류 왜곡 →
   Tier S/A를 잘못 부여 가능. 6/1 재실행에서 OOS 윈도우 갱신 시 자연 해소
   기대.
5. **LivePipeline ↔ monitor exit 정책 분리** — 위 §A 한계. Live trading
   에서 ExitThesis HOLD 신호가 generate_signal 시점에만 영향. 보유 중
   ExitRules trigger는 V3.2.1 inline veto가 처리.

**관찰 항목**:

```bash
# Edge layer + V3.3 ctx.actions 작동 확인
ssh root@77.42.78.9 "tail -f /var/log/quant-v3-error.log | \
    grep -E 'V3.3 (EXIT|TRIM|ROTATE|ADD)|EdgeTier|net_edge'"

# 일별 진단 리포트
ssh root@77.42.78.9 "ls -la /opt/quant/research/reports/daily/"

# 활성 이력
ssh root@77.42.78.9 "cat /opt/quant/v3/saved_models/feature_activations.jsonl"

# Rollback 트리거 여부
ssh root@77.42.78.9 "tail /var/log/v33-rollback.log"
```

### H. Rollback 절차

```bash
# 수동: 12 flags 모두 false 후 redeploy
sed -i 's/: true/: false/g' v3/config/v3_config.yaml  # 검토 필요 (다른 true도 영향)
# 또는 features 섹션만 수동 편집

git commit -am "rollback(V3.3): features OFF — V3.2.1 동작 복원"
git push
bash deploy_v3_git.sh 77.42.78.9

# Edge layer만 비활성 (Phase 1+3+4 유지)
ssh root@77.42.78.9 "rm /opt/quant/v3/config/edge_calibration.json && \
    systemctl restart quant-trading-v3"
# → graceful no-op 분기 활성, EdgeCalibrator None
```

### 페르소나 점수 (활성 직후 측정 불가, 1~2주 paper 데이터 필요)

| 원칙 | V3.2.1 | V3.3 design 목표 | 현재 |
|------|--------|------------------|------|
| 1. 확신 | 5/10 | 7~8/10 (calibration) | **미측정** (calibration FAIL) |
| 2. 크게 | 3/10 | 6~7/10 (allocation+pyramid) | **미측정** |
| 3. 빠르게 | 8/10 | 9/10 (ExitThesis 16h) | **부분 적용** (monitor 잔존) |

측정은 `recommendation_log.jsonl` + `paper_account.json` + daily report
1~2주 누적 후. FOLLOW_UPS §V3.3 활성화 추적 참조.


## V3.3 부분 활성화 + sizing 재해석 (2026-05-13)

V3.3 전체 활성화 (5/10) 이후 4 거래일(5/11~12) entries=0 silent failure가
인지되면서 진단 → 4 commits로 안정화. 핵심 명제는 **"OpportunityScorer는
5d return alpha가 아니다"** 가 데이터로 확정된 것.

### 시작점 — 4 거래일 entries=0 silent failure

5/10 12 features ON 후 첫 거래일(5/11 09:30 KR)부터 5/12 23:40 US 까지
4 세션 연속 `entries_count = 0`. `recommendation_log.jsonl` 에는 매일 MELI
(opp 0.0213, conv 0.97, weight 0.063~0.070)가 `selected_positions`에 등장
하지만 실제 진입 없음. `rejections={}` 빈 dict, `no_trade_logs/` 디렉터리도
빈 채로 silent.

### 진단 — 두 가지 silent failure 체인

**1차: V3.3 Edge layer가 모든 후보 drop**

`features.allocation=True` 활성 상태에서 `BookOptimizer._generate_entries()`
가 `SignalGenerator.positions` passthrough 대신 `AllocationEngine.allocate()`
호출. 그러나 `AllocationEngine` Step 1이 `c.entry_pass AND c.net_edge_5d > 0`
필터를 강제. `EdgeEngine.compute()`는 다음을 계산:

```
net_edge_5d = expected_return_5d - cost - slippage_buffer - λ_mae × |MAE|
```

5/10 publish된 calibration:
- `calibration_table.json` global mean_forward_return_5d = 0.00314
- mean |MAE| = 0.035, λ_mae = 0.20 → 페널티 0.007
- cost ≈ 0.001, slip ≈ 0.002
- **net_edge ≈ 0.003 − 0.010 = −0.007** ← 모든 ticker가 음수
- `entry_pass = net_edge > entry_threshold(0.0040)` → **항상 False**

→ AllocationEngine 빈 dict 반환 → ADD_NEW BookAction 0개 → 진입 0건.

진단 트리거: CLAUDE.md에 명시되어 있던 "calibration validation FAIL —
top-bottom −0.0001" 메모. decile 0과 decile 9의 평균 5d return 차이가
−0.01bp → opportunity → return mapping이 noise 임을 폭로한 것. 이 결과를
publish해도 무사히 통과하던 이유는 ablation runner가 별도 신호 측정만
하지 EDR sanity check 없음.

**2차: `flush_diagnostics()` 호출 누락**

`no_trade_logger=true` 였으나 `BookOptimizer.flush_diagnostics()` 메서드는
정의만 되어 있고 `LivePipeline.run_session()` / `BacktestEngine.run()`
어디에서도 호출 안 됨. `ablation_runner.py:175` 만 호출. 결과:

- `tc_monitor`, `execution_quality`, `no_trade_logger` 셋 모두 메모리
  버퍼에만 쌓이고 process 종료 시 사라짐
- 운영자가 reject reason 분포를 disk에서 확인 못 함 → 4일 silent

### 4 commits 안정화

| Commit | 범위 | 핵심 |
|--------|------|------|
| `5ffcae6` | V3.2.1 sizing 재해석 | `position_scale` 의미 변경 + vol_cc_20d 매핑 fix |
| `daefa48` | V3.3 Edge layer 6 features OFF | Phase 2/4 OFF, Phase 1/3 유지 |
| `fdb9eb0` | flush_diagnostics try/finally | Silent failure 차단 |
| `ebdecc6` | `volume_surprise` 알파 promotion + ic_to_weights 보수화 | 종목 선택 신호 품질 ↑ |

### A. V3.2.1 sizing 재해석 (commit 5ffcae6)

**이전** (`signal._size()` 곱셈 흐름):
```python
raw_weights = sizer.size_portfolio(...)  # 내부 floor 0.15 적용 후 반환
for c in selected:
    w = raw_weights.get(...)
    w = round(w * position_scale, 4)   # ← caution 0.42 × 0.15 = 0.063
    w = min(w, max_weight)
    if w < self.min_weight:            # SignalGen min_weight=0.02
        continue
    positions.append(...)
```

→ sizer 내부 floor 0.15가 곱셈으로 0.063 (6.3%)으로 무력화. 5/11 production
MELI weight 0.0702 정확히 일치.

**이후** (`position_scale = max_gross_exposure` 의미):
```python
raw_weights = sizer.size_portfolio(predicted_vols, confidences)

# Stage 1: per-position cap + floor
weights = {}
for c in selected:
    w = min(raw_weights[c.ticker], max_weight)
    if w < sizer.min_weight:        # 0.15 절대 floor
        continue
    weights[c.ticker] = round(w, 4)

# Stage 2: portfolio exposure cap — drop weakest until total ≤ scale
while weights and sum(weights.values()) > position_scale + 1e-6:
    weakest = min(weights, key=lambda t: candidate_map[t].opportunity)
    del weights[weakest]
```

추가로 `predicted_vols` 매핑 fix:
- 이전: `predicted_vols[t] = vol_scores['vol_score'][t]` (VolTransformer
  cross-sectional ranking을 vol value로 잘못 매핑)
- 이후: `_extract_vols()` 헬퍼 — `ohlcv['vol_cc_20d']` (feature_engineer
  line 99에서 `rolling(20).std() * sqrt(252)`로 annualized) 사용

V3.3 `RegimeBudget` (`v3/strategy/allocation.py:53-61`)과 정합 — 같은 의미를
V3.2.1 sizer에도 적용. 페르소나 ②"1~3종목 집중" + ①"확신 없으면 cash" 부합.

**실증**: 5/13 09:30 KR 첫 세션 — ABNB 매수 자본 38% (37.78M KRW). 이전
PYPL 매수 자본 5.8% (5.76M KRW) 대비 **6.5배**. caution scale 0.47이 그대로
포트폴리오 노출 한도로 작동, 1종목이 한도까지 채움.

회귀 테스트: `test_regression.py` Bug 11 = `TestPositionScaleAsExposure`
5개 invariant (single-candidate floor / total ≤ scale / underexposure
keeps cash / weakest drop / vol_cc_20d source).

### B. V3.3 Edge layer 6 features OFF (commit daefa48)

`v3/config/v3_config.yaml` features 섹션에서 OFF:
- `edge_calibrator` / `edge_engine` / `edge_tier` / `allocation` (Phase 2)
- `pyramid` / `rotation` (Phase 4 — Edge net_edge 의존)

유지:
- `no_trade_logger` / `tc_monitor` / `execution_quality` (Phase 1 read-only)
- `exit_thesis` / `partial_exit` / `signal_decay` (Phase 3 — alpha 가정 무관)

V3.2.1 `SignalGenerator` path 복귀. `BookOptimizer._generate_entries()`가
`allocation=False` 분기로 들어가 `SignalGenerator.positions` passthrough
→ 우리 fix된 `_size()` 결과가 그대로 ADD_NEW BookAction `target_weight`로.

**재활성 조건**: calibration top-bottom 의미값 (예: >1%) + `validate_edge.py`
PASS + paper 1~2주 검증. `docs/FOLLOW_UPS.md` 참조.

### C. flush_diagnostics try/finally (commit fdb9eb0)

```python
def run_session(self) -> dict:
    logger.info("=" * 60)
    logger.info(f"V3 Trading Session — ...")
    logger.info("=" * 60)
    try:
        df = self.collect_data()
        signal = self.generate_signal(df)
        ...
        return summary
    finally:
        self.book_optimizer.flush_diagnostics()
```

`BacktestEngine.run()`은 200줄 본문 indent shift 대신 method 분리:
- `run()` — thin wrapper, `try: return self._run_impl(...) finally: flush_diagnostics()`
- `_run_impl()` — 기존 본문 그대로 (flush 호출 제거)

회귀 테스트 Bug 12 = `TestBookOptimizerFlushed`:
- `_flush_in_finally(func)` 헬퍼: `textwrap.dedent + ast.parse` → `ast.Try.finalbody`
  순회 → `flush_diagnostics` Call 존재 확인. 단순 source 문자열 검사는 try 밖
  호출도 통과시켜 false negative.
- `test_live_pipeline_run_session_flushes_in_finally`
- `test_backtest_engine_run_flushes_in_finally`
- `test_flush_persists_to_disk` — record → flush → file 존재 + 내용 검증

### D. volume_surprise alpha promotion + ic_to_weights 보수화 (commit ebdecc6)

**알파 후보 IC 측정** (`v3/research/test_new_alphas.py`, 3년 panel 14,403 rows):

| Alpha | Vanilla IC | Caution | Verdict |
|-------|----------:|--------:|---------|
| `trend` (기존) | +0.003 | +0.026 | ❌ FAIL (vanilla 미달) |
| `reversion` (기존) | −0.001 | −0.017 | ❌ FAIL |
| **`volume_surprise`** (신규) | **+0.028** | **+0.059** | ✅ PROMOTE |
| `vol_term` (신규) | +0.019 | +0.041 | ⚠️ REGIME_ONLY |
| `earnings_proximity` (신규) | +0.012 | −0.002 | ⚠️ REGIME_ONLY (neutral만) |
| `vol_predicted` (신규) | +0.007 | +0.036 | ⚠️ REGIME_ONLY |

**결정적 발견 — `vol_predicted`의 IC 폭락**:
- VolConviction.expansion_ic (cross-sectional rank vs realized vol) = **+0.178**
- 같은 vol_score를 signed alpha로 변환: IC = **+0.007**
- → VolTransformer 신호는 **magnitude amplification (multiplier)** 이지
  signed directional alpha 아니라는 데이터 확정. CLAUDE.md "Phase 2 원칙:
  VolTransformer는 risk model" 이 옳음.

**`volume_surprise` 정식 promotion**:
```python
class AlphaVolumeSurprise(AlphaSource):
    surprise   = log(today_volume / SMA20(volume))
    direction  = sign(close[-1] / close[-6] - 1)
    raw        = surprise × direction
    → cross-sectional z-score → tanh(z/2) × ALPHA_SCALE
```

`DEFAULT_DIRECTIONAL` 확장 = `{trend, reversion, volume_surprise}`.

**`ic_to_weights` 보수화** (winner-take-most 방지):
```python
def ic_to_weights(self, ic, *, min_weight=0.10):
    shrunk = {a: max(v - MIN_VANILLA_IC, 0.0) for a, v in ic.items()}
    if sum(shrunk.values()) <= 1e-9:
        return uniform(ic)
    smoothed = {a: sqrt(s) for a, s in shrunk.items()}
    total = sum(smoothed.values())
    free_budget = 1.0 - n * min_weight
    return {a: min_weight + free_budget * smoothed[a] / total for a in ic}
```

2026-05-13 publish 신규 weights (`alpha_weights_2026-05.json`):

| Regime | trend | reversion | volume_surprise | n |
|--------|------:|----------:|----------------:|--:|
| strong_bull | 0.80 | 0.10 | 0.10 | 99 |
| bull | 0.10 | 0.80 | 0.10 | 2,373 |
| neutral | 0.10 | 0.10 | 0.80 | 6,316 |
| **caution** | **0.30** | **0.10** | **0.60** | **4,725** |
| bear | 0.10 | 0.80 | 0.10 | 890 |

이전 (winner-take-most, 같은 IC) 대비 caution: volume_surprise 86% → 60%,
trend 14% → 30%, reversion 0 → 10%. winner 60~80% cap, marginal alpha 10%
floor.

**알파 시스템 인프라**:
- `v3/data/earnings_collector.py` — yfinance `get_earnings_dates` 수집기
  (99/99 NASDAQ tickers, 5년치)
- `v3/data/raw/earnings_dates.json` — 캐시
- `v3/strategy/alpha_sources.py` — `AlphaVolumeSurprise` /
  `AlphaVolTermStructure` / `AlphaEarningsProximity` / `AlphaVolPredicted`
  + `load_earnings_dates` 헬퍼 + `compute_directional` `vol_scores` kwarg
- `v3/research/test_new_alphas.py` — production write 차단된 IC 실험
- `v3/research/reports/` — 3차 실험 결과 JSON

회귀 테스트 Bug 13 = `TestExperimentalAlphasContract` 11개:
- 각 candidate alpha 범위/empty/decay invariant
- `compute_directional` `vol_scores` forwarding
- `DEFAULT_DIRECTIONAL = {trend, reversion, volume_surprise}` 정확한 enforce
- `ic_to_weights` floor + dampening + uniform fallback

### E. 페르소나 점수 변화

| 원칙 | 5/10 (V3.3 전체 활성) | 5/13 (부분 활성 후) | 변화 |
|------|:--------------------:|:-------------------:|:----:|
| 1. 확신 있을 때만 | 5/10 (calibration FAIL) | 5/10 (volume_surprise +0.028 marginal) | ≈ |
| 2. 크게 | 3/10 → 0/10 (entries=0) | **7/10** (ABNB 자본 38%) | **+4** |
| 3. 빠르게 | 8/10 | 8/10 (ExitThesis 유지) | ≈ |

원칙 ②"크게"가 가장 큰 lever. 알파 추가/Edge 재활성 없이도 sizing 수식
재정의만으로 6.5배 사이즈 확보. 페르소나 정합성 진단의 본질이 sizing
구조에 있었다는 확인.

### F. 검증

- pytest: **587/587** (이전 547 + 신규 invariant 40)
- v3-evaluator: 3섹션 (regression / invariant / silent-failure) 모두 clean
  (commit 단계마다 2회 호출, 발견 silent failure 5건 모두 fix)
- Paper 첫 결과 (5/13 09:30 KR): ABNB 199주 매수, 자본 38%

### G. Rollback 절차

```bash
# 최근 deploy 직전 백업으로 복귀
ssh root@77.42.78.9 "ls -dt /opt/quant_v33_backup_* | head -1"
# 출력된 경로 사용:
# ssh root@77.42.78.9 "systemctl stop quant-trading-v3 && rm -rf /opt/quant && \
#     mv /opt/quant_v33_backup_<ts> /opt/quant && systemctl start quant-trading-v3"
```

자동 rollback (`v33-rollback-check.timer`)도 1주 PnL −2% 시 features OFF
유지.

### H. 다음 관찰 포인트

- **5/13~5/27 paper**: sizing fix + 알파 weight 변화 종합 효과. sharpe / 승률
  (>65%) / 손익비 (>2:1) / MDD (<5%) 회귀 없는지.
- Edge layer 재활성: calibration 품질 개선 + validate_edge PASS 이후
- 추가 알파 promotion: `vol_term/earnings_proximity/vol_predicted` IC
  시계열 robust 확인 후 (현재 vanilla 미달)
