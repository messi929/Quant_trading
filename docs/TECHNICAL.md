# Alpha Signal Discovery Engine - 기술 상세 문서

**버전**: 2026-03-04 (Phase 14: execution_market split 수정 + KOSPI 수집 KRX API 수정)
**현재 최고 성과**: Sharpe **2.03**, MDD **-4.30%**, Return **+17.82%** (기준선 대비 +0.21 Sharpe)
**배포 상태**: Hetzner Cloud `77.42.78.9` systemd 데몬 실행 중 (Phase 14 배포 완료 — KR/US 혼합 개별 종목 33건 체결 확인)

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [데이터 파이프라인](#2-데이터-파이프라인)
3. [피처 엔지니어링 (34개 피처)](#3-피처-엔지니어링)
4. [VAE - 잠재 지표 발견](#4-vae---잠재-지표-발견)
5. [Temporal Transformer - 시간 패턴 학습](#5-temporal-transformer---시간-패턴-학습)
6. [Conditional WGAN-GP - 시장 시뮬레이션](#6-conditional-wgan-gp---시장-시뮬레이션)
7. [PPO RL Agent - 섹터 매매 결정](#7-ppo-rl-agent---섹터-매매-결정)
8. [Ensemble Fusion - 신호 통합](#8-ensemble-fusion---신호-통합)
9. [리스크 관리](#9-리스크-관리)
10. [포트폴리오 최적화](#10-포트폴리오-최적화)
11. [백테스트 엔진](#11-백테스트-엔진)
12. [훈련 파이프라인](#12-훈련-파이프라인)
13. [하이퍼파라미터 참조표](#13-하이퍼파라미터-참조표)
14. [데이터 흐름 전체 도식](#14-데이터-흐름-전체-도식)
15. [라이브 트레이딩 시스템](#15-라이브-트레이딩-시스템)
16. [신호 생성 파이프라인 수정사항](#16-신호-생성-파이프라인-수정사항)
17. [Phase 8: 추론 파이프라인 버그 수정 + 개별 종목 매매](#17-phase-8-추론-파이프라인-버그-수정--개별-종목-매매)
18. [Phase 11: KR/US 분리 스케줄 + 해외 시세 yfinance 폴백](#18-phase-11-krus-분리-스케줄--해외-시세-yfinance-폴백)
19. [Phase 12: KOSPI 학습 포함 + 데몬 안정성 수정](#19-phase-12-kospi-학습-포함--데몬-안정성-수정)
20. [Phase 13: 해외 Paper Trading + KOSPI Parquet 구축](#20-phase-13-해외-paper-trading--kospi-parquet-구축)
21. [Phase 14: execution_market 수정 + KOSPI 수집 KRX API 수정](#21-phase-14-execution_market-수정--kospi-수집-krx-api-수정)

---

## 1. 시스템 개요

### 핵심 설계 철학

기존 기술적 지표(RSI, MACD, Bollinger Bands 등)는 사람이 수동으로 설계한 수식이다.
이 시스템은 **모델이 원시 데이터에서 직접 새로운 수학적 패턴(Alpha Signal)을 발견**하도록 설계되었다.

### 아키텍처 개요

```
[원시 데이터 OHLCV]
    ↓
[피처 엔지니어링 34개 통계적 피처]
    ↓              ↓              ↓
[VAE]        [Transformer]   [WGAN-GP]
잠재 지표      시간 패턴       시장 레짐
(32-dim)    signal_embed    시뮬레이션
    ↓              ↓
[Attention Fusion Layer]
    ↓
[PPO RL Agent]
11개 섹터 가중치 결정
    ↓
[포트폴리오 최적화 + 리스크 관리]
    ↓
[백테스트 / 실시간 추론]
```

### 시장 유니버스

| 시장 | 데이터 소스 | 수집 종목 수 | 학습 적용 |
|------|-----------|------------|---------|
| KOSPI | pykrx + yfinance | 950종목 수집 | **203종목** (21.6% 섹터 분류 성공) |
| NASDAQ/S&P500 | yfinance | S&P500 + NQ100 = 517종목 | 상위 200종목 |
| **합계** | | **1,467종목** | **max_tickers=400** |

**Phase 12 데이터 규모**: 1,466 ticker × 5년치, 358,454행 × 34 피처
**Phase 11 이전**: 200 ticker (NASDAQ 위주), 238,202행
**섹터 분류**: GICS 11개 섹터 (config/sectors.yaml 정의)

---

## 2. 데이터 파이프라인

### 2.1 데이터 수집 (`data/collector.py`)

```
yfinance/pykrx API
    → raw OHLCV DataFrame
    → data/raw/{market}/{ticker}.parquet
```

수집 필드: `date, open, high, low, close, volume, ticker, market, sector`

### 2.2 데이터 전처리 (`data/processor.py`)

1. **결측값 처리**: Forward fill → Backward fill
2. **이상치 제거**: 일 수익률 > ±50% 제거
3. **정규화**: MinMaxScaler 또는 StandardScaler (피처별 독립 적용)
4. **시계열 분할** (look-ahead bias 방지):
   - Train: 70% (처음 ~3.5년)
   - Validation: 15% (~0.75년)
   - Test: 15% (~0.75년, 2024년 후반 추정)

### 2.3 섹터 분류 (`data/sector_classifier.py`)

```python
# sectors.yaml 매핑 기반 분류
# GICS 11개 섹터:
# Energy, Materials, Industrials, Consumer Discretionary,
# Consumer Staples, Health Care, Financials, Information Technology,
# Communication Services, Utilities, Real Estate
```

---

## 3. 피처 엔지니어링

**파일**: `data/feature_engineer.py`
**피처 수**: 34개
**시간 창**: `WINDOWS = [5, 10, 20, 60]` (거래일 기준)

### 3.1 수익률 계산 (6개)

| 피처명 | 수식 | 설명 |
|--------|------|------|
| `return_1d` | `(close_t - close_{t-1}) / close_{t-1}` | 1일 수익률 |
| `return_5d` | `(close_t - close_{t-5}) / close_{t-5}` | 5일 수익률 |
| `return_10d` | `(close_t - close_{t-10}) / close_{t-10}` | 10일 수익률 |
| `return_20d` | `(close_t - close_{t-20}) / close_{t-20}` | 20일 수익률 |
| `return_60d` | `(close_t - close_{t-60}) / close_{t-60}` | 60일 수익률 |
| `log_return` | `log(close_t / close_{t-1})` | 로그 수익률 |

### 3.2 변동성 (4개)

```
volatility_W = std(log_return_{t-W:t}) × √252
```

| 피처명 | 창 크기 | 설명 |
|--------|--------|------|
| `volatility_5` | W=5 | 단기 변동성 |
| `volatility_10` | W=10 | 중단기 변동성 |
| `volatility_20` | W=20 | 중기 변동성 |
| `volatility_60` | W=60 | 장기 변동성 |

### 3.3 가격 위치 (4개)

```
price_position_W = (close_t - min(low_{t-W:t})) / (max(high_{t-W:t}) - min(low_{t-W:t}) + ε)
```

0~1 범위: 0이면 W일 최저가, 1이면 W일 최고가

### 3.4 거래량 지표 (5개)

| 피처명 | 수식 | 설명 |
|--------|------|------|
| `volume_ratio_5` | `volume_t / mean(volume_{t-5:t})` | 5일 평균 대비 거래량 |
| `volume_ratio_10` | `volume_t / mean(volume_{t-10:t})` | 10일 평균 대비 거래량 |
| `volume_ratio_20` | `volume_t / mean(volume_{t-20:t})` | 20일 평균 대비 거래량 |
| `volume_ratio_60` | `volume_t / mean(volume_{t-60:t})` | 60일 평균 대비 거래량 |
| `volume_change` | `(volume_t - volume_{t-1}) / volume_{t-1}` | 전일 대비 거래량 변화 |

### 3.5 캔들스틱 패턴 (4개)

| 피처명 | 수식 | 설명 |
|--------|------|------|
| `intraday_range` | `(high - low) / close` | 일중 변동폭 |
| `body_ratio` | `abs(close - open) / (high - low + ε)` | 몸통 비율 |
| `upper_shadow` | `(high - max(open, close)) / (high - low + ε)` | 위 꼬리 비율 |
| `lower_shadow` | `(min(open, close) - low) / (high - low + ε)` | 아래 꼬리 비율 |

### 3.6 갭 및 이동평균 이탈 (5개)

| 피처명 | 수식 | 설명 |
|--------|------|------|
| `gap` | `(open_t - close_{t-1}) / close_{t-1}` | 갭 수익률 |
| `ma_distance_5` | `(close_t - MA5) / MA5` | 5일 이평 이탈도 |
| `ma_distance_10` | `(close_t - MA10) / MA10` | 10일 이평 이탈도 |
| `ma_distance_20` | `(close_t - MA20) / MA20` | 20일 이평 이탈도 |
| `ma_distance_60` | `(close_t - MA60) / MA60` | 60일 이평 이탈도 |

### 3.7 자기상관 (2개)

| 피처명 | 수식 | 설명 |
|--------|------|------|
| `return_autocorr_5` | `corr(r_t, r_{t-5})` | 5일 래그 자기상관 |
| `return_autocorr_20` | `corr(r_t, r_{t-20})` | 20일 래그 자기상관 |

### 3.8 피처 요약

| 범주 | 피처 수 |
|------|--------|
| 수익률 | 6 |
| 변동성 | 4 |
| 가격 위치 | 4 |
| 거래량 | 5 |
| 캔들스틱 | 4 |
| 갭/이동평균 | 5 |
| 자기상관 | 2 |
| **합계** | **34** |

> **핵심**: RSI, MACD 등 전통적 기술적 지표를 일절 사용하지 않는다.
> 모든 피처는 통계적/수학적 계산으로, 어떤 금융 이론도 내재되어 있지 않다.

---

## 4. VAE - 잠재 지표 발견

**파일**: `models/autoencoder/model.py`, `models/autoencoder/trainer.py`
**목적**: 원시 피처에서 압축된 잠재 표현(latent indicators) 발견

### 4.1 네트워크 구조

```
입력: (batch, seq_len=60, n_features=34)
    ↓  Flatten
    → (batch, seq_len × n_features) = (batch, 2040)

[Encoder]
    Linear(2040, 256) → LeakyReLU
    Linear(256, 128)  → LeakyReLU
    Linear(128, 64)   → LeakyReLU
    ↓
    Linear(64, 32) → μ  (평균)
    Linear(64, 32) → log σ²  (로그 분산)

[Reparameterization Trick]
    ε ~ N(0, I)
    z = μ + ε × exp(log σ² / 2) = μ + ε × σ

[Decoder]
    Linear(32, 64)   → LeakyReLU
    Linear(64, 128)  → LeakyReLU
    Linear(128, 256) → LeakyReLU
    Linear(256, 2040) → reshape → (batch, 60, 34)

출력: 재구성된 시계열 x̂
```

### 4.2 β-VAE 손실 함수

```
L_VAE = L_recon + β_t × L_KL

L_recon = MSE(x, x̂) = (1/N) Σ (x_i - x̂_i)²

L_KL = -0.5 × mean(1 + log σ² - μ² - σ²)
     = -0.5 × mean(1 + log σ² - μ² - exp(log σ²))

β_t = kl_weight × min(1.0, t / kl_annealing_epochs)
    (KL Annealing: 처음엔 β=0, kl_annealing_epochs 동안 선형 증가)
```

**현재 설정**: `kl_weight=0.01`, `kl_annealing_epochs=20`

### 4.3 추론 시 사용

```python
# 훈련: z = μ + ε×σ (확률적 샘플링)
# 추론: z = μ (결정적, 분산 무시)
latent = vae.encode(x)  # returns μ only
```

**해석**: latent vector의 32개 차원이 각각 **"발견된 시장 지표"**에 해당.
어떤 차원이 어떤 패턴을 포착하는지는 사후 분석 필요 (Phase 7 예정).

---

## 5. Temporal Transformer - 시간 패턴 학습

**파일**: `models/transformer/model.py`
**목적**: 시계열 내 시간적 의존성과 섹터 간 관계 학습

### 5.1 Positional Encoding

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

PyTorch 구현에서 `register_buffer`로 고정값 저장 (학습 안 됨).

### 5.2 Sector Cross-Attention

```
[SectorCrossAttention]
입력:
  - x: (batch, seq_len, d_model)  ← Transformer Encoder 출력
  - sector_id: (batch,)           ← 섹터 인덱스

처리:
  1. sector_embedding = nn.Embedding(n_sectors, sector_embed_dim)[sector_id]
     → (batch, sector_embed_dim)
  2. all_sector_embeddings: (1, n_sectors, sector_embed_dim)
     → expand → (batch, n_sectors, sector_embed_dim)
  3. Q = x (sequence)                   ← 시계열이 Query
     K = V = all_sector_embeddings       ← 모든 섹터가 Key/Value
  4. cross_attn = MultiheadAttention(Q, K, V)
     → attended_x: (batch, seq_len, d_model)
  5. return attended_x, sector_embedding
```

이 레이어는 현재 섹터를 다른 모든 섹터와 비교하여 **섹터 간 관계**를 포착한다.

### 5.3 전체 구조

```
입력: (batch, seq_len=60, n_features=34)
    ↓
Linear(34, d_model=128) → input_proj
    ↓
PositionalEncoding(d_model=128)
    ↓
TransformerEncoder(
    num_layers=4,
    nhead=8,
    dim_feedforward=512,
    dropout=0.1,
    norm_first=True   ← Pre-LayerNorm (더 안정적 훈련)
)
→ encoded: (batch, 60, 128)
    ↓
SectorCrossAttention(encoded, sector_id)
→ cross_attended: (batch, 60, 128)
→ sector_embed: (batch, 32)
    ↓
Last-step pooling: cross_attended[:, -1, :]  → (batch, 128)
    ↓
Concat([last_step, sector_embed])  → (batch, 128+32=160)
    ↓
output_head:  Linear(160, 64) → ReLU → Linear(64, 1)  → prediction (scalar)
signal_head:  Linear(160, d_model=128)                 → signal_embedding

출력: {"prediction": (batch, 1), "signal_embedding": (batch, 128)}
```

**prediction**: 5일 후 수익률 방향 예측 (현재 direction accuracy: 52.4%)
**signal_embedding**: Ensemble Fusion에 전달되는 128-dim 벡터

---

## 6. Conditional WGAN-GP - 시장 시뮬레이션

**파일**: `models/gan/model.py`, `models/gan/trainer.py`
**목적**: 실제 시장 데이터 분포를 학습하여 현실적인 시나리오 생성

### 6.1 시장 레짐 분류

```python
def classify_regime(data: np.ndarray) -> int:
    """
    data: (seq_len, n_features) - seq_len일간의 피처
    returns: 0=Bull, 1=Bear, 2=Sideways, 3=Volatile
    """
    returns = data[:, 0]  # return_1d 피처
    volatility = np.std(returns)
    avg_return = np.mean(returns)

    if volatility > 1.5 * np.median(volatility_history):
        return 3  # Volatile
    elif avg_return > 0.001:   # +0.1%/일 이상
        return 0  # Bull
    elif avg_return < -0.001:  # -0.1%/일 이하
        return 1  # Bear
    else:
        return 2  # Sideways
```

### 6.2 Generator 구조

```
입력: [z, condition]
  z:         (batch, noise_dim=64)  ← N(0, I) 샘플링
  condition: (batch, condition_dim=4)  ← One-hot 레짐 인코딩

concat → (batch, 64+4=68)
    ↓
Linear(68, 128) → LayerNorm → LeakyReLU(0.2)
Linear(128, 256) → LayerNorm → LeakyReLU(0.2)
Linear(256, 128) → LayerNorm → LeakyReLU(0.2)
Linear(128, seq_len × output_dim)  → (batch, 60×34)
    ↓
Tanh  → [-1, 1] 범위 클리핑
reshape → (batch, 60, 34)

출력: 생성된 시장 시계열
```

### 6.3 Discriminator 구조

```
입력: [x, condition]
  x:         (batch, seq_len=60, output_dim=34)
  condition: (batch, 4)

flatten(x) + condition concat → (batch, 60×34 + 4 = 2044)
    ↓
Linear(2044, 128) → LayerNorm → LeakyReLU(0.2)
Linear(128, 256)  → LayerNorm → LeakyReLU(0.2)
Linear(256, 128)  → LayerNorm → LeakyReLU(0.2)
Linear(128, 1)    (Sigmoid 없음 - WGAN)

출력: 실수값 스코어 (높을수록 "진짜"라고 판단)
```

### 6.4 WGAN-GP 훈련 과정

**Wasserstein Distance (근사)**:
```
W(P_r, P_g) ≈ E[D(x_real)] - E[D(x_fake)]
```

**Gradient Penalty**:
```
x̂ = ε × x_real + (1-ε) × x_fake,   ε ~ U(0,1)

GP = E[(||∇_x̂ D(x̂)||₂ - 1)²]
```

**Discriminator 손실**:
```
L_D = E[D(x_fake)] - E[D(x_real)] + λ × GP
    (λ = gradient_penalty_weight = 10.0)
```

**Generator 손실**:
```
L_G = -E[D(G(z, c))]
```

**훈련 루프**:
```
for epoch:
    for batch:
        # Discriminator를 n_critic=5번 업데이트
        for _ in range(n_critic):
            x_real = sample_real_data(batch)
            z = sample_noise(batch)
            c = classify_regime(x_real)

            x_fake = G(z, c)
            L_D = D(x_fake) - D(x_real) + 10 × GP(x_real, x_fake, c)

            optimize D (lr=5e-5)

        # Generator를 1번 업데이트
        z = sample_noise(batch)
        c = one_hot(sampled_regime)
        x_fake = G(z, c)
        L_G = -D(x_fake, c)

        optimize G (lr=5e-5)

        # W-dist = E[D(real)] - E[D(fake)] (모니터링용)
        W_dist = D(x_real) - D(x_fake)
```

**알려진 문제**: Epoch 1 W-dist ≈ 13, 이후 40+ 발산 → mode collapse
**원인**: Generator가 Discriminator를 5번 학습하는 동안 너무 강해짐

---

## 7. PPO RL Agent - 섹터 매매 결정

**파일**: `models/rl/agent.py`, `models/rl/environment.py`, `models/rl/trainer.py`
**목적**: 섹터별 포트폴리오 가중치를 결정하는 강화학습 에이전트

### 7.1 상태 공간 (State Space)

```
state = concat([
    current_features,        # (n_features=34,) 오늘의 시장 피처
    current_positions,       # (n_sectors=11,) 현재 포지션 가중치
    [portfolio_value_ratio,  # 초기 자본 대비 현재 포트폴리오 가치
     step_ratio]             # 에피소드 진행도 (0~1)
])
→ state_dim = 34 + 11 + 2 = 47
```

### 7.2 Actor-Critic 네트워크

```
[공유 백본]
Linear(47, 256) → LayerNorm → Tanh
Linear(256, 128) → LayerNorm → Tanh
→ hidden: (batch, 128)

[Actor Head]
Linear(128, n_sectors=11) → Tanh → actor_mean: (batch, 11)
actor_log_std: nn.Parameter(ones(n_sectors) × 0)  ← 학습 가능 파라미터

[Critic Head]
Linear(128, 64) → Tanh → Linear(64, 1) → value: (batch, 1)
```

### 7.3 행동 분포

```python
# 행동 분포: 다변량 정규분포 (섹터별 독립)
dist = Normal(
    loc=actor_mean,                                    # tanh 출력
    scale=exp(actor_log_std.clamp(min=-5, max=2))     # 클리핑된 표준편차
)

# 행동 샘플링
raw_action = dist.sample()     # 정규분포에서 샘플
action = tanh(raw_action)      # [-1, 1] 범위로 스케일링

# 로그 확률 (Importance Sampling에 사용)
log_prob = dist.log_prob(raw_action).sum(dim=-1)

# 엔트로피 (탐험 장려)
entropy = dist.entropy().sum(dim=-1)
# 현재 문제: log_std 초기값 0 (std=1)이면 entropy ≈ log(√(2πe)) × 11 ≈ 13.5
```

**행동 의미**: `action[i] ∈ [-1, 1]`
- 양수: 섹터 i에 long (최대 max_position=1.0)
- 음수: 섹터 i에 short (현재는 실제로 제약 없음, 버그)
- 0: 중립

### 7.4 보상 함수

```python
def _compute_reward(t):
    returns_window = portfolio_returns[-window_size:]  # 최근 60일 수익률

    # 기본 보상: 롤링 Sharpe Ratio
    if len(returns_window) >= 2 and std(returns_window) > 0:
        rolling_sharpe = mean(returns_window) / std(returns_window) * sqrt(252)
    else:
        rolling_sharpe = 0.0

    # 과도 변동성 페널티 (vol_target=0.15 초과분)
    port_vol = std(returns_window[-20:]) * sqrt(252)
    vol_excess = max(0, port_vol - 0.15)
    vol_penalty = risk_penalty × vol_excess
    # risk_penalty = 1.0 (현재)

    # 비선형 드로다운 페널티 (다단계)
    current_dd = max_value / current_value - 1  # 피크 대비 낙폭
    if current_dd > 0.30:    dd_penalty = drawdown_penalty × current_dd × 10  # 위기: ×50
    elif current_dd > 0.20:  dd_penalty = drawdown_penalty × current_dd × 5   # 고위험: ×25
    elif current_dd > 0.10:  dd_penalty = drawdown_penalty × current_dd × 2   # 경고: ×10
    else:                    dd_penalty = drawdown_penalty × current_dd        # 정상: ×5
    # drawdown_penalty = 5.0 (현재)

    reward = rolling_sharpe - vol_penalty - dd_penalty
```

### 7.5 PPO 알고리즘

```
[데이터 수집]
n_steps = 2048 스텝마다 환경과 상호작용 → 경험 버퍼

[GAE (Generalized Advantage Estimation)]
δ_t = r_t + γ V(s_{t+1}) - V(s_t)           ← TD 오차
A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}         ← λ=0.95로 지수 감쇠 평균
returns_t = A_t + V(s_t)                      ← 학습 타깃

[PPO 업데이트 (n_epochs=10회)]
  for epoch in range(10):
    for mini_batch in shuffle(buffer):
      # Clipped Surrogate Objective
      ratio = π_θ(a|s) / π_θ_old(a|s)  = exp(log_prob - old_log_prob)

      L_clip = min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)
             , ε = clip_epsilon = 0.2

      # Critic 손실
      L_value = MSE(V(s), returns)  × value_coef=0.5

      # 엔트로피 보너스
      L_entropy = -entropy_coef × entropy  , entropy_coef=0.001

      # 최종 손실
      L = -L_clip + L_value - L_entropy

      # 그래디언트 클리핑
      clip_grad_norm(params, max_grad_norm=0.5)
```

**훈련 설정**: `total_timesteps=200,000`, `gamma=0.99`, `learning_rate=3e-4`

---

## 8. Ensemble Fusion - 신호 통합

**파일**: `models/ensemble.py`
**목적**: VAE와 Transformer의 출력을 주의집중(Attention)으로 통합

### 8.1 AttentionFusion 구조

```
입력 신호들:
  vae_latent:       (batch, 32)    ← VAE.encode() 결과
  signal_embedding: (batch, 128)   ← Transformer signal_head 결과

처리:
  1. 각 신호를 공통 차원으로 투영
     vae_proj = Linear(32, fusion_dim=64)
     tf_proj  = Linear(128, fusion_dim=64)

  2. concat([vae_proj, tf_proj]) → (batch, 2×64=128)

  3. Attention 가중치 계산
     attn_input → Linear(128, 64) → Tanh → Linear(64, 2) → Softmax
     attention_weights: (batch, 2)  ← 합이 1인 가중치

  4. 가중합
     fused = attention_weights[:, 0] × vae_proj + attention_weights[:, 1] × tf_proj
           → (batch, 64)

출력: fused_signal (batch, 64)
```

### 8.2 신호 생성 파이프라인

```python
def get_signals(x, sector_id):
    # 1. VAE로 잠재 지표 추출
    latent_indicators = vae.encode(x)  # (batch, 32)

    # 2. Transformer로 시간 패턴 추출
    tf_output = transformer(x, sector_id)
    prediction = tf_output["prediction"]        # (batch, 1)
    signal_embedding = tf_output["signal_embedding"]  # (batch, 128)

    # 3. Attention으로 신호 통합
    fused = fusion(latent_indicators, signal_embedding)  # (batch, 64)

    return {
        "latent_indicators": latent_indicators,   # 발견된 지표 (32차원)
        "signal_embedding": signal_embedding,     # 시간 패턴
        "prediction": prediction,                 # 가격 방향 예측
        "fused_signal": fused                     # 통합 신호 (RL 입력에 사용)
    }
```

> **주의**: GAN은 현재 훈련 데이터 증강에만 사용되며 (3상태 시뮬레이션으로 희귀 시장 레짐 데이터 생성),
> get_signals() 추론 시에는 VAE + Transformer만 사용된다.

### 8.3 체크포인트 포맷

```python
# 저장 형식 (ensemble.pt)
{
    "vae": vae.state_dict(),
    "transformer": transformer.state_dict(),
    "gan_generator": gan.generator.state_dict(),      # GAN은 분리 저장
    "gan_discriminator": gan.discriminator.state_dict(),
    "rl_policy": agent.state_dict(),
    "fusion": fusion.state_dict(),
    "config": {...}
}

# 로드 시 주의: GAN은 model.generator.load_state_dict() 방식 필요
```

---

## 9. 리스크 관리

**파일**: `strategy/risk.py`

### 9.1 다단계 드로다운 차단기

```python
def check_and_adjust(weights, portfolio_value, recent_returns):
    # 현재 드로다운 계산
    if recent_returns is not None:
        cumulative = (1 + recent_returns).cumprod()
        peak = cumulative.max()
        current_dd = 1 - cumulative.iloc[-1] / peak

    # 다단계 차단기
    if current_dd > 0.30:    scale = 0.0    # 전면 청산
    elif current_dd > 0.20:  scale = 0.25   # 75% 축소
    elif current_dd > 0.10:  scale = 0.50   # 50% 축소
    elif current_dd > 0.05:  scale = 0.75   # 25% 축소
    else:                    scale = 1.0    # 정상 운용

    adjusted_weights = weights * scale

    # 변동성 타겟팅
    adjusted_weights = _vol_target(adjusted_weights, recent_returns)

    return adjusted_weights, risk_report
```

### 9.2 변동성 타겟팅

```python
def _vol_target(weights, returns, max_portfolio_vol=0.20):
    if returns is None or len(returns) < 5:
        return weights

    port_returns = returns.dot(weights)  # 포트폴리오 수익률
    port_vol = port_returns.std() * sqrt(252)  # 연환산 변동성

    if port_vol > max_portfolio_vol:
        scale = max_portfolio_vol / port_vol
        return weights * scale

    return weights
```

### 9.3 일별 사전 필터 (백테스트 전용)

백테스트 `cmd_backtest`에서 주간 리밸런싱 엔진 전에 **일별 차단기**를 적용:

```python
# 일별 포트폴리오 가치 시뮬레이션 → 차단기 적용
pre_risk = RiskManager(max_position_pct=1.0)
sim_portfolio = initial_capital

for t_idx in range(n_test):
    weights = model_signals[t_idx].copy()
    recent_ret = test_returns.iloc[max(0, t_idx-20):t_idx+1]

    # 일별 차단기 적용
    adj_weights, _ = pre_risk.check_and_adjust(weights, sim_portfolio, recent_ret)
    model_signals[t_idx] = adj_weights

    # 포트폴리오 가치 업데이트 (다음 날 차단기 계산용)
    day_ret = np.dot(adj_weights, test_returns.iloc[t_idx].values)
    sim_portfolio *= (1.0 + day_ret)
```

**필요한 이유**: BacktestEngine은 주 1회 리밸런싱 기준으로만 RiskManager 적용.
리밸런싱 사이 5거래일 동안 큰 손실이 나도 차단기가 작동 안 함 → 일별 사전 필터로 보완.

---

## 10. 포트폴리오 최적화

**파일**: `strategy/portfolio.py`

### 10.1 신호 가중치 방식 (현재 사용)

```python
def _signal_weighted(signals):
    """
    모델 신호를 포트폴리오 가중치로 변환

    핵심 특성: abs_sum ≤ 1.0이면 원래 스케일 유지 (레버리지 X)
    """
    abs_sum = sum(abs(signals))

    if abs_sum == 0:
        return equal_weight

    # min(abs_sum, 1.0): 신호 합이 1 이하면 그대로, 초과면 정규화
    weights = signals / abs_sum * min(abs_sum, 1.0)
    return weights
```

**예시**:
- 신호 합 0.5 → 가중치 합 0.5 (50% 투자, 50% 현금 보유)
- 신호 합 1.5 → 가중치 합 1.0 (정규화, 레버리지 없음)

### 10.2 MaxSharpe 최적화 (옵션)

scipy.optimize를 사용한 수치 최적화:
```
maximize: Sharpe = E[R_p] / σ[R_p]
subject to: Σ w_i = 1, w_i ≥ 0 (롱 온리)
```

---

## 11. 백테스트 엔진

**파일**: `backtest/engine.py`, `backtest/metrics.py`

### 11.1 백테스트 설정

```yaml
initial_capital: 100,000,000  # 1억원
commission_rate: 0.002        # 0.2% 편도 수수료
slippage_rate: 0.001          # 0.1% 슬리피지
rebalance_frequency: weekly   # 주 1회 리밸런싱
```

### 11.2 백테스트 루프

```python
for date in test_dates:
    current_signals = model_signals[date_idx]

    if is_rebalance_day(date):
        # 포트폴리오 최적화
        target_weights = portfolio_optimizer.optimize(current_signals, recent_returns)

        # 리스크 차단기
        target_weights, risk_report = risk_mgr.check_and_adjust(
            target_weights, portfolio_value, recent_returns
        )

        # 거래 비용 계산
        turnover = sum(abs(target_weights - current_weights))
        commission = portfolio_value * turnover * commission_rate
        slippage = portfolio_value * turnover * slippage_rate
        portfolio_value -= (commission + slippage)

        current_weights = target_weights

    # 일별 수익률 적용
    daily_return = sum(current_weights * sector_returns[date])
    portfolio_value *= (1 + daily_return)
```

### 11.3 성과 지표

| 지표 | 수식 |
|------|------|
| **Total Return** | `(V_final / V_initial) - 1` |
| **Annual Return** | `(1 + total_return)^(252/n_days) - 1` |
| **Sharpe Ratio** | `mean(daily_returns) / std(daily_returns) × √252` |
| **Sortino Ratio** | `mean(daily_returns) / std(negative_returns) × √252` |
| **Max Drawdown** | `max(1 - V_t / max(V_{0:t}))` |
| **Calmar Ratio** | `Annual Return / abs(Max Drawdown)` |
| **Win Rate** | `count(monthly_returns > 0) / total_months` |

---

## 12. 훈련 파이프라인

**파일**: `pipeline/train_pipeline.py`

### 12.1 6단계 파이프라인

```
Phase 1: 데이터 수집 및 전처리
    → data/processed/processed_data.parquet

Phase 2: VAE 훈련 (30 epochs)
    → saved_models/vae_epoch30.pt
    목표: reconstruction loss 최소화, latent space 정규화

Phase 3: Transformer 훈련 (25 epochs)
    → saved_models/transformer_epoch25.pt
    목표: 5일 후 수익률 방향 예측 (direction accuracy 최대화)

Phase 4: WGAN-GP 훈련 (50 epochs)
    → saved_models/gan_epoch{best}.pt
    목표: W-dist 최소화 (실제 분포 근사)

Phase 5: PPO RL 훈련 (200,000 timesteps)
    → saved_models/ppo_agent_epoch{N}.pt
    목표: 누적 Sharpe Ratio 최대화

Phase 6: Ensemble 통합
    → saved_models/ensemble.pt
    AttentionFusion 미세조정 + 전체 파이프라인 통합
```

### 12.2 재훈련 전략

```bash
# RL만 재훈련 (~15분): 리스크/보상 파라미터 변경 시
python main.py train --start-phase 5 --config config/settings_fast.yaml

# GAN부터 재훈련 (~80분): GAN 하이퍼파라미터 변경 시
python main.py train --start-phase 4 --config config/settings_fast.yaml

# 전체 재훈련 (~3시간): 데이터/VAE/Transformer 변경 시
python main.py train --start-phase 2 --config config/settings_fast.yaml
```

### 12.3 GPU 최적화 (RTX 4060 Ti 8GB)

- **Mixed Precision (FP16)**: `torch.autocast("cuda", dtype=torch.float16)`
- **Gradient Accumulation**: 8 스텝마다 1회 업데이트 (effective batch = 8 × 32 = 256)
- **VRAM 안전 배치 크기**: VAE=32, Transformer=32, GAN=32, RL=32

---

## 13. 하이퍼파라미터 참조표

### VAE (`config/model_config_fast.yaml`)

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `hidden_dims` | [256, 128, 64] | 인코더/디코더 레이어 크기 |
| `latent_dim` | 32 | 발견된 지표 수 |
| `kl_weight` | 0.01 | β-VAE 정규화 강도 |
| `kl_annealing_epochs` | 20 | KL annealing 기간 |
| `learning_rate` | 1e-4 | Adam 학습률 |
| `epochs` | 30 | 훈련 에포크 |

### Transformer (Phase 6 업데이트됨)

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `d_model` | **256** | 모델 차원 (128→256) |
| `n_heads` | 8 | 멀티헤드 어텐션 수 |
| `n_encoder_layers` | **6** | Transformer 레이어 수 (4→6) |
| `d_ff` | **1024** | FFN 은닉층 크기 (512→1024) |
| `prediction_horizon` | **1** | 예측 목표 (5일→1일, 단기 예측이 더 안정적) |
| `learning_rate` | **3e-5** | 학습률 (5e-5→3e-5) |
| `warmup_steps` | **1000** | Warmup (500→1000) |
| `epochs` | **50** | 훈련 에포크 (25→50) |

### GAN (Phase 4 업데이트됨)

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `noise_dim` | 64 | Generator 입력 노이즈 차원 |
| `generator_hidden_dims` | **[64, 128, 64]** | Generator 구조 (과적합 방지, 축소됨) |
| `spectral_norm` | **true** | Lipschitz 조건 강제 |
| `learning_rate_g/d` | 5e-5 | 학습률 (1e-4에서 감소) |
| `n_critic` | **10** | Discriminator 업데이트 횟수 (5→10) |
| `gradient_penalty_weight` | 10.0 | GP 강도 λ |
| `early_stopping_delta` | **5.0** | W-dist 상승 허용폭 (best+5 초과 시 조기 종료) |
| `epochs` | 50 | 훈련 에포크 |

### PPO RL

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `learning_rate` | 3e-4 | Adam 학습률 |
| `gamma` | 0.99 | 할인 계수 |
| `gae_lambda` | 0.95 | GAE λ |
| `clip_epsilon` | 0.2 | PPO 클리핑 ε |
| `entropy_coef` | 0.001 | 탐험 장려 (0.01에서 감소) |
| `risk_penalty` | 1.0 | 변동성 초과 페널티 계수 |
| `drawdown_penalty` | 5.0 | 드로다운 페널티 계수 |
| `window_size` | 60 | 롤링 Sharpe 계산 창 |
| `vol_target` | 0.15 | 목표 연간 변동성 |
| `total_timesteps` | 200,000 | 총 훈련 스텝 |
| `n_steps` | 2048 | 업데이트 전 수집 스텝 |

---

## 14. 데이터 흐름 전체 도식

### 훈련 시

```
[Yahoo Finance / KRX Data]
    ↓ (collector.py)
[Raw OHLCV DataFrames]
    ↓ (processor.py + feature_engineer.py)
[Processed Parquet: 238K rows × 34 features]
    ↓
    ┌─────────────────────────┬────────────────────────┐
    ↓                         ↓                        ↓
[VAE Training]          [Transformer Training]    [GAN Training]
재구성 손실 최소화       방향 예측 학습             분포 학습
kl_weight=0.01          lr=5e-5, 25ep             lr=5e-5, 50ep
    ↓                         ↓                        ↓
[vae.pt]               [transformer.pt]          [gan.pt]
    └─────────────────────────┴────────────────────────┘
                              ↓
                    [PPO RL Environment]
                    state = (34 features + 11 positions + 2)
                    reward = rolling_sharpe - vol_penalty - dd_penalty
                    total_timesteps = 200,000
                              ↓
                    [ppo_agent.pt]
                              ↓
                    [Ensemble Fusion Training]
                    AttentionFusion (VAE latent + TF signal)
                              ↓
                    [ensemble.pt]
```

### 추론/백테스트 시 (Phase 7 수정 반영)

```
[Test Period Data: 180 days, 11 sectors × N tickers]
    ↓
[ensemble.pt 로드]
[sectors.yaml → sector_to_train_id 로드]  ← Bug 2 수정
    ↓
For each test day t:
    For each sector s:
        train_sector_id = sector_to_train_id[s]  ← Bug 2 수정
        sector_id = torch.tensor([train_sector_id])

        For each ticker in sector:               ← Bug 3 수정 (per-ticker)
            window = ticker_data[-60:t]          ← 정확한 60일 단일 ticker
            ① vae.encode(window) → latent (32-dim)
            ② transformer(window, sector_id) → prediction (scalar)
            ticker_preds.append(prediction)

        model_signals[t, s] = mean(ticker_preds)  ← sector 평균
    ↓
[Top-K 정규화 (k=5)]
    상위 5개 섹터만 양수 가중치, 나머지 0
    normalize to sum=1
    ↓
[Alpha Blending]                                 ← Phase 7 추가
    final = 0.4 × model_weights + 0.6 × equal_weight
    ↓
[일별 리스크 차단기 (사전 필터)]
    portfolio_value 시뮬레이션 → 5단계 차단기
    ↓
[BacktestEngine.run()]                           ← Bug 1 수정 (state reset)
    주간 리밸런싱
    포트폴리오 최적화 → 리스크 관리 → 거래 비용 차감
    ↓
[백테스트 결과]
    Sharpe: 2.03 | MDD: -4.30% | Return: +17.82%
    → results/equity_curve.png
    → results/backtest_report.html
```

---

## 15. 라이브 트레이딩 시스템

**파일**: `broker/`, `live/`, `tracking/`, `scheduler/`, `dashboard/`

### 15.1 전체 아키텍처

```
[InferencePipeline.generate_signals()]
    ↓ sector_allocations + sector_top_tickers (섹터별 top-3 종목 + score)
[DailyRunner.step_signal()]
    ↓ top-K 정규화 + alpha blend (40% model + 60% EW)
    → final_weights (11,), sector_top_tickers dict
[DailyRunner.step_sell_check(sector_top_tickers)]
    ↓ 보유 종목 중 오늘 양수 신호 없는 종목 → 즉시 매도
[OrderGenerator.execute_rebalance(final_weights, sector_top_tickers)]
    ↓ 현재 포지션 조회 → 개별 종목 목표 포지션 계산 → 매도 먼저 → 매수
[KISApi.order_domestic() / order_overseas()]
    ↓ 체결
[TradeLogger.log_trade()]
    → tracking/trades.db
```

### 15.2 KIS API (`broker/kis_api.py`)

```python
# TR_ID: sandbox/production 모드별 다른 엔드포인트
TR_IDS = {
    "sandbox": {
        "domestic_buy": "VTTC0802U", "domestic_sell": "VTTC0801U",
        "balance_domestic": "VTTC8434R", ...
    },
    "production": {
        "domestic_buy": "TTTC0802U", "domestic_sell": "TTTC0801U",
        "balance_domestic": "TTTC8434R", ...
    }
}

# OAuth2 토큰 캐싱 (24시간 유효)
token = KISApi.get_token()  # disk cache → 재시작시 재발급 불필요
```

**설정**: `.env` 파일에 `KIS_APP_KEY`, `KIS_APP_SECRET`, `KIS_CANO`, `KIS_ACNT_PRDT_CD`

### 15.3 개별 종목 선택 (`pipeline/inference_pipeline.py`)

Phase 8에서 ETF 매매 → 개별 종목 매매로 전환.

```python
# 섹터당 ticker 예측값 계산
ticker_scores = {}
for ticker in tickers_in_sector:
    tkr_df = sector_df[sector_df["ticker"] == ticker].sort_values("date")
    window = tkr_df[feature_cols].values[-self.seq_length:]  # 60일 창
    features = torch.FloatTensor(window).unsqueeze(0).to(device)
    with torch.no_grad():
        model_signals = self.ensemble.get_signals(features, sector_id_tensor)
        ticker_scores[ticker] = model_signals["prediction"].cpu().item()

# score 내림차순 정렬 → top-3, 양수만 (상승 신호 있는 종목만 매수)
ranked = sorted(ticker_scores.items(), key=lambda x: x[1], reverse=True)
top_tickers = [
    {"ticker": t, "score": s, "market": "overseas" if not t.isdigit() else "domestic"}
    for t, s in ranked[:3]
    if s > 0
]
```

**섹터별 자금 배분**:
- 섹터 배분 금액 / top 종목 수 = 종목당 투자금액
- 1종목당 최소 10만원 미만 → 1종목에 섹터 전액 집중
- 동일 ticker가 여러 섹터 top-3에 등장 시 수량 합산

### 15.4 일간 자동화 스케줄 (Phase 11 기준)

```
06:00  DataCollector.collect_all(last 10 days)          KR + US 전일 종가 수집
06:10  _daemon_us_signal()                              [US] NASDAQ 종가 신호 생성
         InferencePipeline (use_cache=False)             → 캐시 무시, 방금 수집 데이터로 재추론
         step_sell_check(top_tickers)                    → 해외 하락 신호 즉시 매도
         _us_weights / _us_top_tickers 저장              → Wave 1/2/3에서 재사용
06:30  _daemon_signal_and_sell()                        [KR] 신호 생성 + 국내 하락 매도
         (캐시 있으면 06:10 결과 재사용)
09:10  execute_rebalance(..., market_filter="domestic")  [KR] Wave 1 (40%)
11:00  execute_rebalance(..., market_filter="domestic")  [KR] Wave 2 (35%)
13:30  execute_rebalance(..., market_filter="domestic")  [KR] Wave 3 (25%)
16:00  DailyRunner.step_eod()                           종가 기록 + 성과 업데이트
23:40  execute_rebalance(..., market_filter="overseas")  [US] Wave 1 (40%, 06:10 신호)
02:00  execute_rebalance(..., market_filter="overseas")  [US] Wave 2 (35%, 06:10 신호)
04:30  execute_rebalance(..., market_filter="overseas")  [US] Wave 3 (25%, 06:10 신호)
```

**설계 원칙**: US/KR 모두 신호는 하루 1회 생성, TWAP는 동일 신호의 시간 분산 집행.
- US: NASDAQ 종가 직후(06:10) 신호 생성 → 미국 장 중(23:40/02:00/04:30) 분산 집행
- KR: 장 전(06:30) 신호 생성 → 한국 장 중(09:10/11:00/13:30) 분산 집행
- Wave 간 재추론 없음 (daily 모델에서 intraday 재추론은 의미 없음)

**재훈련 스케줄** (변경 없음):
- **토요일 22:00**: `RetrainRunner.run_weekly()` — Transformer Phase 3+ (~30분)
- **매월 1일**: `RetrainRunner.run_monthly()` — VAE Phase 2+ (~2시간)

### 15.5 자동 재학습 + Rollback

```python
# 재학습 필요 조건 (should_retrain)
if dir_acc < 0.48:     → weekly retrain  # 신호 정확도 저하
if sharpe_14d < 0.5:   → weekly retrain  # 최근 성과 저하
if mdd_14d < -0.10:    → monthly retrain # MDD 위험 수준

# 재학습 후 품질 검증 (auto-rollback)
if dir_acc_after < 0.48:
    rollback_model()  # 가장 최근 backup에서 복원
    # saved_models/backups/ 에 최근 5개 버전 보관
```

### 15.6 성과 모니터링 (`tracking/trade_log.py`)

SQLite 4개 테이블:
- `trades`: 거래 이력 (ticker, side, qty, price, sector, order_no)
- `daily_performance`: 일별 성과 (portfolio_value, daily_return, sharpe_30d, mdd_cumul)
- `model_signals`: 신호 이력 (sector, signal, weight, actual_return) — 방향 정확도 계산용
- `retrain_log`: 재학습 이력 (trigger, duration_sec, dir_acc, val_loss)

### 15.7 Streamlit 대시보드

실행: `streamlit run dashboard/app.py` → http://localhost:8501

- 포트폴리오 가치 / 누적 수익률 / 30일 Sharpe / MDD / 승률
- 수익 곡선 vs Equal-weight 벤치마크 (Plotly)
- 섹터 배분 파이 차트 (최신 신호 기준)
- 방향 예측 정확도 게이지 (기준: 50%)
- 일별 수익률 바 차트 (녹색/적색)
- 최근 거래 내역 테이블
- 재학습 이력 테이블

---

## 16. 신호 생성 파이프라인 수정사항

Phase 7에서 수정된 **3개의 핵심 버그** 및 **Alpha Blending 추가**:

### 16.1 Bug 1: RiskManager 상태 오염 (`backtest/engine.py`)

```python
# 수정 전: 두 번의 run() 호출이 같은 RiskManager 인스턴스 공유
# → baseline run이 model run의 peak_value를 물려받아 결과 오염

# 수정 후: 각 run() 시작 시 state 초기화
def run(self, ...):
    self.risk_mgr.peak_value = 0.0    # ← 추가
    self.risk_mgr.daily_pnl = 0.0     # ← 추가
    self.risk_mgr.is_risk_off = False  # ← 추가
    ...
```

### 16.2 Bug 2: sector_id 항상 0 (`main.py`, `inference_pipeline.py`)

```python
# 수정 전: 모든 섹터에 Energy(0) 임베딩 적용
sector_id = torch.tensor([0], dtype=torch.long)

# 수정 후: sectors.yaml 키 순서 = 훈련 시 id
with open("config/sectors.yaml") as f:
    sectors_cfg = yaml.safe_load(f)
sector_to_train_id = {
    name: idx for idx, name in enumerate(sectors_cfg["sectors"].keys())
}
# → energy=0, materials=1, industrials=2, ...
train_sector_id = sector_to_train_id.get(sector, 0)
sector_id = torch.tensor([train_sector_id], dtype=torch.long)
```

### 16.3 Bug 3: Ticker 혼합 행 (`main.py`, `inference_pipeline.py`)

```python
# 수정 전: sector_data (N_tickers × N_days 행)에서 tail(60)
# → 6일 × 10 ticker = 60행 (60일 시퀀스가 아님!)
sector_data = df[df["sector"] == sector].tail(seq_length)
features = torch.FloatTensor(sector_data[feature_cols].values)

# 수정 후: ticker별 독립적으로 60일 시퀀스 구성 → 평균
for ticker in tickers_in_sector:
    tkr_df = sector_df[sector_df["ticker"] == ticker].sort_values("date")
    window = tkr_df[feature_cols].values[-60:]  # 정확한 60일
    features = torch.FloatTensor(window).unsqueeze(0)
    prediction = ensemble.get_signals(features, sector_id)["prediction"]
    ticker_preds.append(prediction.item())
sector_signal = np.mean(ticker_preds)
```

### 16.4 Alpha Blending 추가 (`main.py`, `scheduler/daily_runner.py`)

```python
# 순수 모델 신호는 상위 섹터에 집중 → equal-weight 대비 열세
# Grid search: alpha=0.4 (40% model + 60% equal-weight) 최적

# top-K 정규화 (모델 신호 처리)
top_k = 5
top_indices = np.argsort(raw_weights)[-top_k:]
model_weights = np.zeros(n_sectors)
for idx in top_indices:
    if raw_weights[idx] > 0:
        model_weights[idx] = raw_weights[idx]
model_weights /= model_weights.sum()

# alpha blending
equal_weight = np.ones(n_sectors) / n_sectors
final_weights = 0.4 * model_weights + 0.6 * equal_weight
```

**효과**: Sharpe 1.51 (pure model) → **2.03** (alpha=0.4)

---

## 17. Phase 8: 추론 파이프라인 버그 수정 + 개별 종목 매매

**파일**: `pipeline/inference_pipeline.py`, `live/signal_to_order.py`, `scheduler/daily_runner.py`

### 17.1 체크포인트 Shape 자동 감지

훈련 시 사용된 하이퍼파라미터와 추론 시 기본값 불일치 문제를 체크포인트에서 직접 읽어 해결:

```python
# VAE n_features 자동 감지
ckpt = torch.load(path, map_location="cpu", weights_only=False)
flat_dim = ckpt["vae"]["encoder.network.0.weight"].shape[1]
n_features = flat_dim // seq_length    # 2040 // 60 = 34

# RL state_dim 자동 감지
rl_first_w = next(v for k, v in ckpt["rl_policy"].items() if "weight" in k)
rl_state_dim = rl_first_w.shape[1]    # PPO 첫 번째 Linear 입력 크기

# Transformer sector_embed_dim 명시적 전달
transformer = TemporalTransformer(
    ...,
    sector_embed_dim=tf_cfg.get("sector_embed_dim", 32),  # config에서 읽기
)
```

### 17.2 DataProcessor 추론 모드 설정

```python
# 훈련 시: min_history_days=500 (5년치 기준)
# 추론 시: seq_length=60 (마지막 60일만 필요)
self.processor = DataProcessor(
    min_history_days=self.config["data"]["sequence_length"]  # 60
)
```

### 17.3 Sector Column 처리

추론 시 새로 수집한 OHLCV 데이터에는 sector 컬럼이 없으므로 훈련 데이터에서 로드:

```python
if "sector" not in df.columns:
    processed_path = Path("data/processed/processed_data.parquet")
    if processed_path.exists():
        ticker_sector = (
            pd.read_parquet(processed_path, columns=["ticker", "sector"])
            .drop_duplicates("ticker")
            .set_index("ticker")["sector"]
        )
        df["sector"] = df["ticker"].map(ticker_sector).fillna("unknown")
```

### 17.4 개별 종목 목표 포지션 계산 (`live/signal_to_order.py`)

```python
def _compute_target_positions(self, weights, portfolio_value, sector_top_tickers):
    for i, sector in enumerate(self.sectors):
        if weights[i] < 1e-4:
            continue
        sector_amount = portfolio_value * weights[i]
        top_tickers = sector_top_tickers.get(sector, [])
        if not top_tickers:
            continue   # 해당 섹터 양수 신호 종목 없으면 패스

        # 종목 수에 따라 균등 분배
        per_ticker_amount = sector_amount / len(top_tickers)
        if per_ticker_amount < self.min_order_amount:
            top_tickers = top_tickers[:1]  # 소액이면 1종목 집중
            per_ticker_amount = sector_amount

        for tkr_info in top_tickers:
            ticker = tkr_info["ticker"]
            current_price = self.api.get_domestic_price(ticker)["price"]
            qty = int(per_ticker_amount / current_price)
            if qty >= 1:
                targets[ticker] = {
                    "target_qty": qty,
                    "target_amount": per_ticker_amount,
                    ...
                }
```

### 17.5 하락 신호 즉시 매도 (`execute_sell_check`)

매일 06:30 신호 생성 직후 실행:

```python
def execute_sell_check(self, sector_top_tickers):
    # 오늘 양수 신호 있는 ticker 집합
    positive_tickers = {
        t["ticker"]
        for tickers in sector_top_tickers.values()
        for t in tickers
    }
    # 보유 중인데 양수 신호 없는 종목 → 즉시 매도
    for ticker, pos in self._get_current_positions().items():
        if pos["qty"] > 0 and ticker not in positive_tickers:
            self._submit_order({"side": "sell", "qty": pos["qty"], ...})
```

### 17.6 일간 리밸런싱 흐름 요약

```
매일 06:30:
  step_signal() → (final_weights, sector_top_tickers)
  step_sell_check(sector_top_tickers)
    → 하락 신호 종목 전량 매도 → 현금 확보

매일 08:50:
  step_order(final_weights, sector_top_tickers)
    ↓ execute_rebalance()
    1. _validate_weights()         가중치 검증 + cash_buffer(5%) 적용
    2. _get_current_positions()    현재 보유 종목/수량 조회
    3. _compute_target_positions() 개별 종목 목표 수량 계산
    4. _compute_orders()           현재 vs 목표 비교 → 주문 목록
       (amount_diff < 3% → skip / 매도 먼저 → 매수)
    5. _is_daily_loss_exceeded()   일간 -3% 초과 시 중단
    6. _submit_order()             KIS API 실제 주문
```

---

## 18. Phase 11: KR/US 분리 스케줄 확정 + 해외 시세 yfinance 폴백

**파일**: `config/live_config.yaml`, `live/signal_to_order.py`, `scheduler/daily_runner.py`

### 18.1 배경 — 오늘 발견된 버그들 (2026-02-23)

서버 로그 분석으로 다음 문제가 발견되어 수정:

| 증상 | 원인 | 파일 |
|------|------|------|
| `OverflowError: cannot convert float infinity to integer` | 현재가 0 반환 → `int(amount/0)` | `signal_to_order.py:558` |
| KODEX ETF 조회 (개별 종목이어야 함) | `execution_market: "kospi"` — Phase 8 후 미변경 | `live_config.yaml` |
| 해외 종목 현재가 404 (전 종목) | KIS sandbox 해외 시세 API 미지원 | `broker/kis_api.py` 한계 |
| `get_overseas_balance()` 500 | KIS sandbox 해외 잔고 API 미지원 | `broker/kis_api.py` 한계 |

### 18.2 OverflowError 수정 (`live/signal_to_order.py`)

```python
# 수정 전: current_price=0 → ZeroDivisionError 또는 float('inf') → int() OverflowError
qty = int(per_ticker_amount / current_price)

# 수정 후: 0 이하 가격 종목 스킵
if current_price is None or current_price <= 0:
    logger.warning(f"{ticker} 현재가 조회 실패 ({current_price}) → 스킵")
    continue
qty = int(per_ticker_amount / current_price)
```

### 18.3 yfinance 폴백 — KIS sandbox 해외 API 미지원 해결

KIS sandbox 서버(`openapivts.koreainvestment.com:9443`)는 **국내 주식만 지원**. 해외주식 엔드포인트는 sandbox에 없어 404 반환. 3곳에 yfinance 폴백 적용:

**`_compute_target_positions` (해외 종목 현재가)**:
```python
try:
    price_info    = api.get_overseas_price(ticker, exchange)
    current_price = float(price_info["price"])
except Exception:
    import yfinance as yf
    current_price = yf.Ticker(ticker).fast_info.last_price
    logger.info(f"{ticker} KIS 시세 불가 → yfinance 폴백: ${current_price:.2f}")
```

**`_get_limit_price` (지정가 계산)**:
```python
if market == "domestic":
    p = float(api.get_domestic_price(ticker)["price"])
else:
    import yfinance as yf
    p = yf.Ticker(ticker).fast_info.last_price
    logger.info(f"{ticker} 호가 yfinance 폴백: ${p:.2f}")
bid = ask = p  # yfinance는 호가가 없으므로 현재가로 대체
```

**`get_overseas_balance()` 오류 처리**:
```python
if self.execution_market in ("nasdaq", "split"):
    try:
        b = self.api_overseas.get_overseas_balance()
        total += b.get("total_eval", 0) + b.get("cash", 0)
    except Exception as e:
        logger.warning(f"해외 잔고 조회 실패 (sandbox 미지원): {e}")
        # 잔고 0으로 계속 진행 (국내 잔고는 정상 조회됨)
```

**검증**: `yf.Ticker("AAPL").fast_info.last_price` → $264.58 (정상) ✅

### 18.4 KR/US 분리 스케줄 구현 (확정 설계)

**핵심 원칙**: daily 신호 모델 → 신호는 하루 1회 생성, Wave는 동일 신호의 분산 집행.
- Wave 간 재추론은 의미 없음 (intraday 데이터 없음, 모델이 daily OHLCV 기반)

#### market_filter 파라미터 (`live/signal_to_order.py`)

```python
def execute_rebalance(self, sector_weights, sector_top_tickers=None, market_filter=None):
    ...
    target_positions = self._compute_target_positions(...)

    # market_filter로 국내/해외 주문 분리
    if market_filter:
        target_positions = {
            t: v for t, v in target_positions.items()
            if v.get("market") == market_filter
        }
```

#### 데몬 함수 (`scheduler/daily_runner.py`)

```python
def _daemon_us_signal(self):
    """NASDAQ 종가 직후 (06:10) — 신호 생성 + 저장 + 해외 하락 매도."""
    weights, top_tickers = self.step_signal(use_cache=False)  # 캐시 무시, 재추론
    self._us_weights     = weights
    self._us_top_tickers = top_tickers
    self.step_sell_check(top_tickers)   # 해외 하락 신호 즉시 매도
    # → 23:40/02:00/04:30 Wave에서 _us_weights/_us_top_tickers 재사용

def _daemon_us_order(self, twap_wave: int):
    """해외 종목 TWAP 집행 (06:10 신호 그대로 사용, 재추론 없음)."""
    weights     = getattr(self, "_us_weights",     self._last_weights)
    top_tickers = getattr(self, "_us_top_tickers", self._last_top_tickers)
    self.step_order(weights, top_tickers, twap_wave=twap_wave, market_filter="overseas")

def _daemon_kr_order(self, twap_wave: int):
    """국내 종목 TWAP 집행 (06:30 신호 그대로 사용)."""
    weights     = getattr(self, "_last_weights",     np.ones(11) / 11)
    top_tickers = getattr(self, "_last_top_tickers", {})
    self.step_order(weights, top_tickers, twap_wave=twap_wave, market_filter="domestic")
```

#### run_daemon() 스케줄 등록

```python
# 공통 수집
schedule.every().day.at(sc["data_collect_time"]).do(self.step_collect)   # 06:00

# US: 신호 생성 1회 → 장 중 3-파 집행
schedule.every().day.at(sc["us_signal_time"]).do(self._daemon_us_signal) # 06:10
schedule.every().day.at(us["wave1"]).do(lambda: self._daemon_us_order(1)) # 23:40
schedule.every().day.at(us["wave2"]).do(lambda: self._daemon_us_order(2)) # 02:00
schedule.every().day.at(us["wave3"]).do(lambda: self._daemon_us_order(3)) # 04:30

# KR: 신호 생성 1회 → 장 중 3-파 집행
schedule.every().day.at(sc["signal_gen_time"]).do(self._daemon_signal_and_sell) # 06:30
schedule.every().day.at(kr["wave1"]).do(lambda: self._daemon_kr_order(1)) # 09:10
schedule.every().day.at(kr["wave2"]).do(lambda: self._daemon_kr_order(2)) # 11:00
schedule.every().day.at(kr["wave3"]).do(lambda: self._daemon_kr_order(3)) # 13:30
schedule.every().day.at(sc["eod_record_time"]).do(self.step_eod)          # 16:00
```

### 18.5 live_config.yaml 확정 구조

```yaml
trading:
  execution_market: "split"    # "kospi" → "split" (개별 종목 활성화)

schedule:
  data_collect_time: "06:00"   # KR + US 전일 종가 수집
  us_signal_time:    "06:10"   # NASDAQ 종가 직후 신호 생성 (1회)
  us_order_waves:
    wave1: "23:40"    # 40% — US 장 중 분산 집행
    wave2: "02:00"    # 35%
    wave3: "04:30"    # 25%
  signal_gen_time:   "06:30"   # KR 신호 생성
  kr_order_waves:
    wave1: "09:10"    # 40%
    wave2: "11:00"    # 35%
    wave3: "13:30"    # 25%
  eod_record_time:   "16:00"
```

---

## 부록: 알려진 문제와 해결 방향

### A. GAN 불안정 (부분 해결)

**현상**: Epoch 1 best (W-dist≈6.66), 이후 점진적 상승
**현재 상태**: Spectral Norm + n_critic=10 + 구조 축소로 발산 패턴 개선 (13→6.66)
**잔존 문제**: 여전히 epoch마다 W-dist 증가 추세. mode collapse 근본 미해결.
**다음 시도**: Progressive GAN 또는 Diffusion 기반 시장 시뮬레이터

### B. RL 포트폴리오 폭발 (해결됨)

**현상**: portfolio_value 66조원까지 폭발 (해결 전)
**해결**: simplex 투영 (long-only, 합=1) + reward clipping ±10 → best_reward 1.75→4.75

### C. 백테스트 신호 생성 개선 이력

**이전**: softmax 정규화 → 항상 100% long (하락장 회피 불가)
**현재**: top-K (k=5) + alpha blending (0.4/0.6) → MDD 개선됨

### D. 섹터 정보 일관성 (해결됨)

**이전**: 백테스트에서 alphabetical sort, 추론에서 YAML 키 순서 → 섹터 임베딩 오인식
**현재**: 모든 경로에서 `sectors.yaml` 키 순서를 training-time id로 사용

---

---

## 19. Phase 12: KOSPI 학습 포함 + 데몬 안정성 수정

**파일**: `pipeline/train_pipeline.py`, `config/settings_fast.yaml`, `scheduler/daily_runner.py`, `broker/kis_api.py`

### 19.1 배경 — 서버 운영 중 발견된 문제들 (2026-02-25)

서버 로그 분석:
- **매일 06:01** 데몬이 종료됨 → systemd가 60초 후 재시작 반복
- **재시작 후 KR/US Wave 모두 0건 주문** — 재시작으로 메모리 신호 초기화
- **Wave 3(04:30) 크래시** → 토큰 rate limit 403

KOSPI 개별 종목이 신호에 전혀 안 나타남 → 학습 데이터 자체에 KOSPI 없음 확인.

### 19.2 데몬 크래시 버그 수정 3개

#### Bug 1: step_collect 크래시 (`scheduler/daily_runner.py`)

```python
# 수정 전: 예외 raise → 데몬 종료 (systemd 재시작)
def step_collect(self):
    try:
        ...
    except Exception as e:
        logger.error(f"데이터 수집 실패: {e}")
        raise  # ← 데몬 크래시

# 수정 후: 경고만 남기고 계속 (step_signal이 자체 수집 수행)
    except Exception as e:
        logger.error(f"데이터 수집 실패: {e}")
        return  # 데몬 유지
```

**원인**: 증분 수집(10일)된 데이터가 `DataProcessor(min_history_days=500)` 필터에 전부 제거됨 → "No objects to concatenate". step_signal은 어차피 1년치 재수집하므로 step_collect 실패는 무해.

#### Bug 2: 재시작 후 신호 메모리 소실 (`scheduler/daily_runner.py`)

```python
# 수정 전: 메모리에서만 신호 로드 → 재시작 후 None
def _daemon_kr_order(self, twap_wave: int = 1):
    weights     = self._last_weights      # None after restart
    top_tickers = self._last_top_tickers  # None after restart
    self.step_order(weights, top_tickers, ...)

# 수정 후: None이면 당일 캐시 파일에서 로드
def _daemon_kr_order(self, twap_wave: int = 1):
    weights     = getattr(self, "_last_weights",     None)
    top_tickers = getattr(self, "_last_top_tickers", None)
    if weights is None or not top_tickers:
        logger.info("[KR] 메모리 신호 없음 → 캐시에서 로드")
        weights, top_tickers = self.step_signal(use_cache=True)
    self.step_order(weights, top_tickers, twap_wave=twap_wave, market_filter="domestic")

# _daemon_us_order도 동일 패턴 적용
```

**캐시 파일**: `tracking/signal_cache_{YYYYMMDD}.json` — 당일 신호 생성 시 자동 저장, 재시작 후 Wave 실행 시 이 파일에서 복원.

#### Bug 3: 토큰 rate limit 크래시 (`broker/kis_api.py`)

```python
# 원인: 국내(29443) + 해외(29443) API 인스턴스가 동시에 토큰 갱신 시도
# KIS API 정책: 분당 1회 → 두 번째 요청 403 Forbidden

# 수정 전: 403 → raise → 데몬 크래시
resp = self._session.post(url, json=body, timeout=10, verify=False)
resp.raise_for_status()

# 수정 후: 3회 재시도 (5초 간격)
import time as _time
for attempt in range(3):
    resp = self._session.post(url, json=body, timeout=10, verify=False)
    if resp.status_code == 403 and attempt < 2:
        logger.warning(f"토큰 발급 403 (시도 {attempt+1}/3) → 5초 후 재시도")
        _time.sleep(5)
        continue
    resp.raise_for_status()
    break
```

**추가 수정**: `step_order` 내 주문 실패 시 `raise` → `return []` (주문 실패가 데몬 전체를 종료시키지 않도록).

### 19.3 KOSPI 섹터 분류 버그 수정 2개 (`pipeline/train_pipeline.py`)

#### Bug 1: 회사명 대신 ticker code로 키워드 매칭

```python
# 수정 전: OHLCV ticker ("005930")를 name으로 사용 → 키워드 매칭 0%
ticker_info["name"] = ticker_info["ticker"]   # "005930" — 의미 없음

# 수정 후: ticker_info.parquet에서 실제 회사명 로드
ticker_info_path = raw_dir / "ticker_info.parquet"
if ticker_info_path.exists():
    name_df = pd.read_parquet(ticker_info_path)[["yf_ticker", "name"]]
    ticker_info = ticker_info.merge(
        name_df, left_on="ticker", right_on="yf_ticker", how="left"
    )
    ticker_info["name"] = ticker_info["name"].fillna(ticker_info["ticker"])
```

#### Bug 2: ticker 형식 불일치로 merge 실패

```python
# 문제: OHLCV는 "095570.KS" (yfinance 형식)
#       ticker_info.parquet는 "095570" (KRX 코드)
# → left_on="ticker" merge 시 .KS 접미사 차이로 모두 실패

# 수정: yf_ticker 컬럼(= "095570.KS" 형식) 으로 right_on 지정
name_df = pd.read_parquet(ticker_info_path)[["yf_ticker", "name"]]
ticker_info.merge(name_df, left_on="ticker", right_on="yf_ticker", how="left")
# ticker_info["ticker"] == "095570.KS"
# name_df["yf_ticker"] == "095570.KS"  → 매칭 성공
```

**결과 비교**:

| 항목 | 수정 전 | 수정 후 |
|------|--------|--------|
| KOSPI 섹터 분류 수 | 0 / 941 | **203 / 941** (21.6%) |
| 학습 종목 수 | ~200 (NASDAQ 위주) | **1,466** (KOSPI + NASDAQ) |
| max_tickers | 200 | **400** (KOSPI 200 + NASDAQ 200) |
| 학습 배치 수 | ~200 | **~1,210** (약 6배) |

### 19.4 재학습 및 배포

**재학습 설정**: `config/settings_fast.yaml`, Phase 2 전체 (~3시간)

```bash
# 로컬 PC (RTX 4060 Ti 8GB)
$PYTHON main.py train --start-phase 2 --config config/settings_fast.yaml
# → 2026-02-25 01:06 완료, ensemble.pt (26MB)
```

**서버 배포** (`77.42.78.9`, 07:42 재시작 완료):

```bash
# 자동 배포 스크립트가 타임아웃 (log mtime > ensemble.pt mtime 조건 실패)
# → 수동 scp 배포
scp saved_models/ensemble.pt root@77.42.78.9:/opt/quant/saved_models/
scp data/processed/processed_data.parquet root@77.42.78.9:/opt/quant/data/processed/
scp pipeline/inference_pipeline.py root@77.42.78.9:/opt/quant/pipeline/
scp pipeline/train_pipeline.py root@77.42.78.9:/opt/quant/pipeline/
scp scheduler/daily_runner.py root@77.42.78.9:/opt/quant/scheduler/
scp broker/kis_api.py root@77.42.78.9:/opt/quant/broker/
ssh root@77.42.78.9 "systemctl start quant-trading"
```

**자동 배포 스크립트 한계** (`deploy_on_complete.sh`):
- `find "$ENSEMBLE" -newer "$LOG"` 조건: 재학습 로그 파일이 tqdm에 의해 지속 갱신 → 로그 mtime이 항상 ensemble.pt보다 최신 → 조건 항상 false → 타임아웃
- 향후 개선: log 파일 대신 sentinel 파일(완료 플래그) 방식으로 변경 필요

---

## 20. Phase 13: 해외 Paper Trading + KOSPI Parquet 구축 (2026-02-26)

**파일**: `live/signal_to_order.py`, `pipeline/inference_pipeline.py`, `scheduler/daily_runner.py`, `config/live_config.yaml`, `deploy_on_complete.sh`

### 20.1 배경

서버 운영 1일차 로그 분석 결과:
- NASDAQ 주문 33건 모두 실패 → 해외 sandbox USD 잔고 $0
- KOSPI domestic Wave "리밸런싱 불필요" → KOSPI 종목이 overseas로 분류됨
- EOD 기록 실패 → NOT NULL constraint (daily_return = NaN)
- deploy_on_complete.sh 5시간 타임아웃 반복

### 20.2 버그 수정 6개

#### Bug 1: deploy_on_complete.sh 타임아웃 (`deploy_on_complete.sh`)

```bash
# 수정 전: find -newer $LOG → log가 ensemble.pt보다 21ms 늦게 기록됨
# → ensemble이 항상 "더 오래된" 파일 → 조건 항상 false
if find "$ENSEMBLE" -newer "$LOG" | grep -q .; then ...

# 수정 후: 스크립트 시작 시각(epoch)과 ensemble mtime 비교
START_TIME=$(date +%s)
# ...루프 내...
ENSEMBLE_MTIME=$(stat -c %Y "$ENSEMBLE" 2>/dev/null)
if [ "$ENSEMBLE_MTIME" -ge "$START_TIME" ]; then
    # deploy
fi
```

#### Bug 2: 해외 sandbox $0 USD → Paper Trading 모드 (`live/signal_to_order.py`, `config/live_config.yaml`)

KIS 해외 모의투자 계좌는 USD 예수금이 $0이며 API로 입금 불가.

```yaml
# live_config.yaml
broker:
  paper_trading: true   # 해외 주문을 가상 체결로 처리
```

```python
# signal_to_order.py
self.paper_trading = self.cfg["broker"].get("paper_trading", False)
if self.paper_trading:
    self.api_overseas = None  # 토큰 403 방지 — overseas KISApi init 생략

def _execute_paper_trade(self, order: dict) -> dict:
    """yfinance 시세 기준 가상 체결. DB에 mode='paper' 기록."""
    price = float(yf.Ticker(order["ticker"]).fast_info.last_price)
    return {
        "ticker": order["ticker"], "side": order["side"],
        "qty": order["qty"],       "price": price,
        "order_no": f"PAPER_{ticker}_{int(time.time())}",
        "mode": "paper",
    }
```

**결과**: NASDAQ 33/33 주문 가상 체결, DB 정상 기록.

#### Bug 3: KRW/USD 수량 계산 오류 (`live/signal_to_order.py`)

```python
# 수정 전: amount(KRW) / price(USD) → 단위 불일치
qty = int(per_ticker_amount / current_price)  # 41,862주 오계산

# 수정 후: FX 환율 변환 (yfinance USDKRW=X)
def _get_fx_rate(self) -> float:
    rate = float(yf.Ticker("USDKRW=X").fast_info.last_price)  # ex: 1425.2
    self._fx_rate_cache = rate  # 세션 캐시
    return rate

# overseas paper trading 시 KRW → USD 변환
if self.paper_trading and market == "overseas":
    amount_for_qty = per_ticker_amount / self._get_fx_rate()
qty = int(amount_for_qty / current_price)  # 29주 정상 계산
```

#### Bug 4: KOSPI 종목 overseas 분류 오류 (`pipeline/inference_pipeline.py`)

```python
# 수정 전: "095570.KS".isdigit() == False → overseas
"market": "domestic" if t.isdigit() else "overseas"

# 수정 후: .KS/.KQ 접미사 확인
"market": "domestic" if (t.isdigit() or t.endswith(".KS") or t.endswith(".KQ")) else "overseas"
```

#### Bug 5: step_eod NOT NULL constraint (`scheduler/daily_runner.py`)

```python
# 수정 전: split 모드 → else → overseas balance($0) → ZeroDivision/NaN
if self.execution_market == "kospi":
    balance = api.get_domestic_balance()
else:
    balance = api.get_overseas_balance()  # split 모드도 여기로 → $0

# 수정 후: split/paper_trading → domestic balance + NaN 방어
paper_trading = self.cfg["broker"].get("paper_trading", False)
if self.execution_market in ("kospi", "split") or paper_trading:
    balance = api.get_domestic_balance()
else:
    balance = api.get_overseas_balance()

if prev_value > 0:
    daily_return = (portfolio_value - prev_value) / prev_value
else:
    daily_return = 0.0
if math.isnan(daily_return) or math.isinf(daily_return):
    daily_return = 0.0
```

#### Bug 6: step_collect KOSPI 데이터 필터 (`scheduler/daily_runner.py`)

```python
# 수정 전: DataProcessor(500) → 10일 증분 데이터 전부 제거
processor = DataProcessor()  # min_history_days=500 기본값

# 수정 후: 증분 수집은 히스토리 체크 불필요
processor = DataProcessor(min_history_days=1)  # 모든 행 유지
```

### 20.3 KOSPI Parquet 구축 (서버 직접)

서버의 `processed_data.parquet`가 NASDAQ 전용(400 tickers)으로, KR 신호 생성 시 KOSPI 종목이 없어 모두 "unknown" 처리됨.

서버에 이미 존재하는 raw 데이터를 활용해 직접 구축:

```python
# 서버에서 1회 실행 (Python script)
# 1. 섹터 분류
from data.sector_classifier import SectorClassifier
ticker_info = pd.read_parquet('data/raw/ticker_info.parquet')  # 950 KOSPI tickers
classified = SectorClassifier().classify_kospi(ticker_info[ticker_info['market']=='KOSPI'])
# → 203/950 종목 섹터 분류 성공 (financials 44, healthcare 37, industrials 26...)

# 2. OHLCV 처리
kdf = pd.read_parquet('data/raw/kospi_ohlcv.parquet')  # 1,123,789행, 2021~
kdf = kdf[kdf['ticker'].isin(sector_map.keys())]       # 203 classified tickers
new_df = DataProcessor(min_history_days=60).process(kdf)
new_df = FeatureEngineer().compute_all(new_df)         # 220,946행, 34 features

# 3. 기존 parquet 병합
combined = pd.concat([old_df, new_df]).drop_duplicates(subset=['date','ticker'])
# 477,802 + 220,946 → 698,748행, 602 tickers
```

**결과**:

| 항목 | 변경 전 | 변경 후 |
|------|--------|--------|
| Total tickers | 400 (NASDAQ only) | **602** (400 NASDAQ + 202 KOSPI) |
| Total rows | 477,802 | **698,748** |
| 파일 크기 | 158MB | **221MB** |
| KS sector coverage | 0% | **11개 섹터 전부** |

### 20.4 inference_pipeline.py — pykrx 재호출 방지

```python
# 수정 전: generate_signals()마다 pykrx 재수집 (DNS 실패 위험)
collector = MarketDataCollector(history_years=1)
market_data_dict = collector.collect_all(save=False)

# 수정 후: step_collect(06:00)이 저장한 parquet 재사용
processed_path = Path("data/processed/processed_data.parquet")
if processed_path.exists():
    market_data = pd.read_parquet(processed_path)
    logger.info(f"Loaded market data from parquet: {len(market_data):,} rows")
else:
    logger.warning("parquet 없음 → pykrx 재수집 (느림)")
    # ... fallback
```

### 20.5 배포 현황 (2026-02-26 23:16)

```bash
# 5개 파일 SCP 배포
scp config/live_config.yaml root@77.42.78.9:/opt/quant/config/
scp live/signal_to_order.py root@77.42.78.9:/opt/quant/live/
scp pipeline/inference_pipeline.py root@77.42.78.9:/opt/quant/pipeline/
scp scheduler/daily_runner.py root@77.42.78.9:/opt/quant/scheduler/
# processed_data.parquet는 서버에서 직접 구축 (scp 불필요)
ssh root@77.42.78.9 "systemctl restart quant-trading"
```

---

---

## 21. Phase 14: execution_market 수정 + KOSPI 수집 KRX API 수정

**날짜**: 2026-03-04
**수정 파일**: `config/live_config.yaml`, `data/collector.py`

### 21.1 execution_market 버그 — KR domestic Wave 0건

#### 원인 분석

`live_config.yaml`의 `execution_market: "nasdaq"` 설정으로 인해, `_compute_target_positions()`가 `NASDAQ_ETF_MAP`(XLE, XLB 등)만 사용하고 모든 target에 `market="overseas"`를 붙인다.

KR domestic Wave 실행 시 `execute_rebalance(market_filter="domestic")` 호출 → 모든 target이 `market="overseas"` → 필터 후 빈 dict → `_compute_orders()` 빈 list → `리밸런싱 불필요`.

```
실행 흐름 (수정 전):
  execution_market="nasdaq"
  → _compute_target_positions(): XLE(overseas), XLB(overseas), ... (11개 ETF)
  → market_filter="domestic" 적용: {} (전부 제거)
  → _compute_orders({}, {}): []
  → "리밸런싱 불필요" ← 항상 0건
```

모델 신호는 `322000.KS(market=domestic)`, `INTU(market=overseas)` 혼합으로 생성되는데, `"nasdaq"` 모드는 이를 무시하고 ETF만 사용한다.

#### 수정

```yaml
# config/live_config.yaml
execution_market: "split"   # "nasdaq" → "split"
```

`"split"` 모드에서 `_compute_target_positions()`:
```python
else:  # split: 모델 top-ticker 그대로 사용
    top_tickers = sector_top_tickers.get(sector, [])
    ticker_list = top_tickers   # KS → domestic, US → overseas
```

#### 결과

| market_filter | 수정 전 (nasdaq) | 수정 후 (split) |
|---|---|---|
| `"domestic"` (KR Wave) | 0건 — ETF 전부 overseas | KOSPI 개별 종목 집행 |
| `"overseas"` (US Wave) | XLE/XLB 등 ETF | INTU/EPAM 등 개별 US 주식 |

### 21.2 KOSPI 수집 KRX API 버그 — 06:00 항상 0건

#### 원인 분석

`pykrx.get_market_ticker_list(date)` 내부 구조:

```python
def get_market_ticker_list(date, market="KOSPI"):
    s = krx.get_market_ticker_and_name(date, market)   # OHLCV 기반 조회
    return s.index.to_list()
```

이 함수는 해당 날짜의 **OHLCV 데이터**를 조회하여 ticker 목록을 추출한다. KRX는 당일 장 종료(15:30 KST) 이후에야 당일 OHLCV를 게시하므로, **06:00 KST 조회 시 당일 데이터 없음 → 0건**.

반면 `상장종목검색().fetch("STK")`은 KRX **상장종목 검색 API** (`dbms/comm/finder/finder_stkisu`)를 사용하여 OHLCV와 무관하게 현재 상장 종목 목록을 반환한다.

```python
# 기존 — OHLCV 기반, 06:00 0건
krx.get_market_ticker_list("20260304", market="KOSPI")  # → []

# 수정 — 상장종목 검색 API, 시간 무관
from pykrx.website.krx.market.core import 상장종목검색
상장종목검색().fetch("STK")  # → 950행 (항상)
```

#### 수정 (`data/collector.py`)

```python
def get_kospi_tickers(self) -> pd.DataFrame:
    from pykrx.website.krx.market.core import 상장종목검색
    raw = 상장종목검색().fetch("STK")   # STK = 유가증권(KOSPI)
    records = [
        {"ticker": row["short_code"], "name": row["codeName"],
         "yf_ticker": f"{row['short_code']}.KS", "market": "KOSPI"}
        for _, row in raw.iterrows()
    ]
    return pd.DataFrame(records)   # 950건
```

#### 결과

| | 수정 전 | 수정 후 |
|---|---|---|
| 06:00 수집 | `KOSPI 수집 실패: KRX 0건 반환` | KOSPI tickers: **950건** 정상 |
| 주말/공휴일 | 0건 (OHLCV 없음) | 950건 (상장 목록 항상 유효) |

### 21.3 배포

```bash
scp config/live_config.yaml root@77.42.78.9:/opt/quant/config/
scp data/collector.py root@77.42.78.9:/opt/quant/data/
ssh root@77.42.78.9 "systemctl restart quant-trading"
```

테스트 결과 (2026-03-04 12:41): KR 15건 + US 18건 = **33/33 체결**

---

*이 문서는 실제 소스 코드를 기반으로 작성되었습니다.*
*마지막 업데이트: 2026-03-04 (Phase 14 — execution_market split + KOSPI KRX API 수정)*
