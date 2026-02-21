# Alpha Signal Discovery Engine - 기술 상세 문서

**버전**: 2026-02-21
**현재 최고 성과**: Sharpe 0.62, MDD -25.89% (Phase 2 기준)

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

| 시장 | 데이터 소스 | 종목 수 |
|------|-----------|--------|
| KOSPI | pykrx | ~100 종목 |
| NASDAQ/S&P500 | yfinance | ~100 종목 |
| **합계** | | **~200 종목** |

**데이터 규모**: 238,202행 × 34 피처 (5년치, 2020-2024)
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

### Transformer

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `d_model` | 128 | 모델 차원 |
| `n_heads` | 8 | 멀티헤드 어텐션 수 |
| `n_encoder_layers` | 4 | Transformer 레이어 수 |
| `d_ff` | 512 | FFN 은닉층 크기 |
| `prediction_horizon` | 5 | 예측 목표 (5일 후) |
| `learning_rate` | 5e-5 | 학습률 (워밍업 포함) |
| `epochs` | 25 | 훈련 에포크 |

### GAN

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `noise_dim` | 64 | Generator 입력 노이즈 차원 |
| `generator_hidden_dims` | [128, 256, 128] | Generator 구조 |
| `learning_rate_g/d` | 5e-5 | 학습률 (1e-4에서 감소) |
| `n_critic` | 5 | Discriminator 업데이트 횟수 |
| `gradient_penalty_weight` | 10.0 | GP 강도 λ |
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

### 추론/백테스트 시

```
[Test Period Data: 180 days, 11 sectors]
    ↓
[ensemble.pt 로드]
    ↓
For each test day t:
    For each sector s:
        window = data[-60:t, sector_s]  (60일 롤링 창)

        ① vae.encode(window) → latent_indicators (32-dim)
        ② transformer(window, sector_id) → signal_embedding (128-dim)
        ③ fusion([latent, signal_embed]) → prediction (scalar)

        model_signals[t, s] = prediction
    ↓
[신호 정규화]
    shifted = row - row.min() + 1e-8
    weights = shifted / shifted.sum()
    → 항상 100% long (개선 예정: top-K 선택)
    ↓
[일별 리스크 차단기 (사전 필터)]
    portfolio_value 시뮬레이션 → 차단기 적용
    ↓
[BacktestEngine.run()]
    주간 리밸런싱
    포트폴리오 최적화 → 리스크 관리 → 거래 비용 차감
    ↓
[백테스트 결과]
    Sharpe: 0.62 | MDD: -25.89% | Return: +29.9%
    → results/equity_curve.png
    → results/backtest_report.html
```

---

## 부록: 알려진 문제와 해결 방향

### A. GAN 불안정 (W-dist 발산)

**현상**: Epoch 1 (W-dist≈13) 이후 Epoch 2부터 발산 (40+)
**원인**: Generator의 mode collapse — 실제 데이터 분포의 특정 모드만 생성
**해결 방향**:
1. GAN 조기 종료 (W-dist가 best+5 초과 시 중단)
2. n_critic 증가 (5 → 10)
3. Generator 구조 축소 ([128,256,128] → [64,128,64])
4. Spectral Normalization 추가

### B. RL 포트폴리오 폭발

**현상**: 내부 평가 MDD=102%, portfolio_value 66조원
**원인**: action이 simplex 제약 없음 → 레버리지 무제한
**해결 방향**:
```python
action = np.clip(action, 0, None)
if action.sum() > 0:
    action = action / action.sum()  # simplex 투영
```

### C. 백테스트 항상 100% Long

**현상**: 하락 예상 섹터도 long 포지션
**원인**: `row - row.min() + 1e-8` 처리로 모든 값 양수화
**해결 방향**: 상위 3개 섹터만 투자 (음수 신호는 neutral 처리)

---

*이 문서는 실제 소스 코드를 기반으로 작성되었습니다.*
*마지막 업데이트: 2026-02-21*
