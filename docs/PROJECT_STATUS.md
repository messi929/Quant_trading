# Alpha Signal Discovery Engine - 프로젝트 현황

**최종 업데이트**: 2026-02-21 (Phase 4: GAN 안정화)
**현재 최고 성과**: Sharpe 1.94, MDD -1.26% (Phase 3~4 유지)

---

## 프로젝트 개요

KOSPI/NASDAQ 시장 데이터에서 딥러닝으로 **새로운 수학적 지표(Alpha Signal)**를 자동 발견하고,
이를 기반으로 GICS 11개 섹터별 포트폴리오를 운용하는 퀀트 트레이딩 시스템.

기존 기술적 지표(RSI, MACD 등)에 의존하지 않고, 모델이 원시 데이터에서 직접 학습한다.

---

## 아키텍처

```
[Yahoo Finance / KRX] → [Data Pipeline] → [Feature Engineering (34 features)]
                                                    │
                         ┌──────────────────────────┼──────────────────────┐
                         ▼                           ▼                     ▼
                  [VAE (32-dim)]           [Transformer]         [Cond. WGAN-GP]
                 잠재 지표 발견            시간 패턴 학습          시장 시뮬레이션
                         └──────────────────────────┼──────────────────────┘
                                                    ▼
                                        [Attention Fusion Layer]
                                                    ▼
                                           [PPO RL Agent]
                                          섹터별 매매 결정
                                                    │
                         ┌──────────────────────────┼──────────────────────┐
                         ▼                           ▼                     ▼
                  [Signal Gen]             [Portfolio Opt]         [Risk Mgmt]
                                                    │
                                           [Backtest Engine]
                                        (슬리피지/수수료 포함)
```

---

## 개발 이력

### Phase 0: 초기 구현 (2026-02-11)
- 47개 파일, 5,610줄 Python 코드 구현 완료
- 6단계 파이프라인: Data → VAE → Transformer → GAN → RL → Ensemble

### Phase 0.5: 초기 실행 및 버그 수정 (2026-02-19 ~ 2026-02-20)
- Miniconda + CUDA 12.x 환경 설정 완료
- GPU: RTX 4060 Ti (8.6GB VRAM) 정상 인식
- 데이터 수집: **238,202행, 200 종목, 34 피처** (5년치)
- 초기 전체 학습 완료 (VAE 30ep, Transformer 20ep, GAN best_epoch1, RL 97updates)
- 버그 수정:
  - GAN 체크포인트 포맷 불일치: `model.load_state_dict()` → `model.generator.load_state_dict()` + `model.discriminator.load_state_dict()`
  - `python` PATH 문제: `/c/Users/wogus/miniconda3/envs/quant/python.exe` 사용

### Phase 1: 리스크/보상 함수 개선 (2026-02-20)
**변경 파일**: `config/model_config_fast.yaml`, `models/rl/environment.py`, `strategy/risk.py`, `pipeline/train_pipeline.py`

| 변경 항목 | 이전 | 이후 | 이유 |
|----------|------|------|------|
| `rl.risk_penalty` | 0.1 | 1.0 | 드로다운 억제력 강화 |
| `rl.drawdown_penalty` | 0.5 | 5.0 | 비선형 4단계 페널티 |
| `rl.window_size` | 20 | 60 | 롤링 Sharpe 윈도우 |
| `rl.vol_target` | - | 0.15 | 연간 변동성 목표 추가 |
| `vae.kl_weight` | 0.001 | 0.01 | 잠재 지표 품질 향상 |
| `vae.kl_annealing_epochs` | 10 | 20 | 점진적 KL annealing |
| 리스크 차단기 | 단일 임계값(15%) | 5단계(5/10/20/30%) | 세밀한 포지션 제어 |

**백테스트 결과**: Sharpe 0.36, MDD -54.4%, Return +31.9%

### Phase 2: RL 정책 + 일별 리스크 관리 (2026-02-21)
**변경 파일**: `config/model_config_fast.yaml`, `main.py`

| 변경 항목 | 이전 | 이후 | 이유 |
|----------|------|------|------|
| `rl.entropy_coef` | 0.01 | 0.001 | RL 정책 확정성 향상 |
| `gan.learning_rate_g/d` | 1e-4 | 5e-5 | GAN 훈련 안정화 |
| backtest 일별 RiskManager | 없음 | 추가 | 주간 리밸런싱 보완 |

**백테스트 결과 (Phase 5만 재훈련)**: **Sharpe 0.62, MDD -25.89%, Return +29.9%**

### Phase 3: RL Simplex + 보상 클리핑 + Top-K 신호 (2026-02-21)
**변경 파일**: `models/rl/environment.py`, `models/gan/trainer.py`, `main.py`

| 변경 항목 | 이전 | 이후 | 이유 |
|----------|------|------|------|
| RL action 정규화 | 없음 | simplex 투영 (합=1, long-only) | 레버리지/폭발 방지 |
| reward clipping | 없음 | `clip(-10, 10)` | V.Loss 50,000+ → 76 해결 |
| 백테스트 신호 생성 | 항상 100% long | Top-3 섹터만 long, 나머지 neutral | 하락 섹터 회피 |
| GAN 조기 종료 | 없음 | `W-dist > best+5 → break` | 항상 epoch1 best 보장 |

**백테스트 결과**: **Sharpe 1.94, MDD -1.26%, Return +38.8%, Calmar 46.4**

> ⚠️ **주의**: MDD -1.26%는 과도하게 낙관적일 수 있음.
> Top-K + 일별 리스크 차단기 조합이 대부분의 기간을 현금 보유 상태로 만들 가능성 있음.
> Walk-forward 검증 필요.

### Phase 4: GAN 안정화 (2026-02-21)
**변경 파일**: `models/gan/model.py`, `config/model_config_fast.yaml`

| 변경 항목 | 이전 | 이후 | 이유 |
|----------|------|------|------|
| Generator Spectral Norm | 없음 | `spectral_norm(nn.Linear(...))` | Lipschitz 조건 강제, mode collapse 방지 |
| `n_critic` | 5 | 10 | Discriminator 강화 → Generator 발산 억제 |
| `generator_hidden_dims` | [128, 256, 128] | [64, 128, 64] | 용량 축소 → 과적합 방지 |

**GAN 훈련 결과**: Best W-dist 13.0 → **7.17** (개선), Early stopping Epoch 6에서 작동
**백테스트 결과**: **Sharpe 1.94, MDD -1.26% (Phase 3 수준 유지)**

---

## 현재 상태 (2026-02-21 04:37 기준)

### 훈련 상태

| 모델 | 최신 체크포인트 | 날짜 | 비고 |
|------|--------------|------|------|
| VAE | `vae_epoch30.pt` | 2026-02-20 | 30 epochs, kl_weight=0.01 |
| Transformer | `transformer_epoch20.pt` | 2026-02-20 | 20 epochs, dir_acc=52.4%, IC=0.094 |
| GAN | `gan_epoch1.pt` | 2026-02-21 | Spectral Norm + n_critic=10, best W-dist=7.17 |
| RL | `ppo_agent_epoch97.pt` | 2026-02-21 | simplex + reward clipping, best_reward=1.75 |
| Ensemble | `ensemble.pt` | 2026-02-21 | Phase 4 통합 완료 |

---

## 백테스트 성과 이력

| 버전 | Sharpe | MDD | Return | Calmar | 날짜 |
|------|--------|-----|--------|--------|------|
| 초기 (equal-weight 비교) | 0.25 | -36.4% | +14.3% | - | 2026-02-20 |
| Phase 1 후 | 0.36 | -54.4% | +31.9% | 0.87 | 2026-02-20 |
| Phase 2 후 | 0.62 | -25.9% | +29.9% | 1.71 | 2026-02-21 |
| Phase 3 후 | 1.94 | -1.26% | +38.8% | 46.4 | 2026-02-21 |
| **Phase 4 후 (현재 최고)** | **1.94** | **-1.26%** | **+38.8%** | **46.4** | 2026-02-21 |

**테스트 기간**: 180일 (전체 데이터의 15%, 2024년 후반 추정)
**초기 자본**: 1억원

---

## 알려진 문제점

### 1. GAN 부분 안정화 (개선됨)
- **현상**: Epoch 1이 best (W-dist 7.17), Early stopping으로 Epoch 6 종료
- **개선**: Spectral Norm + n_critic=10 + 구조 축소로 best W-dist 13→7.17
- **잔존 문제**: 여전히 epoch마다 W-dist 증가 패턴, 근본 해결은 미완
- **영향**: GAN 기여도 낮으나 best 품질이 개선됨

### 2. RL 에이전트 과레버리지 (심각)
- **현상**: 내부 평가 MDD=102%, portfolio_value가 66조원까지 폭발
- **원인**: SectorTradingEnv가 포지션 합계를 제한하지 않음 (레버리지 허용)
- **영향**: 실제 시그널이 불안정, 백테스트에서 매우 다른 성과

### 3. RL 엔트로피 여전히 높음 (보통)
- **현상**: entropy_coef=0.001에도 entropy 10+로 유지
- **원인**: Gaussian 연속 액션 공간에서 분산이 좁아지지 않음
- **영향**: 정책이 아직 충분히 확정적이지 않음

### 4. Transformer 방향 정확도 낮음 (보통)
- **현상**: Direction accuracy 52.4% (거의 무작위 수준)
- **영향**: 신호 품질의 근본적 제한

### 5. 백테스트 신호 생성 (개선 여지)
- **현상**: softmax 정규화로 항상 100% long 포지션
- **영향**: 하락장에서 현금 보유 불가

---

## 파일 구조 (핵심)

```
C:\src\Qunat_trading\
├── main.py                    # CLI 진입점 (train/infer/backtest)
├── config/
│   ├── settings_fast.yaml     # 빠른 실험용 설정
│   ├── model_config_fast.yaml # 빠른 실험용 하이퍼파라미터
│   └── sectors.yaml           # GICS 11 섹터 정의
├── data/                      # 데이터 파이프라인
├── models/                    # VAE / Transformer / GAN / RL / Ensemble
├── strategy/                  # 신호 생성 / 포트폴리오 최적화 / 리스크 관리
├── backtest/                  # 백테스트 엔진 / 메트릭 / 시각화
├── pipeline/                  # 학습/추론 파이프라인
├── saved_models/              # 체크포인트 저장
├── data/processed/            # 전처리된 데이터 (Parquet)
└── results/                   # 백테스트 결과 (PNG, HTML)
```

---

## 실행 명령어

```bash
# Python 경로 (conda 환경)
PYTHON=/c/Users/wogus/miniconda3/envs/quant/python.exe

# Phase 5(RL)만 재훈련 (~15분)
$PYTHON main.py train --start-phase 5 --config config/settings_fast.yaml

# Phase 4(GAN)부터 재훈련 (~80분)
$PYTHON main.py train --start-phase 4 --config config/settings_fast.yaml

# 전체 재훈련 (~3시간)
$PYTHON main.py train --start-phase 2 --config config/settings_fast.yaml

# 백테스트
$PYTHON main.py backtest --config config/settings_fast.yaml
```
