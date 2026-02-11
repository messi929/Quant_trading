# Alpha Signal Discovery Engine - 프로젝트 현황

## 프로젝트 개요

KOSPI/NASDAQ 시장 데이터에서 딥러닝으로 **새로운 수학적 지표(Alpha Signal)**를 자동 발견하고,
이를 기반으로 GICS 11개 섹터별 포트폴리오를 운용하는 퀀트 트레이딩 시스템.

기존 기술적 지표(RSI, MACD 등)에 의존하지 않고, 모델이 원시 데이터에서 직접 학습한다.

---

## 아키텍처

```
[Yahoo Finance / KRX] → [Data Pipeline] → [Feature Engineering (40+ raw features)]
                                                    │
                                 ┌──────────────────┼──────────────────┐
                                 ▼                  ▼                  ▼
                          [VAE (32-dim)]   [Transformer]      [Cond. WGAN-GP]
                         잠재 지표 발견    시간 패턴 학습     시장 시뮬레이션
                                 └──────────────────┼──────────────────┘
                                                    ▼
                                        [Attention Fusion Layer]
                                                    ▼
                                           [PPO RL Agent]
                                          섹터별 매매 결정
                                                    │
                                 ┌──────────────────┼──────────────────┐
                                 ▼                  ▼                  ▼
                          [Signal Gen]     [Portfolio Opt]     [Risk Mgmt]
                                                    │
                                           [Backtest Engine]
                                        (슬리피지/수수료 포함)
```

---

## 구현 현황

### 전체 통계

| 항목 | 수치 |
|------|------|
| 총 파일 수 | 47개 (Python 35, YAML 3, 기타 9) |
| Python 코드 라인 | 5,610줄 |
| YAML 설정 라인 | 267줄 |
| 패키지/모듈 | 11개 |
| 구현 완료율 | **100%** (전체 코드 구조) |

### 모듈별 상세

#### 1. Config (`config/`) - 3 파일, 267줄
| 파일 | 줄 수 | 설명 |
|------|-------|------|
| `settings.yaml` | 72 | 전역 설정 (GPU, 학습, 데이터, 백테스트) |
| `sectors.yaml` | 107 | GICS 11 섹터 정의 + 종목 매핑 |
| `model_config.yaml` | 88 | VAE/Transformer/GAN/RL/Ensemble 하이퍼파라미터 |

#### 2. Data Pipeline (`data/`) - 5 파일, 885줄
| 파일 | 줄 수 | 설명 |
|------|-------|------|
| `collector.py` | 228 | KOSPI(pykrx) + NASDAQ(yfinance) OHLCV 수집, 배치 다운로드 |
| `processor.py` | 170 | 정제, 상폐종목 제거, 결측치, 이상치 클리핑, RobustScaler |
| `sector_classifier.py` | 141 | GICS 섹터 분류 (KOSPI: 키워드 매칭, NASDAQ: yfinance 섹터) |
| `feature_engineer.py` | 150 | 40+ 원시 피처 (수익률, 변동성, 거래량비, 캔들 패턴, MA 거리 등) |
| `dataset.py` | 196 | PyTorch Dataset, 시간 기반 train/val/test 분할 |

#### 3. Models (`models/`) - 8 파일, 2,134줄
| 파일 | 줄 수 | 설명 |
|------|-------|------|
| `autoencoder/model.py` | 184 | VAE (Encoder→μ,σ→Reparameterize→Decoder), β-VAE 손실 |
| `autoencoder/trainer.py` | 229 | KL annealing, gradient accumulation, mixed precision |
| `transformer/model.py` | 214 | Temporal Transformer + Sector Cross-Attention + Positional Encoding |
| `transformer/trainer.py` | 188 | Cosine warmup, HuberLoss, 방향 정확도 + Rank IC 메트릭 |
| `gan/model.py` | 210 | Conditional WGAN-GP (Generator + Discriminator), 4 시장 레짐 |
| `gan/trainer.py` | 225 | Critic/Generator 교대 학습, Gradient Penalty |
| `rl/environment.py` | 212 | Gymnasium 섹터 트레이딩 환경 (Sharpe/Sortino 보상) |
| `rl/agent.py` | 268 | PPO Actor-Critic + GAE, 연속 액션 공간 |
| `rl/trainer.py` | 174 | 롤아웃 수집 → PPO 업데이트 루프 |
| `ensemble.py` | 183 | Attention 기반 모델 퓨전 (VAE + Transformer → RL) |

#### 4. Indicators (`indicators/`) - 3 파일, 445줄
| 파일 | 줄 수 | 설명 |
|------|-------|------|
| `generator.py` | 139 | 잠재 차원 → 해석 가능한 지표 변환 (상관 분석 기반 네이밍) |
| `evaluator.py` | 176 | 예측력(Rank IC), 안정성, 단조성, 신규성, 정보량 평가 |
| `registry.py` | 130 | 발견된 지표 저장/관리/버전 관리 (JSON 기반) |

#### 5. Strategy (`strategy/`) - 3 파일, 465줄
| 파일 | 줄 수 | 설명 |
|------|-------|------|
| `signal.py` | 134 | 모델 예측 + RL 할당 + 모멘텀 오버레이 → 최종 시그널 |
| `portfolio.py` | 150 | Max Sharpe, Min Variance, Risk Parity, Signal-weighted 최적화 |
| `risk.py` | 181 | 드로다운 한도, Vol 타겟팅, VaR/CVaR, Kelly 포지션 사이징 |

#### 6. Backtest (`backtest/`) - 3 파일, 571줄
| 파일 | 줄 수 | 설명 |
|------|-------|------|
| `engine.py` | 223 | 이벤트 드리븐 백테스트 + Walk-forward 최적화 |
| `metrics.py` | 147 | Sharpe, Sortino, Calmar, MDD, 승률, Profit Factor, 롤링 메트릭 |
| `visualizer.py` | 201 | Equity curve, Drawdown, 월별 수익률 히트맵, 섹터 배분 차트 |

#### 7. Pipeline & Entry (`pipeline/`, `main.py`) - 3 파일, 836줄
| 파일 | 줄 수 | 설명 |
|------|-------|------|
| `train_pipeline.py` | 423 | 6단계 학습 파이프라인 오케스트레이션 |
| `inference_pipeline.py` | 203 | 실시간 추론 파이프라인 |
| `main.py` | 210 | CLI 진입점 (train / infer / backtest) |

#### 8. Utils (`utils/`) - 3 파일, 321줄
| 파일 | 줄 수 | 설명 |
|------|-------|------|
| `logger.py` | 53 | loguru 기반 로깅 (콘솔 + 파일, 로테이션) |
| `device.py` | 147 | GPU 관리, Mixed Precision, torch.compile, 시드 설정 |
| `storage.py` | 121 | Parquet/Numpy/체크포인트 저장, 메모리 맵 지원 |

---

## RTX 4060 Ti (8GB) 최적화 전략

| 기법 | 구현 위치 | 효과 |
|------|-----------|------|
| Mixed Precision (FP16) | `utils/device.py` - `autocast()` | VRAM ~50% 절약 |
| Gradient Accumulation (8x) | 모든 trainer | 실효 배치 256 (물리 32×8) |
| Gradient Checkpointing | Transformer | 메모리 절약 |
| torch.compile | `utils/device.py` | PyTorch 2.x 최적화 |
| TF32 활성화 | `utils/device.py` | Ampere+ GPU 속도 향상 |
| Memory-mapped files | `utils/storage.py` | 대용량 데이터 관리 |
| 순차 섹터 학습 | `train_pipeline.py` | 동시 VRAM 사용 방지 |

---

## 데이터 규모 추정

| 시장 | 종목 수 | 기간 | 예상 행 수 |
|------|---------|------|-----------|
| KOSPI | ~2,500 | 20년 | ~12.5M |
| NASDAQ (S&P500+NQ100) | ~600 | 20년 | ~3M |
| **합계** | ~3,100 | - | **~15.5M** |

피처 포함 시 디스크 ~3-5GB, 메모리는 청크/메모리맵 처리

---

## 사용법

```bash
# 1. 환경 설정
setup_env.bat                      # Conda 환경 생성 + 패키지 설치

# 2. 학습
python main.py train               # 전체 파이프라인 (데이터 수집 포함)
python main.py train --skip-data   # 데이터 수집 건너뛰기

# 3. 추론
python main.py infer               # 현재 시장 시그널 생성

# 4. 백테스트
python main.py backtest            # 테스트 기간 백테스트
python main.py backtest --walk-forward  # Walk-forward 최적화
```

---

## 검증 계획

| 단계 | 검증 항목 | 기대 지표 |
|------|-----------|-----------|
| 데이터 | 수집 무결성, 섹터 분류 정확도 | >80% 섹터 매핑 |
| VAE | 재구성 오류 수렴, 잠재 공간 분리 | 재구성 MSE < 0.01 |
| Transformer | 방향 예측 정확도, Rank IC | Dir Acc > 52%, IC > 0.03 |
| GAN | 합성 데이터 통계적 유사성 | KS test p > 0.05 |
| RL | 학습 곡선 수렴, 보상 증가 | 양의 평균 보상 |
| 백테스트 | OOS Sharpe, MDD | Sharpe > 0.5, MDD < 20% |
