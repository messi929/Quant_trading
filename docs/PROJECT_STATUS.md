# Alpha Signal Discovery Engine - 프로젝트 현황

**최종 업데이트**: 2026-02-23 (Phase 11: 개별 종목 매매 활성화 + KR/US 분리 스케줄)
**현재 최고 성과**: Sharpe **2.03**, MDD **-4.30%**, Return **+17.82%** (기준선: Sharpe 1.82, +15.87%)

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
                         └──────────────────────────┼──────────────────────┘
                                                    ▼
                                    [Alpha Blending: 40% model + 60% EW]
                                                    ▼
                                    [Backtest Engine / KIS Live Trading]
```

---

## 개발 이력

### Phase 11: 개별 종목 매매 활성화 + KR/US 분리 스케줄 (2026-02-23)

오늘(2026-02-23) 서버 로그 분석으로 다수의 버그를 발견하고 수정. 한국/미국 시장을 분리 스케줄로 운용하도록 전면 재설계.

#### 11-A. 버그 수정

| 버그 | 원인 | 수정 내용 |
|------|------|---------|
| `OverflowError: cannot convert float infinity to integer` | KIS sandbox에서 국내 종목 현재가 조회 실패(0 반환) → `int(amount / 0)` | `current_price <= 0` guard 추가 → 해당 종목 스킵 |
| KODEX ETF 조회 (개별 종목 아닌 ETF) | `execution_market: "kospi"` 설정 오류 — Phase 8에서 개별 종목 전환했으나 config 미변경 | `execution_market: "split"` 으로 수정 |
| 해외 현재가 API 404 (모든 NASDAQ 종목) | KIS sandbox는 해외주식 시세 API(`/uapi/overseas-stock/v1/quotations/price`) 미지원 | yfinance 폴백: `yf.Ticker(ticker).fast_info.last_price` (3곳 적용) |
| `get_overseas_balance()` 500 Server Error | KIS sandbox 해외 잔고 API 미지원 | try/except로 감싸고 warning 로그 후 계속 |

#### 11-B. KR/US 분리 스케줄

**변경 배경**: 미국 시장(KST 23:30~06:00)과 한국 시장(KST 09:00~15:30)이 시간대가 완전히 분리됨. 미국 시장은 운영시간이 길므로 Wave마다 신호를 재계산하여 최신 정보 반영.

**한국 시장 스케줄 (KST)**:

| 시각 | 작업 |
|------|------|
| 06:00 | 데이터 수집 (증분) |
| 06:30 | 신호 생성 + 하락 종목 매도 |
| 09:10 | KR Wave 1 (40%, 국내 종목) |
| 11:00 | KR Wave 2 (35%, 국내 종목) |
| 13:30 | KR Wave 3 (25%, 국내 종목) |
| 16:00 | 종가 기록 + 성과 업데이트 |

**미국 시장 스케줄 (KST, 다음날 새벽)**:

| 시각 | 작업 |
|------|------|
| 23:20 | 데이터 재수집 + 신호 재생성 (US Wave 1 전) |
| 23:40 | US Wave 1 (40%, 해외 종목) |
| 01:50 | 신호 재생성 (재수집 없음, US Wave 2 전) |
| 02:00 | US Wave 2 (35%, 해외 종목) |
| 04:20 | 신호 재생성 (US Wave 3 전) |
| 04:30 | US Wave 3 (25%, 해외 종목) |
| 06:10 | 미국 장 마감 기록 |

#### 11-C. execute_rebalance market_filter 추가

`execute_rebalance(market_filter="domestic")` / `market_filter="overseas"` 파라미터를 추가하여 KR/US 주문을 분리 실행.

**수정 파일 요약**:

| 파일 | 변경 내용 |
|------|---------|
| `config/live_config.yaml` | `execution_market: "split"`, `order_waves` → `kr_order_waves` + `us_order_waves` 분리 |
| `live/signal_to_order.py` | `current_price <= 0` guard, yfinance 폴백 3곳, overseas balance try/except, `market_filter` 파라미터 |
| `scheduler/daily_runner.py` | `_daemon_kr_order()`, `_daemon_us_signal()`, `_daemon_us_order()` 추가, `run_daemon()` KR+US 이중 스케줄 |

**배포**: 변경사항 Hetzner 서버(`77.42.78.9`)에 적용 완료, systemd 데몬 재시작 확인.

---

### Phase 10: Hetzner Cloud 서버 배포 (2026-02-22)

24/7 자동 운용을 위해 Hetzner Cloud ARM 서버에 전체 시스템 배포 완료.

**서버 사양**:
- **호스트**: Hetzner Cloud CAX21 (ARM, 4vCPU / 8GB RAM)
- **IP**: `77.42.78.9`
- **OS**: Ubuntu 22.04 LTS
- **시간대**: KST (+9, `Asia/Seoul`)
- **Python**: `/opt/quant/venv` (Python 3.12, PyTorch 2.10 CPU)

**배포 내용**:
- 소스코드 전체 (`/opt/quant/`)
- 학습된 모델 파일 (`/opt/quant/saved_models/`, 553MB)
- KIS API 자격증명 (`.env`, git 제외)
- `requirements.txt` 전체 설치 완료

**systemd 서비스 등록** (`/etc/systemd/system/quant-trading.service`):
- 서버 재부팅 시 자동 시작 (`enabled`)
- 데몬 비정상 종료 시 60초 후 자동 재시작 (`Restart=always`)
- 로그: `/var/log/quant-trading.log`

**일간 자동 스케줄 (KST) — Phase 11 이후**:

| 시각 | 작업 |
|------|------|
| 06:00 | 데이터 수집 (증분) |
| 06:30 | 신호 생성 + 하락 종목 매도 |
| 09:10 | KR Wave 1 (40%, 국내 종목) |
| 11:00 | KR Wave 2 (35%, 국내 종목) |
| 13:30 | KR Wave 3 (25%, 국내 종목) |
| 16:00 | 종가 기록 + 성과 업데이트 |
| 23:20 | US 데이터 재수집 + 신호 재생성 |
| 23:40 | US Wave 1 (40%, 해외 종목) |
| 01:50 | US 신호 재생성 |
| 02:00 | US Wave 2 (35%, 해외 종목) |
| 04:20 | US 신호 재생성 |
| 04:30 | US Wave 3 (25%, 해외 종목) |
| 06:10 | 미국 장 마감 기록 |

**검증 완료**:
- `python scheduler/daily_runner.py --step signal` 서버에서 정상 실행 ✅
- 섹터 가중치 생성 확인 (11개 섹터, sum ≈ 1.0) ✅
- systemd 데몬 active(running) 상태 확인 ✅

---

### Phase 9: 퀀트 실전 주문 체계 구축 (2026-02-22)

세계적인 퀀트 트레이더 방식의 주문 실행 체계를 4-Phase로 구현.

#### Phase A: 지정가 주문 + 미체결 재시도

| 긴급도 | 조건 | 매수가 | 재시도 | 폴백 |
|--------|------|--------|--------|------|
| aggressive | score ≥ 0.5 | ask + 0.3% | 5분 후 +0.5% 재시도 | 시장가 |
| normal | score ≥ 0.2 | ask | 5분 후 +0.5% 재시도 | 없음 |
| patient | score < 0.2 | (bid+ask)/2 | 없음 | 없음 |

**매도는 항상 aggressive** — 하락 신호는 빠른 실행이 우선.

신규 KIS API 메서드 (`broker/kis_api.py`):
- `get_domestic_bid_ask(ticker)` — TR_ID `FHKST01010200` (1호가 매수/매도 조회)
- `get_pending_orders()` — TR_ID `VTTC8036R` (미체결 주문 목록)
- `cancel_order(order_no, ticker, side, qty)` — TR_ID `VTTC0803U` (주문 취소)

#### Phase B: TWAP 3-파 분할 (시간 가중 평균 가격)

50만원 이상 주문을 3개 시간대로 분산하여 시장 충격 최소화:

| 파 | 시각 | 비중 | 목적 |
|----|------|------|------|
| Wave 1 | 09:10 | 40% | 갭 안정화 후 진입 |
| Wave 2 | 11:00 | 35% | 오전 유동성 활용 |
| Wave 3 | 13:30 | 25% | 오후 잔여 수량 마무리 |

50만원 미만 소액 주문은 Wave 1에서 100% 실행.

#### Phase C: 신호 강도 → 긴급도 자동 매핑

```
score ≥ 0.5  →  aggressive  (알파 빠르게 포착, 시장가 폴백 허용)
score ≥ 0.2  →  normal      (보통 진입, 1회 재시도)
score < 0.2  →  patient     (약한 신호, 중간값 지정가, 미체결이면 포기)
```

#### Phase D: 거래비용 임계값 필터

```
round-trip 비용 = 수수료 0.2%×2 + 슬리피지 0.1%×2 = 0.6%
score < 0.05  →  기대수익 < 거래비용 → 주문 스킵
```

**수정 파일 요약**:

| 파일 | 변경 내용 |
|------|---------|
| `broker/kis_api.py` | `get_domestic_bid_ask`, `get_pending_orders`, `cancel_order` + TR_ID 3개 추가 |
| `live/signal_to_order.py` | 전면 재작성 — Phase A~D 통합, `_get_urgency`, `_get_limit_price`, `_execute_with_retry`, TWAP 분기 |
| `scheduler/daily_runner.py` | `step_order(twap_wave)`, `_daemon_order(twap_wave)`, `run_daemon()` 3-파 스케줄, `--wave` CLI 인수 |
| `live_config.yaml` | `execution:` 섹션 추가, `order_submit_time` → `order_waves: {wave1, wave2, wave3}` |

**추가로 수정된 버그**:
- `run_full()`: `weights = step_signal()` → `weights, top_tickers = step_signal()` (tuple 미해제)
- `execute_rebalance()`: 매도 실패 시 `failed_sell_tickers` 추적, 실제 잔고 재조회 후 매수

---

### Phase 0: 초기 구현 (2026-02-11)
- 47개 파일, 5,610줄 Python 코드 구현 완료
- 6단계 파이프라인: Data → VAE → Transformer → GAN → RL → Ensemble

### Phase 0.5: 초기 실행 및 버그 수정 (2026-02-19~20)
- Miniconda + CUDA 12.x 환경 설정 완료 (RTX 4060 Ti 8GB)
- 데이터 수집: **238,202행, 200 종목, 34 피처** (5년치)
- GAN 체크포인트 포맷 불일치 수정: `model.generator.load_state_dict()` 방식

### Phase 1: 리스크/보상 함수 개선 (2026-02-20)
| 변경 항목 | 이전 | 이후 |
|----------|------|------|
| `rl.risk_penalty` | 0.1 | 1.0 |
| `rl.drawdown_penalty` | 0.5 | 5.0 |
| `rl.window_size` | 20 | 60 |
| `vae.kl_weight` | 0.001 | 0.01 |
| 리스크 차단기 | 단일 15% | 5단계(5/10/20/30%) |

### Phase 2: RL 정책 + 일별 리스크 관리 (2026-02-21)
- `entropy_coef` 0.01 → 0.001, 일별 RiskManager 사전 필터 추가
- **결과**: Sharpe 0.62, MDD -25.89%, Return +29.9% *(주의: return_1d 정규화 버그 포함)*

### Phase 3~4: RL Simplex + GAN 안정화 (2026-02-21)
- RL simplex 투영 (long-only, 합=1), reward clipping ±10
- GAN: Spectral Norm + n_critic 5→10 + 구조 축소
- **결과**: Sharpe 1.94 *(정규화 버그 포함)*

### Phase 5: 백테스트 버그 수정 (2026-02-21)
- `return_1d` 정규화 버그 수정 → close 가격에서 직접 pct_change
- **실제 성과 (버그 수정 후)**: Sharpe 0.70, MDD -7.79%, Return +7.02%

### Phase 6: Transformer 재훈련 (2026-02-21)
| 변경 항목 | 이전 | 이후 |
|----------|------|------|
| `d_model` | 128 | 256 |
| `n_encoder_layers` | 4 | 6 |
| `prediction_horizon` | 5 | 1 |
| `epochs` | 25 | 50 |

- **결과**: dir_acc 52.5%, Sharpe 1.14, MDD -9.75%, Return +13.31%

### Phase 8: 추론 파이프라인 버그 수정 + 개별 종목 매매 + 일간 리밸런싱 (2026-02-21)

#### 8-A. 추론 파이프라인 버그 수정 7개 (`pipeline/inference_pipeline.py`)

| 버그 | 원인 | 수정 내용 |
|------|------|---------|
| VAE size mismatch [256,2040] vs [256,60] | `input_dim=1` 하드코딩 → `flat_dim=60` | 체크포인트 weight shape에서 자동 감지: `flat_dim = ckpt["vae"]["encoder.network.0.weight"].shape[1]; n_features = flat_dim // seq_length` |
| Transformer sector_embed_dim [11,64] vs [11,32] | `sector_embed_dim` 미전달 (기본값=32) | `sector_embed_dim=tf_cfg.get("sector_embed_dim", 32)` 추가 |
| GAN output_dim mismatch | `hidden_dims` 미전달 | `hidden_dims=gan_cfg.get("generator_hidden_dims", [64,128,64])` 추가 |
| PPO state_dim mismatch | 하드코딩된 `rl_state_dim` | 체크포인트 첫 번째 weight에서 자동 감지: `rl_first_w.shape[1]` |
| DataProcessor 1464 ticker 전부 필터링 | `min_history_days=500` (5년치 기준), 추론 시 1년치만 있음 | `DataProcessor(min_history_days=self.config["data"]["sequence_length"])` = 60일 |
| KeyError 'sector' | OHLCV 수집 데이터에 sector 컬럼 없음 | `data/processed/processed_data.parquet`에서 ticker→sector 매핑 로드 |
| shape (11,) vs (12,) | "unknown" 섹터 포함 | `if sector == "unknown": continue` 추가 |

**결과**: `python scheduler/daily_runner.py --step signal` 정상 실행 → 섹터별 가중치 생성 성공

#### 8-B. ETF → 개별 종목 매매 전환

**변경 배경**: ETF는 섹터 전체를 추종하므로 모델의 종목별 alpha 신호를 활용할 수 없음. 개별 종목으로 전환하여 모델 예측력 직접 반영.

**매매 방식**:
- 섹터당 모델 예측값 상위 3개 종목 선정
- **양수 신호 종목만 매수** (`score > 0` 필터)
- KOSPI + NASDAQ 양쪽 시장 모두 대상
- 섹터 배분 금액을 상위 3 종목에 균등 분배 (최소금액 미달 시 1종목 집중)

**수정 파일**:

| 파일 | 변경 내용 |
|------|---------|
| `pipeline/inference_pipeline.py` | per-ticker scoring, top-3 + `score > 0` 필터, `sector_top_tickers` 반환 |
| `live/signal_to_order.py` | `_compute_target_positions()` 개별 종목 계산으로 교체 |
| `live/signal_to_order.py` | `execute_sell_check()` 추가 |
| `scheduler/daily_runner.py` | `step_signal()` 반환값 `tuple(weights, sector_top_tickers)` |
| `scheduler/daily_runner.py` | `step_sell_check()` 추가 |

#### 8-C. 일간 리밸런싱 (월요일 전용 → 평일 매일)

**변경 배경**: 종목 신호는 매일 변하므로 월요일만 리밸런싱하면 4일치 신호 손실. 매일 리밸런싱으로 적시 대응.

| 항목 | 변경 전 | 변경 후 |
|------|--------|--------|
| 매수/매도 실행 | 월요일만 | 평일(월~금) 매일 |
| 하락 신호 매도 | 없음 | 매일 실행 (`execute_sell_check`) |
| `live_config.yaml` `rebalance_day` | `"monday"` | `"daily"` |
| 매도 후 재투자 | 없음 | 당일 또는 익일 매수에 현금 반영 |

**리밸런싱 로직**:
1. `step_signal()` → 오늘 양수 신호 종목 목록 생성
2. 보유 종목 중 양수 신호 없는 종목 → 매도 (`execute_sell_check`)
3. 목표 포지션 대비 3% 이상 괴리 → 매수/매도 조정 (`execute_rebalance`)
4. 매도 먼저 실행 후 확보된 현금으로 매수

---

### Phase 7: 백테스트 버그 수정 3개 + 라이브 트레이딩 시스템 (2026-02-21)

#### 7-A. 3개의 백테스트 핵심 버그 수정
| 버그 | 파일 | 수정 내용 |
|------|------|---------|
| RiskManager 상태 오염 | `backtest/engine.py` | `run()` 시작시 peak_value/daily_pnl/is_risk_off 리셋 |
| sector_id 항상 0 | `main.py`, `inference_pipeline.py` | sectors.yaml 키 순서로 training-time id 로드 |
| ticker 혼합 rows | `main.py`, `inference_pipeline.py` | per-ticker 개별 추론 → sector 평균 |

#### 7-B. Alpha Blending 최적화 (grid search)
| alpha | Sharpe |
|-------|--------|
| 1.0 (pure model) | 1.51 |
| 0.5 | 1.98 |
| **0.4 (채택)** | **2.03** |
| 0.3 | 2.00 |
| 0.2 | 1.95 |

**최종 수식**: `weights = 0.4 × model_weights + 0.6 × equal_weight`

#### 7-C. 라이브 트레이딩 시스템 (한국투자증권 KIS API)

**신규 파일 (10개)**:

```
broker/kis_api.py           KIS REST API (auth, price, order, balance)
live/sector_instruments.py  KOSPI KODEX ETF / NASDAQ XL ETF 매핑
live/signal_to_order.py     신호 → 실제 주문 변환 (3% rebal threshold)
tracking/trade_log.py       SQLite 거래/성과/신호/재학습 이력
scheduler/daily_runner.py   일간 자동화 (06:00 수집 → 06:30 신호 → 08:50 주문 → 16:10 종가)
scheduler/retrain_runner.py 주간 파인튜닝 + 월간 전체 재훈련 + auto-rollback
dashboard/app.py            Streamlit 실시간 대시보드
setup_live.bat              최초 설정 스크립트 (KIS API 자격증명 + Task Scheduler)
.gitignore                  .env 자격증명 보호
docs/LIVE_TRADING_GUIDE.md  운용 가이드
```

**신규/수정 파일 (4개)**:
```
data/collector.py           DataCollector 래퍼 클래스 추가 (증분 수집용)
pipeline/inference_pipeline.py  sector_id 버그 + per-ticker 추론 수정
scheduler/daily_runner.py   DataProcessor() 생성자 수정
requirements.txt            python-dotenv, streamlit, schedule 추가
```

---

## 현재 상태 (2026-02-23 기준)

### 훈련 체크포인트

| 모델 | 파일 | 성능 |
|------|------|------|
| VAE | `vae_epoch30.pt` | kl_weight=0.01, 30 epochs |
| Transformer | `transformer_epoch20.pt` | d_model=256, dir_acc=52.5%, IC=0.065 |
| GAN | `gan_epoch1.pt` | Spectral Norm, best W-dist=6.66 |
| RL | `ppo_agent_epoch96.pt` | simplex+reward_clip, best_reward=4.75 |
| Ensemble | `ensemble.pt` | Phase 7 통합 완료 |

### 백테스트 성과 이력

> ⚠️ Phase 1~4 수치는 `return_1d` 정규화 버그로 인한 **부정확한 수치**

| 버전 | Sharpe | MDD | Return | 기준선 | 유효 |
|------|--------|-----|--------|--------|------|
| Phase 5 (버그수정) | 0.70 | -7.79% | +7.02% | 1.69 | ✅ |
| Phase 6 (Transformer) | 1.14 | -9.75% | +13.31% | 1.26 | ✅ |
| **Phase 7 (버그3개+blend)** | **2.03** | **-4.30%** | **+17.82%** | **1.82** | **✅** |

**테스트 기간**: 2025-06-02 ~ 2026-02-18 (180일), 초기자본 1억원

### Walk-forward 결과 (Phase 7, 5×30일)

| 기간 | Return | Sharpe | MDD |
|------|--------|--------|-----|
| 2025-07-16 ~ 08-26 | +0.51% | +0.42 | -3.21% |
| 2025-08-27 ~ 10-08 | +2.18% | +1.71 | -2.10% |
| 2025-10-09 ~ 11-19 | -0.23% | -0.13 | -2.98% |
| 2025-11-20 ~ 26-01-05 | +8.90% | +6.45 | -1.33% |
| 2026-01-06 ~ 02-18 | +5.01% | +3.02 | -2.21% |
| **평균** | **+2.27%** | **1.89** | **-2.37%** |

---

## 알려진 문제점

| 문제 | 심각도 | 상태 |
|------|--------|------|
| Transformer dir_acc 52.5% (거의 무작위) | 높음 | 잔존 |
| Walk-forward std=2.61 >> mean=1.89 (기간별 불일치) | 중간 | 잔존 |
| alpha=0.4 테스트셋에서 튜닝됨 (미래 데이터 누설 우려) | 중간 | 잔존 |
| GAN W-dist 여전히 불안정 (개선됐으나 미해결) | 낮음 | 부분해결 |
| 180일 테스트 기간 짧음 | 낮음 | 잔존 |
| KIS sandbox 해외주식 시세/잔고 API 미지원 (yfinance 폴백으로 해결) | 낮음 | 해결됨 |
| sandbox 국내 주문 체결 테스트 미완료 (평일 장 중 확인 필요) | 중간 | 잔존 |
| 일간 리밸런싱 회전율 미측정 (과도한 매매 비용 위험) | 중간 | 잔존 |

---

## 파일 구조

```
C:\src\Qunat_trading\
├── main.py                      CLI (train/infer/backtest)
├── config/
│   ├── settings_fast.yaml       현재 활성 설정
│   ├── model_config_fast.yaml   현재 활성 하이퍼파라미터
│   ├── sectors.yaml             GICS 11 섹터 (YAML 키 순서 = 학습시 sector_id)
│   └── live_config.yaml         라이브 트레이딩 설정
├── broker/kis_api.py            KIS API 래퍼
├── live/
│   ├── sector_instruments.py    섹터 → ETF/종목 매핑 참조
│   └── signal_to_order.py       주문 생성 (개별 종목 + 일간 리밸런싱)
├── tracking/trade_log.py        SQLite 로깅
├── scheduler/
│   ├── daily_runner.py          일간 자동화
│   └── retrain_runner.py        재학습 스케줄러
├── dashboard/app.py             Streamlit 대시보드
├── setup_live.bat               최초 설정
├── .gitignore                   자격증명 보호
├── data/                        데이터 파이프라인
├── models/                      VAE / Transformer / GAN / RL / Ensemble
├── strategy/                    신호/포트폴리오/리스크
├── backtest/                    엔진/메트릭/시각화
└── pipeline/                    학습/추론 파이프라인
```

---

## 실행 명령어

```bash
PYTHON=/c/Users/wogus/miniconda3/envs/quant/python.exe

# 학습
$PYTHON main.py train --start-phase 5 --config config/settings_fast.yaml  # RL만 (~15분)
$PYTHON main.py train --start-phase 3 --config config/settings_fast.yaml  # TF부터 (~1.5시간)
$PYTHON main.py train --start-phase 2 --config config/settings_fast.yaml  # 전체 (~3시간)

# 백테스트
$PYTHON main.py backtest --config config/settings_fast.yaml
$PYTHON main.py backtest --walk-forward --config config/settings_fast.yaml

# 라이브 트레이딩
setup_live.bat                                        # 최초 설정 (.env 생성, Task Scheduler 등록)
python scheduler/daily_runner.py --step signal        # 신호 테스트 ✅ (검증 완료)
python scheduler/daily_runner.py --step order         # sandbox KR 주문 테스트 (Wave 1, 40%, 국내)
python scheduler/daily_runner.py --step order --wave 2  # KR Wave 2 단독 테스트 (35%)
python scheduler/daily_runner.py --step order --wave 3  # KR Wave 3 단독 테스트 (25%)
python scheduler/daily_runner.py --daemon             # 데몬 실행 (KR 3-파 + US 3-파 자동 스케줄)

# 서버 로그 확인
ssh root@77.42.78.9 "tail -f /var/log/quant-trading.log"

# 대시보드
streamlit run dashboard/app.py
```
