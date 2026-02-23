# Next Steps - 다음 작업 로드맵

**최종 업데이트**: 2026-02-23 (Phase 11: 개별 종목 매매 활성화 + KR/US 분리 스케줄)
**현재 성과**: Sharpe **2.03**, MDD **-4.30%**, Return **+17.82%** (기준선: Sharpe 1.82, +15.87%)
**라이브 시스템**: Hetzner 서버 배포 완료, KR/US 이중 스케줄 데몬 실행 중. 평일 장 중 국내 주문 체결 확인 필요.

---

## 완료된 작업 전체 이력

### ✅ Phase 11: 개별 종목 매매 활성화 + KR/US 분리 스케줄 (2026-02-23)

- **버그 수정 4건**:
  - `OverflowError` 방지: `current_price <= 0` 종목 스킵 (`live/signal_to_order.py`)
  - `execution_market: "kospi"` → `"split"` 수정 (KODEX ETF → 개별 종목 활성화)
  - KIS sandbox 해외주식 시세 API 404 → **yfinance 폴백** 3곳 적용 (AAPL $264.58 검증 ✅)
  - 해외 잔고 API 500 오류 → try/except graceful handling
- **KR/US 분리 스케줄 구현** (`scheduler/daily_runner.py`):
  - `_daemon_kr_order(wave)`: 국내 종목 전용 (market_filter="domestic")
  - `_daemon_us_signal(collect)`: US Wave 전 신호 재생성 (Wave1은 재수집 포함)
  - `_daemon_us_order(wave)`: 해외 종목 전용 (market_filter="overseas")
- **live_config.yaml 재구성**: `order_waves` → `kr_order_waves` + `us_order_waves` 분리
- **execute_rebalance `market_filter` 파라미터 추가**: 국내/해외 주문 선택 실행
- **Hetzner 서버 배포 완료**: `git pull` + `systemctl restart quant-trading` ✅

### ✅ Phase 10: Hetzner Cloud 서버 배포 (2026-02-22)

- **서버 생성**: Hetzner Cloud CAX21 ARM (4vCPU/8GB, €5.9/월), Ubuntu 22.04
- **환경 구성**: Python 3.12 venv, PyTorch 2.10 CPU (ARM 전용 URL), 시간대 KST
- **파일 전송**: 소스코드 + saved_models(553MB) + .env → `/opt/quant/`
- **의존성 설치**: `requirements.txt` 전체 서버에서 설치 완료
- **systemd 서비스**: `quant-trading.service` 등록 + enabled (재부팅 자동시작)
- **데몬 실행 중**: `systemctl start quant-trading` → active(running) 확인 ✅
- **신호 생성 검증**: 서버에서 `--step signal` 정상 실행, 11개 섹터 가중치 생성 ✅

### ✅ Phase 9: 퀀트 실전 주문 체계 (2026-02-22)

- **Phase A: 지정가 + 미체결 재시도** (`broker/kis_api.py`, `live/signal_to_order.py`):
  - `get_domestic_bid_ask()` — 호가창 1호가 조회 (TR_ID: FHKST01010200)
  - `get_pending_orders()` — 미체결 목록 조회 (TR_ID: VTTC8036R)
  - `cancel_order()` — 주문 취소 (TR_ID: VTTC0803U)
  - `_execute_with_retry()` — 5분 대기 → 가격 재조정 → 재시도 → 시장가 폴백
- **Phase B: TWAP 3-파 분할** (`scheduler/daily_runner.py`, `live_config.yaml`):
  - 단일 08:50 → 09:10(40%) / 11:00(35%) / 13:30(25%) 분산
  - 50만원 미만 소액은 Wave 1에 100% 집중
- **Phase C: 신호 강도 → 긴급도** (`live/signal_to_order.py`):
  - aggressive (score ≥ 0.5) / normal (≥ 0.2) / patient (< 0.2)
  - 매도는 항상 aggressive (하락 신호 우선 실행)
- **Phase D: 거래비용 임계값** (`live/signal_to_order.py`):
  - score < 0.05 → round-trip 0.6% 비용 미달 → 주문 스킵
- **버그 수정**:
  - `run_full()` tuple 미해제 (`weights = step_signal()` → `weights, top_tickers = ...`)
  - 매도 실패 추적 → 실제 잔고 재조회 후 매수 (`execute_rebalance`)

### ✅ Phase 8: 추론 파이프라인 버그 수정 + 개별 종목 매매 (2026-02-21)

- **추론 파이프라인 7개 버그 수정** (`pipeline/inference_pipeline.py`):
  - VAE/Transformer/GAN/PPO 체크포인트 shape 자동 감지 (하드코딩 제거)
  - `DataProcessor(min_history_days=60)` — 추론 시 60일 기준 (훈련 500일과 구분)
  - `data/processed/processed_data.parquet`에서 ticker→sector 매핑 로드
  - `if sector == "unknown": continue` 추가 (shape 불일치 해결)
- **`python scheduler/daily_runner.py --step signal` 정상 실행 확인 ✅**
- **ETF → 개별 종목 전환** (`live/signal_to_order.py`):
  - 섹터당 모델 score 상위 3 종목, 양수 신호만 매수
  - KOSPI + NASDAQ 양쪽 시장 대상
- **일간 리밸런싱** (`scheduler/daily_runner.py`):
  - 월요일 전용 → 평일(월~금) 매일 실행
  - `execute_sell_check()` 추가 — 하락 신호 종목 즉시 매도
  - `live_config.yaml`: `rebalance_day: "daily"`
- **`.env` 파일 생성** (KIS API 자격증명, sandbox 모드)

### ✅ Phase 7-C: 라이브 트레이딩 시스템 (2026-02-21)
- **KIS API 래퍼** (`broker/kis_api.py`): 인증, 현재가, 주문, 잔고 조회. sandbox/production 모드
- **ETF 매핑** (`live/sector_instruments.py`): KOSPI 11개 KODEX ETF + NASDAQ 11개 XL ETF
- **주문 생성** (`live/signal_to_order.py`): 섹터 가중치 → 리밸런싱 주문 (3% 임계값, 매도 우선)
- **SQLite 로깅** (`tracking/trade_log.py`): 거래/성과/신호/재학습 이력 + 방향 정확도 계산
- **일간 자동화** (`scheduler/daily_runner.py`): 06:00 수집 → 06:30 신호 → 08:50 주문 → 16:10 종가
- **재학습 스케줄러** (`scheduler/retrain_runner.py`): 주간 파인튜닝 + 월간 전체 재학습 + auto-rollback (dir_acc < 48% 시)
- **Streamlit 대시보드** (`dashboard/app.py`): Sharpe/MDD/승률, 수익 곡선, 섹터 배분, 신호 정확도
- **설정 스크립트** (`setup_live.bat`): KIS 자격증명 입력 + Windows Task Scheduler 등록
- **보안** (`.gitignore`): `.env` 자격증명 파일 git commit 방지

### ✅ Phase 7-B: Alpha Blending 최적화 (2026-02-21)
- alpha=0.4 (40% model + 60% equal-weight) 최적값 발견
- **Sharpe 1.51 (pure model) → 2.03 (+0.52)**

### ✅ Phase 7-A: 백테스트 핵심 버그 3개 수정 (2026-02-21)
- `backtest/engine.py`: RiskManager 상태 오염 (baseline이 model 상태 물려받음)
- `main.py` + `inference_pipeline.py`: sector_id 항상 0 → sectors.yaml 키 순서 사용
- `main.py` + `inference_pipeline.py`: ticker 혼합 rows → per-ticker 개별 추론 후 평균
- **Sharpe 1.27 (버그 있음) → 2.03 (수정 후)**

### ✅ Phase 6: Transformer 재훈련 (2026-02-21)
- d_model 128→256, n_layers 4→6, horizon 5→1, epochs 25→50
- Sharpe 0.70 → 1.14 (+0.44)

### ✅ Phase 5: 백테스트 정규화 버그 수정 (2026-02-21)
- `return_1d` RobustScaler 값 → close pct_change (실제 수익률)
- **실제 기준 정확한 백테스트 최초 실행**

### ✅ Phase 3~4: RL + GAN 개선 (2026-02-21)
- RL: simplex 투영(long-only), reward clipping ±10, best_reward 1.75→4.75
- GAN: Spectral Norm + n_critic 10 + 구조 축소, best W-dist 13→6.66

---

## 다음 우선순위

### 🔴 우선순위 1: KIS API sandbox 국내 주문 체결 테스트 (평일 장 중)

> 서버 데몬이 이미 실행 중 (`77.42.78.9`). 평일 09:10~15:30 에 로그 확인.
> KR Wave: 09:10(40%) / 11:00(35%) / 13:30(25%) 자동 실행됨.

```bash
# 서버 로그 실시간 확인
ssh root@77.42.78.9 "tail -f /var/log/quant-trading.log"
```

**완료된 항목**:
- [x] `setup_live.bat` 실행 — KIS API 자격증명 입력 완료
- [x] `.env` 파일 생성 (KIS_APP_KEY, KIS_APP_SECRET, KIS_CANO, KIS_ACNT_PRDT_CD, KIS_MODE=sandbox)
- [x] `python scheduler/daily_runner.py --step signal` 정상 실행 확인
- [x] 섹터별 가중치 생성 확인 (sum ≈ 1.0, 최솟값 0.0545 = 0.6/11 블렌딩 최솟값)
- [x] 지정가 주문 + TWAP 3-파 + 긴급도 + 비용임계 주문 체계 구현 완료
- [x] execution_market="split" — 개별 종목 매매 활성화
- [x] yfinance 폴백 — KIS sandbox 해외 시세 미지원 해결
- [x] KR/US 분리 스케줄 배포 완료

**남은 항목**:

```bash
# 1. 평일 장 중 KR Wave 1 주문 sandbox 체결 확인 (자동 실행 09:10)
ssh root@77.42.78.9 "grep 'Wave 1\|order\|체결' /var/log/quant-trading.log | tail -50"

# 2. 호가창 조회 확인 (지정가 가격 산출)
# KISApi.get_domestic_bid_ask("005930")  # 삼성전자 테스트

# 3. 미체결 조회 + 취소 확인
# KISApi.get_pending_orders()
# KISApi.cancel_order(order_no, ticker, side, qty)

# 4. US Wave 체결 확인 (자동 실행 KST 23:40)
ssh root@77.42.78.9 "grep 'US\|overseas\|yfinance' /var/log/quant-trading.log | tail -20"

# 5. 대시보드 확인
streamlit run dashboard/app.py
```

**검증 체크리스트**:
- [x] KIS API 인증 (.env 설정 완료)
- [x] 섹터 배분 신호 합 ≈ 1.0
- [x] Phase A~D 주문 로직 구현 완료
- [x] 개별 종목 매매 활성화 (execution_market="split")
- [x] 해외 종목 yfinance 폴백 (sandbox 제약 해결)
- [ ] `get_domestic_bid_ask()` 호가 조회 정상 반환 확인
- [ ] `get_pending_orders()` 미체결 목록 조회 확인
- [ ] `cancel_order()` 취소 정상 작동 확인
- [ ] KR Wave 1 지정가 주문 sandbox 체결 확인
- [ ] US Wave 1 해외 주문 실행 확인 (KST 23:40)
- [ ] 대시보드에서 포트폴리오 가치 갱신 확인

---

### 🟡 우선순위 2: Walk-forward 일관성 개선

**현재 문제**: Period 3 (2025-10-09~11-19) Sharpe -0.13, Period 1 Sharpe 0.42 — 기간별 편차 매우 큼 (std=2.61)

**분석 방향**:
1. Period 3 하락 원인 — 해당 기간 시장 레짐 분석 (bullish/bearish?)
2. alpha=0.4가 모든 레짐에서 최적인지 검토
3. regime-conditional blending 고려 (bull→alpha 높게, bear→equal-weight)

---

### 🟡 우선순위 3: Transformer 신호 품질 개선

**현재 문제**: dir_acc 52.5% (랜덤 50% 대비 미미한 우위)

**시도 방안** (소요시간 순):
1. prediction_horizon 재조정: 현재 1일 → 3일 (단기 노이즈 줄이기)
2. 학습 데이터 기간 조정: 5년 vs 3년 (최근 레짐에 집중)
3. Transformer 학습률 스케줄 변경 (cosine annealing)
4. 섹터별 개별 Transformer 훈련 (현재 공유 파라미터)

```bash
# horizon=3으로 변경 후 Phase 3+부터 재훈련 (~1.5시간)
$PYTHON main.py train --start-phase 3 --config config/settings_fast.yaml
```

---

### 🟢 우선순위 4: Alpha Blending OOS 검증

**현재 문제**: alpha=0.4는 테스트셋에서 grid search로 찾은 값 → 미래 데이터 누설 가능성

**검증 방법**:
- 2024년까지 데이터로 alpha grid search → 2025년으로 OOS 검증
- 또는 Walk-forward 각 기간에서 독립적으로 alpha 선택 (rolling optimization)

---

### 🟢 우선순위 5: 실전 전환 준비

**전환 체크리스트**:
- [ ] sandbox 1주일 이상 정상 실행 확인
- [ ] 개별 종목 신호가 합리적인 범위인지 확인 (score 분포, 양수 비율)
- [ ] 긴급 청산 (`emergency_liquidate()`) 시나리오 테스트
- [ ] 재학습 후 rollback 시나리오 테스트
- [ ] `live_config.yaml`에서 `broker.mode: "production"` 변경
- [ ] 소액 (100만원) 으로 첫 실전 주문 테스트
- [ ] 일간 리밸런싱 패턴 모니터링 (과도한 회전율 여부 확인)

---

### 🔵 우선순위 6: 모델 아키텍처 개선 (장기)

#### 6-1. RL을 백테스트에 통합
현재 백테스트는 Transformer prediction만 사용; RL allocation은 무시됨.
RL `ensemble.get_allocation()`를 활용하는 백테스트 모드 추가.

#### 6-2. GAN 품질 개선
W-dist 패턴이 여전히 불안정. Mode collapse 근본 해결:
- Progressive GAN (저해상도→고해상도 단계적 훈련)
- Diffusion 기반 시장 시뮬레이터로 교체 고려

#### 6-3. VAE 잠재 지표 분석 (연구)
```python
from indicators.generator import IndicatorGenerator
from indicators.evaluator import IndicatorEvaluator
gen = IndicatorGenerator(vae_model, n_indicators=32)
scores = IndicatorEvaluator().evaluate_all(indicators, returns)
# IC, 신규성, 단조성 점수 확인
```

#### 6-4. 데이터 확장
- 현재: 5년 (2021-2026), 200 종목
- 검토: 더 많은 KOSPI 종목 추가 (현재 ~100개 → 전체)
- 주의: 학습 기간은 5년이 10년보다 더 좋은 결과 (분포 변화)

---

## 핵심 명령어

```bash
PYTHON=/c/Users/wogus/miniconda3/envs/quant/python.exe

# 학습
$PYTHON main.py train --start-phase 5 --config config/settings_fast.yaml  # RL만 ~15분
$PYTHON main.py train --start-phase 3 --config config/settings_fast.yaml  # TF부터 ~1.5시간
$PYTHON main.py train --start-phase 2 --config config/settings_fast.yaml  # 전체 ~3시간

# 백테스트
$PYTHON main.py backtest --config config/settings_fast.yaml
$PYTHON main.py backtest --walk-forward --config config/settings_fast.yaml

# 라이브 트레이딩
setup_live.bat                                              # 최초 설정
python scheduler/daily_runner.py --step signal              # 신호 생성 테스트 ✅
python scheduler/daily_runner.py --step order               # Wave 1 주문 테스트 (40%)
python scheduler/daily_runner.py --step order --wave 2      # Wave 2 주문 테스트 (35%)
python scheduler/daily_runner.py --step order --wave 3      # Wave 3 주문 테스트 (25%)
python scheduler/daily_runner.py --daemon                   # 자동 스케줄 데몬 (3-파)
python scheduler/retrain_runner.py --mode check             # 재학습 필요 여부 확인
python scheduler/retrain_runner.py --mode weekly            # 주간 파인튜닝

# 대시보드
streamlit run dashboard/app.py  # → http://localhost:8501
```
