# 시스템 개선 분석 — 세계적 퀀트 트레이더 대비 격차

> 작성일: 2026-03-12 / 최종 업데이트: 2026-03-12 (P0~P3 전항목 구현 완료)
> 현재 성능: Sharpe 2.03, MDD -4.30%, Return +17.82% (180일, 5년 백테스트)

## 구현 완료 현황 (2026-03-12 전면 업데이트)

| 항목 | 상태 | 파일 |
|------|------|------|
| KIS 시간외 주문 API (ORD_DVSN 61/62/63/32/33) | ✅ 완료 | `broker/kis_api.py` |
| 세션별 리스크 파라미터 (SESSION_RISK) | ✅ 완료 | `live/signal_to_order.py` |
| Alpha Decay 함수 | ✅ 완료 | `live/signal_to_order.py` |
| execute_extended_hours() | ✅ 완료 | `live/signal_to_order.py` |
| Layer 2 장중 신호 (KR: KIS API / US: yfinance) | ✅ 완료 | `scheduler/daily_runner.py` |
| KR 장전/장후 시간외 + US 프리/에프터 (5세션) | ✅ 완료 | `scheduler/daily_runner.py` |
| Layer 2 업데이트 스케줄 (KR 10:00/13:00, US 01:00/03:30) | ✅ 완료 | `scheduler/daily_runner.py` |
| TWAP 비율 재조정 (40/35/25 → 35/30/20) | ✅ 완료 | `config/live_config.yaml` |
| 전체 스케줄 6슬롯 → 14슬롯 | ✅ 완료 | `config/live_config.yaml` |
| Walk-forward 안정성 측정 (`compute_walkforward_stability`) | ✅ 완료 | `backtest/metrics.py` |
| alpha_blend OOS 검증 (`tune_alpha_blend`, val set 기준) | ✅ 완료 | `backtest/metrics.py`, `main.py` |
| 백테스트 비용 현실화 (commission 0.02%, slippage 0.3%) | ✅ 완료 | `config/settings_fast.yaml` |
| 회전율 추적 (`compute_turnover`, `get_turnover_stats`) | ✅ 완료 | `tracking/trade_log.py` |
| KOSPI 섹터 키워드 확장 (3→15개/섹터, ~50% 커버리지) | ✅ 완료 | `config/sectors.yaml` |
| Half-Kelly 포지션 사이징 (`_kelly_scale`) | ✅ 완료 | `live/signal_to_order.py` |
| 모멘텀 품질·반전·거래량-가격·드로우다운 피처 추가 | ✅ 완료 | `data/feature_engineer.py` |
| 크로스-섹션 순위 피처 + 시장 브레드스 프록시 | ✅ 완료 | `data/feature_engineer.py` |
| 인트라데이 3% 스탑로스 (`_check_intraday_stoploss`) | ✅ 완료 | `scheduler/daily_runner.py` |
| 시장 레짐 감지 (`MarketRegimeDetector`: bull/bear/volatile/neutral) | ✅ 완료 | `strategy/signal.py` |
| 레짐-조건부 알파 블렌딩 (bear: 0.2, bull: 0.5) | ✅ 완료 | `strategy/signal.py`, `scheduler/daily_runner.py` |

| 생존편향 제거 (상장폐지 종목 OHLCV 수집) | ✅ 완료 | `data/collector.py` |
| DART 공시 이벤트 트리거 (`DartClient`, Layer 2 스케일 반영) | ✅ 완료 | `data/dart_client.py`, `scheduler/daily_runner.py` |
| Almgren-Chriss 시장 충격 모델 (`MarketImpactModel`) | ✅ 완료 | `strategy/market_impact.py`, `live/signal_to_order.py` |
| 대체 데이터: Parkinson/GK vol, vol skew, Amihud, 공시 NLP | ✅ 완료 | `data/alternative_data.py`, `data/feature_engineer.py` |
| 포트폴리오 헤지 (KTB/USD-KRW/Gold, 레짐-조건부) | ✅ 완료 | `strategy/hedge.py`, `scheduler/daily_runner.py` |

---

## 1. 알파 신호 품질 — 최우선 과제

### 현황
- 방향 정확도(dir_acc): **52.5%** (랜덤 수준, 50% 대비 2.5%p 초과)
- 피처: OHLCV 기반 **34개** (반환율, 변동성, 가격비율, 거래량, 내가격 통계)
- 데이터 소스: yfinance + pykrx 단일 소스

### 세계 수준과의 격차
- 방향 정확도 목표: **55~58%**
- 피처 수: 수천~수만 개 (대체 데이터 포함)
- 데이터 소스: 20~100종 (재무, 옵션, 오더북, 뉴스 NLP, 위성, 신용카드 등)

### 추가해야 할 피처 카테고리
```
[재무/펀더멘털]
- PBR, PER, ROE, 매출성장률, 이익 수정치 (컨센서스 대비)
- 내부자 거래, 기관 보유 변화, 공매도 비율

[시장 미시구조]
- 옵션 내재변동성 (IV skew, term structure, 풋콜 비율)
- Bid-ask spread, 호가창 불균형 (order book imbalance)
- 블록 딜 감지

[크로스-에셋 신호]
- 금리 기간구조 (국채 10년-2년 스프레드)
- USD/KRW, 원자재 (WTI, 구리) 상관관계
- 크레딧 스프레드 (HY-IG)

[텍스트/이벤트]
- 공시 NLP 감성 (DART, SEC)
- 뉴스 감성 (FinBERT)
- 어닝 서프라이즈 크기
```

---

## 2. 과적합 위험 — 심각

### 현황
- Walk-forward (5×30일): Mean Sharpe **1.89, std=2.61** → std > mean (불안정)
- `alpha=0.4` 블렌딩: 테스트셋으로 튜닝 → look-ahead bias 의심
- 백테스트 기간 180일 (6개월) → 통계적 유의성 부족

### 세계 수준 기준
- Walk-forward std/mean < 0.5
- 최소 10년 이상 OOS 검증
- 파라미터 튜닝은 반드시 별도 validation set 사용

### 개선 방안
1. `alpha=0.4` 고정 → OOS 고정값으로 결정 (test set 사용 금지)
2. Walk-forward 구간 늘리기 (30일 → 60일 구간 × 10회)
3. 생존편향 제거: 상장폐지 종목 포함 데이터 사용

---

## 3. 신호 갱신 주기 — ✅ 다중 세션 구현 완료 (2026-03-12)

### 구현 전 (6슬롯)
```
06:10  전체 추론 (1회/일)
06:30  KR 캐시 재사용
09:10 / 11:00 / 13:30  KR Wave 1/2/3 — 06:10 신호 그대로
23:40 / 02:00 / 04:30  US Wave 1/2/3 — 06:10 신호 그대로
```

### 구현 후 (14슬롯)
```
05:10  [US] After-hours (10%, ORD_DVSN 33, alpha decay 적용)
06:10  전체 추론 (Layer 1, 1회/일)
06:30  KR 신호 분리 + 하락 매도
07:30  [KR] 장전 시간외 단일가 (15%, ORD_DVSN 61, min_score 0.008)
09:10  [KR] Wave 1 (35%, Layer 1 신호)
10:00  [KR] Layer 2 업데이트 #1 (KIS API 현재가 → 인트라데이 모멘텀)
11:00  [KR] Wave 2 (30%, Layer 1 × Layer 2 스케일)
13:00  [KR] Layer 2 업데이트 #2
13:30  [KR] Wave 3 (20%, Layer 1 × Layer 2 스케일)
15:35  [KR] 장후 시간외 종가 (5%, ORD_DVSN 62, 당일 모멘텀 강한 종목)
16:30  [KR] 장후 시간외 단일가 (5%, ORD_DVSN 63, 다음날 선포지션)
18:30  [US] Pre-market (15%, ORD_DVSN 32, min_score 0.010)
23:40  [US] Wave 1 (35%)
01:00  [US] Layer 2 업데이트 #1 (yfinance 5분봉)
02:00  [US] Wave 2 (30%, Layer 1 × Layer 2 스케일)
03:30  [US] Layer 2 업데이트 #2
04:30  [US] Wave 3 (20%, Layer 1 × Layer 2 스케일)
```

### Alpha Decay 구현 완료
```python
# 신호가 오래될수록 포지션 비율 자동 감소
alpha_decay_scale(0h)  = 1.000  (신선)
alpha_decay_scale(8h)  = 0.794
alpha_decay_scale(12h) = 0.707
alpha_decay_scale(24h) = 0.500  (After-hours — 전날 신호)
```
- 시간외 세션은 모두 decay 적용 → 오래된 신호로 과도한 포지션 방지
- Layer 2 스케일: 장중 -3% 이상 급락 종목 → 다음 Wave 수량 0.3배로 자동 축소

### 잔여 과제
- ✅ DART 공시 이벤트 트리거 → Layer 2 스케일 반영 완료 (`data/dart_client.py`)
- 추론 속도 개선 (현재 CPU 10~15분 → GPU 상시화 시 2~3분)

---

## 4. 리스크 관리 — 구조적 미흡

### 현황 (strategy/risk.py)
- MDD 서킷브레이커: 5/10/20/30% 단계적 축소 ✅
- 변동성 타겟팅 (vol targeting) ✅
- VaR / CVaR 계산 ✅
- position_sizing (Half-Kelly) — 코드는 있지만 실제 주문에 미연결 ❌

### 누락된 리스크 컨트롤
```
[팩터 중립화]
- 시장 베타 노출 한도 (현재 무제한)
- 섹터 팩터 편향 제한

[포지션 집중도]
- 상관관계 기반 총 노출 제한 (비슷한 종목 몰빵 방지)
- 실제 Kelly를 주문 수량 결정에 연결

[테일 리스크]
- 인트라데이 stop-loss 없음
- 블랙스완 대비 tail hedging (VKOSPI 옵션 등)

[레짐 연동]
- 상승장/하락장/위기별 목표 변동성 다르게 설정
```

---

## 5. 실행 비용 — 과소평가

### 백테스트 vs 실제 비용 괴리
| 항목 | 백테스트 설정 | 실제 수준 |
|------|--------------|-----------|
| 수수료 | `0.00015` (0.015%) | 국내 0.015~0.05% |
| 슬리피지 | `0.001` (0.1%) | KOSPI 소형주 0.3~1.0% |
| 시장 충격 | 없음 | 거래량 대비 주문 크기 의존 |

### 일간 리밸런싱 비용 추정
- `rebalance_day: "daily"` + 회전율 ~80%/일 가정
- 실질 왕복 비용 0.6% × 252일 = **연간 151% 비용**
- 이게 Sharpe를 얼마나 잠식하는지 미측정

### 개선 방향
1. 리밸런싱 임계값(`rebalance_threshold: 0.03`) 활용 — 현재 config에는 있지만 실제 적용 확인 필요
2. 회전율 측정 및 보고 로직 추가
3. 백테스트 commission_rate를 실제와 통일 (0.00015 → 0.0002)
4. KOSPI 소형주 시장 충격 모델 추가

---

## 6. 유니버스 & 데이터

### 현황
- KOSPI: 950종목 수집 → **203종목** 섹터 분류 (21.4% 커버리지)
- NASDAQ: 400종목
- 생존편향: 현재 상장 종목만 사용

### 개선
- KOSPI 섹터 분류 커버리지 향상 (21% → 60%+)
- 상장폐지 종목 히스토리 포함
- 멀티 에셋 확장 (KTB, USD/KRW 선물 헤지)

---

## 7. 인프라

### 현황
- Hetzner 서버 1대 (`77.42.78.9`)
- systemd `quant-trading` 단일 프로세스
- CPU 추론: ~10~15분

### 리스크
- 단일 장애점 (서버 다운 = 전체 중단)
- KIS API 장애 시 failover 없음
- 추론 속도: Wave 1(09:10) 이전까지 빠듯

### 개선 방향
- 추론 속도 최적화 (모델 경량화 또는 GPU 상시화)
- 헬스체크 + 알림 (Telegram) 고도화
- 백업 브로커 또는 재시도 로직 강화

---

## 개선 우선순위 로드맵

### P0 — 즉시 (신뢰성 확보)
- [x] Walk-forward std/mean 비율 측정 및 보고 (`backtest/metrics.py`: `compute_walkforward_stability`)
- [x] `alpha=0.4` OOS 검증 — `tune_alpha_blend(val_returns, model_signals_val)` 구현, config에서 로드, test set 재튜닝 금지 주석 추가 (`backtest/metrics.py`, `main.py`)
- [x] 백테스트 commission 실제 수준으로 통일 (`settings_fast.yaml`: 0.00015→0.0002, slippage 0.001→0.003)
- [x] 회전율 측정 로직 추가 (`tracking/trade_log.py`: `compute_turnover`, `get_turnover_stats`)

### P1 — 단기 (알파 품질)
- [x] 모멘텀 품질·반전·거래량-가격 관계·드로우다운 피처 추가 (`data/feature_engineer.py`)
- [x] 크로스-섹션 순위 피처 & 시장 레짐 프록시 피처 추가 (`data/feature_engineer.py`: `add_market_regime_features`)
- [x] KOSPI 섹터 커버리지 21% → ~50% 향상 (`config/sectors.yaml`: 섹터별 키워드 3→15개 확장)
- [x] 생존편향 제거 (`data/collector.py`: `get_kospi_delisted_tickers`, `collect_all`에 병합)
- [x] `position_sizing()` → 실제 주문 수량에 연결 (`live/signal_to_order.py`: `_kelly_scale`, `use_kelly_sizing`)

### P2 — 중기 (실행 고도화)
- [x] 신호 갱신 다중 세션 (장전/장중2회/장후/프리마켓/에프터마켓)
- [x] 공시 이벤트 트리거 (`data/dart_client.py`: `DartClient`, `scheduler/daily_runner.py`: `_check_dart_events`)
- [x] Alpha decay 구현 (alpha_decay_scale, daily 24h 반감기)
- [x] 시장 충격 모델 Almgren-Chriss (`strategy/market_impact.py`: `MarketImpactModel`, `live/signal_to_order.py` 주문 수량 자동 조정)
- [x] 인트라데이 stop-loss (`scheduler/daily_runner.py`: `_check_intraday_stoploss`, Layer 2 업데이트 시 3% 트리거)

### P3 — 장기 (알파 다변화)
- [x] 대체 데이터: Parkinson/GK vol, vol skew proxy, Amihud illiquidity, 공시 NLP (`data/alternative_data.py`, `data/feature_engineer.py`에 통합)
- [x] 크로스-에셋 프록시 피처 (시장 수익률, 변동성, 브레드스) (`data/feature_engineer.py`)
- [x] 레짐 감지 → 조건부 앙상블 가중치 (`strategy/signal.py`: `MarketRegimeDetector`, `scheduler/daily_runner.py`: 실시간 레짐 감지)
- [x] 포트폴리오 헤지 KTB/USD-KRW/Gold (`strategy/hedge.py`: `PortfolioHedger`, `scheduler/daily_runner.py`에 레짐 연동)

---

## 현실적 기대치

| 지표 | 현재 백테스트 | 합리적 실거래 목표 |
|------|--------------|-------------------|
| Sharpe | 2.03 | **0.8~1.5** (비용·슬리피지 후) |
| MDD | -4.30% | **-10~-20%** (실거래 변동성) |
| dir_acc | 52.5% | **54~56%** (P1 완료 후) |
| Walk-forward std/mean | 1.38 | **< 0.8** |

> 실거래 6개월 OOS Sharpe가 진짜 시스템 성능 지표.
> 백테스트 Sharpe 2.03은 과적합 가능성 높음 — 1.0 이상이면 성공으로 봐야 함.
