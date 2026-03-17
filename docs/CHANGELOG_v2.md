# Changelog v2.0 — 세계적 퀀트 기준 전면 개선

> 작업 기간: 2026-03-12 ~ 2026-03-13
> 커밋: `17bdce5` (main)
> 서버 배포: 2026-03-13 02:01 KST (`77.42.78.9`)

---

## 변경 규모

| 구분 | 수치 |
|------|------|
| 수정/생성 파일 | 18개 |
| 추가 코드 | +2,015줄 |
| 삭제 코드 | -384줄 |
| 신규 모듈 | 4개 |
| 구현 항목 (P0~P3) | 25개 전항목 완료 |

---

## 성능 비교

| 지표 | Before (34 피처) | After (72 피처) |
|------|-----------------|-----------------|
| 피처 수 | 34 | **72** |
| KOSPI 섹터 분류 | 21.4% (203/950) | **30.5%** (287/941) |
| Sharpe Ratio | 2.03 | **1.98** |
| Total Return | +17.82% | **+16.78%** |
| Max Drawdown | -4.30% | **-4.54%** |
| Sortino Ratio | - | **3.17** |
| Win Rate | - | **56.7%** |
| Positive Months | - | **88.9%** (8/9) |
| Calmar Ratio | - | **5.34** |
| 스케줄 슬롯 | 6 | **14** |

> Sharpe 소폭 하락(-0.05)은 비용 현실화(commission 0.015%→0.02%, slippage 0.1%→0.3%)에 기인.
> VAE에서 NaN loss 발생 — 신규 피처 스케일링 개선 시 추가 상승 여지 있음.

---

## 1. 신규 모듈 (4개 파일 생성)

### `data/dart_client.py` — DART 공시 이벤트 수집
- DART OpenAPI 연동 (환경변수 `DART_API_KEY` 필요)
- `DartClient.get_recent_disclosures()`: 최근 N일 공시 조회
- `DartClient.score_disclosure()`: 15개 키워드 기반 이벤트 스코어 (-1.0~+1.0)
- `DartClient.get_event_signals()`: 종목별 공시 신호 생성
- Layer 2 장중 업데이트 시 자동 반영 (`_check_dart_events`)

### `data/alternative_data.py` — 대체 데이터 피처
- `AlternativeDataFeatures`: OHLCV 기반 17개 프록시 피처
  - Parkinson Volatility (5d, 20d) — IV 프록시
  - Garman-Klass Volatility (5d, 20d) — 고효율 IV 프록시
  - Vol Skew Proxy (20d, 60d) — 풋-콜 비율 대용
  - Return Kurtosis (20d, 60d) — 테일 리스크
  - Volume Z-score (20d) — 거래량 레짐
  - Amihud Illiquidity (20d) — 유동성 역수
  - Vol Premium (20d) — Realized vs Implied 스프레드
  - Overnight/Intraday Return + Ratio — 정보 비대칭 프록시
- `DisclosureNLP`: 공시 키워드 기반 감성 분석 (20 긍정 / 24 부정 키워드)

### `strategy/market_impact.py` — Almgren-Chriss 시장 충격 모델
- `MarketImpactModel.estimate_impact()`: 영구/일시 충격 비용 추정
- `MarketImpactModel.adjust_order_size()`: 충격 0.5% 이하로 주문 수량 자동 제한
- `MarketImpactModel.get_volume_tier()`: 거래량 기반 large/mid/small 분류
- 시장별 6개 유동성 파라미터 (KOSPI/NASDAQ x large/mid/small)
- Square-root law (alpha=0.5) 비선형 충격 모델

### `strategy/hedge.py` — 포트폴리오 헤지 프레임워크
- `PortfolioHedger.compute_hedge_signal()`: 레짐-조건부 헤지 비율 (0~30%)
  - bull: 5%, neutral: 10%, volatile: 20%, bear: 30%
- `PortfolioHedger.get_hedge_orders()`: 헤지 주문 목록 생성
- 3개 헤지 수단: KTB 10Y (TLT proxy), USD/KRW (UUP proxy), Gold (GLD proxy)
- 포트폴리오 베타 계산 + 고변동성/연속 하락 추가 헤지
- 5% 리밸런싱 임계값으로 불필요한 회전 방지

---

## 2. 수정된 기존 파일 (14개)

### `data/feature_engineer.py`
- 모멘텀 품질: `pos_day_ratio`, `trend_strength`
- 반전 신호: `reversal_1w`, `reversal_momentum_spread`, `zscore_1d`
- 거래량-가격: `obv_direction`, `vol_price_divergence`
- 드로우다운: `drawdown_20d`, `drawdown_60d`
- 크로스-섹션 순위: 5개 `{feature}_rank` 컬럼
- 시장 레짐 프록시: `market_return`, `market_vol`, `breadth`, `relative_return`
- 대체 데이터 통합: `AlternativeDataFeatures.compute_all()` 자동 호출
- **합계: 34 → 72 피처**

### `data/collector.py`
- `get_kospi_delisted_tickers()`: pykrx 상장폐지종목검색 API + 연도별 비교 폴백
- `collect_all()`: 폐지 종목 OHLCV 수집 → KOSPI 데이터에 병합, `is_delisted` 플래그
- 생존편향 제거를 위한 데이터 파이프라인 완성

### `data/sector_classifier.py`
- `classify_kospi()`: best-match 로직 (가장 많은 키워드 매칭 섹터 할당)
- 기존 단순 first-match → 다중 매칭 비교로 정확도 향상

### `config/sectors.yaml`
- GICS 11개 섹터 키워드: 3~5개 → 11~20개 확장 (총 186개)
- 대형주 회사명 포함 (삼성전자, SK하이닉스, 현대차 등)
- KOSPI 분류율: 21.4% → 30.5%

### `config/settings_fast.yaml`
- `commission_rate`: 0.00015 → **0.0002** (실제 수준)
- `slippage_rate`: 0.001 → **0.003** (KOSPI 소형주 반영)
- `alpha_blend: 0.4` 명시 + val set 튜닝 주석

### `backtest/metrics.py`
- `compute_walkforward_stability()`: N분할 윈도우별 Sharpe → std/mean 안정성 비율
- `tune_alpha_blend()`: val set 그리드 서치 (0.0~1.0, 0.05 간격), 최적 alpha 권장

### `main.py`
- `alpha = 0.4` 하드코딩 → `config["backtest"].get("alpha_blend", 0.4)` 동적 로드

### `tracking/trade_log.py`
- `daily_turnover` 컬럼 추가 (ALTER TABLE 자동 마이그레이션)
- `log_daily_performance()`: `turnover` 파라미터 추가
- `compute_turnover(date_str)`: 일별 회전율 계산
- `get_turnover_stats(days=30)`: 기간별 회전율 통계

### `live/signal_to_order.py`
- Half-Kelly 포지션 사이징: `_kelly_scale(ticker, score, market)`
  - yfinance 20일 변동성 조회 + 캐싱
  - 정규화된 score / volatility → 0.3x~1.5x 스케일
- Almgren-Chriss 통합: `MarketImpactModel` 초기화 + `_compute_target_positions`에서 주문 수량 자동 조정

### `scheduler/daily_runner.py`
- `MarketRegimeDetector` 통합: `_daemon_us_signal`에서 레짐 감지
- 레짐-조건부 알파 블렌딩: bear=0.2, neutral=0.4, bull=0.5
- `_check_intraday_stoploss(stoploss_pct=0.03, market_filter)`: 3% 스탑로스
- DART 공시 이벤트: `_check_dart_events()` → Layer 2 스케일 반영
- `PortfolioHedger` 통합: 레짐 감지 후 헤지 신호 계산 + 로깅

### `strategy/signal.py`
- `MarketRegimeDetector`: 20일 모멘텀 + 변동성 백분위 기반 4-레짐 분류
  - bull (>3%↑, 정상 vol), bear (<-3%↓), volatile (상위 70% vol), neutral
- `REGIME_SCALE`: bull=1.2, bear=0.6, volatile=0.8, neutral=1.0
- `REGIME_ALPHA`: bull=0.5, bear=0.2, volatile=0.3, neutral=0.4

### `docs/IMPROVEMENT_ANALYSIS.md`
- P0~P3 전항목 ✅ 완료 표시
- 구현 현황 테이블 25개 항목으로 확장
- 잔여 과제 업데이트

---

## 3. 삭제된 파일

| 파일 | 사유 |
|------|------|
| `NEXT_STEPS.md` | 내용이 `docs/IMPROVEMENT_ANALYSIS.md`로 통합 |
| `PROJECT_STATUS.md` | 내용이 `docs/IMPROVEMENT_ANALYSIS.md`로 통합 |

---

## 4. 14슬롯 트레이딩 스케줄 (KST)

```
05:10  [US] After-hours (10%, alpha decay 적용)
06:00  데이터 수집
06:10  [US] 신호 생성 + 레짐 감지 + 헤지 신호 + 하락 매도
06:30  [KR] 신호 분리 + 하락 매도
07:30  [KR] 장전 시간외 단일가 (15%)
09:10  [KR] Wave 1 (50%)
10:00  [KR] Layer 2 #1 (인트라데이 모멘텀 + DART 공시 + 스탑로스 3%)
11:00  [KR] Wave 2 (30%, Layer 1 x Layer 2)
13:00  [KR] Layer 2 #2 (+ DART + 스탑로스)
13:30  [KR] Wave 3 (20%)
15:35  [KR] 장후 시간외 종가 (5%)
16:00  EOD 기록
16:30  [KR] 장후 시간외 단일가 (5%)
18:30  [US] Pre-market (15%)
23:40  [US] Wave 1 (50%)
01:00  [US] Layer 2 #1 (+ 스탑로스)
02:00  [US] Wave 2 (30%)
03:30  [US] Layer 2 #2 (+ 스탑로스)
04:30  [US] Wave 3 (20%)
```

---

## 5. 주문 실행 플로우 (개선 후)

```
신호 생성 (Layer 1)
  → 레짐 감지 (bull/bear/volatile/neutral)
    → 레짐-조건부 알파 블렌딩 (0.2~0.5)
      → Half-Kelly 포지션 사이징 (0.3x~1.5x)
        → Almgren-Chriss 시장 충격 조정 (max 0.5%)
          → TWAP 분할 주문 (Wave 1/2/3)

장중 업데이트 (Layer 2)
  → KIS/yfinance 현재가 조회
    → 인트라데이 모멘텀 스케일 (0.3~1.5)
      → DART 공시 이벤트 스케일 (0.3~1.3)
        → 3% 스탑로스 체크
          → 다음 Wave 수량에 반영
```

---

## 6. 활성화 필요 항목

| 기능 | 필요 작업 | 없어도 작동 여부 |
|------|----------|----------------|
| DART 공시 이벤트 | `.env`에 `DART_API_KEY=xxx` 추가 | O (비활성 상태로 스킵) |
| 포트폴리오 헤지 실행 | 별도 선물/FX 브로커 API 연동 | O (신호만 로깅, 미체결) |
| 생존편향 제거 데이터 | `main.py train` 재실행 (phase 1) | O (기존 데이터로 학습) |

---

## 7. 알려진 이슈

| 이슈 | 심각도 | 상태 | 비고 |
|------|--------|------|------|
| VAE loss NaN | ~~중~~ | **✅ 수정됨** | log 연산 clip, log_var clamp, NaN batch skip |
| Transformer dir_acc 48.9% | 중 | 미해결 | VAE NaN 수정 후 재학습 필요 |
| 모델 vs 베이스라인 동일 수치 | 저 | 미해결 | VAE 수정 후 alpha 검증 필요 |
| inference_pipeline n_features=34 | ~~고~~ | **✅ 수정됨** | 하드코딩 → processed data에서 자동 감지 |
| TWAP 비율 config/code 불일치 | ~~중~~ | **✅ 수정됨** | 코드를 config과 동기화 (35/30/20) |
| score_cost_threshold 기본값 불일치 | ~~중~~ | **✅ 수정됨** | 0.05 → 0.001 (config과 일치) |
| 라이브 트레이딩 유휴 현금 과다 | ~~고~~ | **✅ 수정됨** | 비중 재배분, TWAP 50%, Kelly 하한 0.5 |
| KOSPI 섹터 30.5% (목표 50%+) | 저 | 미해결 | 키워드 추가 확장으로 개선 가능 |
| 추론 속도 CPU 10~15분 | 저 | 미해결 | GPU 상시화 시 2~3분 가능 |

---

## 8. v2.1 수정 사항 (2026-03-16)

### VAE NaN 근본 원인 해결
- `feature_engineer.py`: `np.log()` 연산에 `.clip(lower=1e-8)` 적용, return 값 `-1~10` 클리핑
- `feature_engineer.py`: inf 값 감지 및 제거 로직 추가 (dropna 전 inf→NaN 변환)
- `alternative_data.py`: 기존 보호 로직 확인 (clip/1e-8 이미 적용)
- `model.py (VAE)`: `log_var`를 `[-10, 10]` 범위로 clamp → KL loss exp() 오버플로 방지
- `model.py (VAE)`: forward()에서 NaN/inf 입력 자동 정리 (`torch.nan_to_num`)
- `trainer.py (VAE)`: NaN loss 배치 스킵 + optimizer 리셋, NaN 지속 시 조기 종료

### 데이터 파이프라인 안정성
- `train_pipeline.py`: 정규화 후 inf/NaN 검증 + 극단값 경고 (abs_max > 100)
- `dataset.py`: 텐서 변환 전 `np.nan_to_num()` 안전망 추가

### 차원 불일치 해결
- `inference_pipeline.py`: 하드코딩 `n_features=34` → processed_data.parquet에서 자동 감지

### 설정 동기화
- `signal_to_order.py`: TWAP 비율 40/35/25 → 35/30/20 (live_config.yaml과 일치)
- `signal_to_order.py`: `score_cost_threshold` 기본값 0.05 → 0.001 (config과 일치)

---

## 9. v2.2 전략 개선 (2026-03-16)

> 수익률 제로-알파 문제 해결을 위한 6대 전략 개선

### 9.1 수급 데이터 파이프라인 (`data/flow_data.py` 신규)
- `FlowDataCollector`: pykrx `get_market_trading_value_by_date()` 활용
- 외국인/기관 순매수 금액, 순매수 비율, 누적 순매수, 수급 모멘텀 피처
- `FlowFeatureEngineer`: 15개 수급 피처 생성 (5/10/20일 윈도우)
- `feature_engineer.py`: flow 데이터 존재 시 자동 피처 생성 연동
- `train_pipeline.py`: Phase 1에서 수급 데이터 수집 → parquet 저장 → 머지

### 9.2 유니버스 집중 (400종목 → KOSPI200 + NQ100)
- `collector.py`: `get_kospi_tickers(kospi200_only=True)` 옵션 추가
- pykrx `get_index_portfolio_deposit_file("1028")` 로 KOSPI200 구성종목 조회
- `settings_fast.yaml`: `max_tickers: 200`, `kospi200_only: true`
- 노이즈 제거 + 유동성 확보 → 실거래 가능한 종목만 집중

### 9.3 Cross-sectional 상대 예측
- `dataset.py`: `cross_sectional_target=True` 옵션 추가
- 절대 수익률 대신 시장 대비 초과수익률(alpha) 예측으로 전환
- 타겟: `log(stock_return) - log(market_return)` (시장 beta 제거)
- `settings_fast.yaml`: `cross_sectional_target: true`

### 9.4 포트폴리오 최적화 업그레이드
- `backtest/engine.py`: signal_weighted → **risk_parity + signal_tilt 블렌딩**
  - 60% risk-parity base + 40% signal tilt (충분한 히스토리 있을 때)
  - 리스크 균등 배분 + 모델 시그널 방향성 반영

### 9.5 실행 알파 (Execution Alpha)
- `backtest/engine.py`: `rebalance_threshold=0.05` 드리프트 기반 리밸런싱
  - 포지션 변화가 5% 미만이면 리밸런싱 스킵 → 불필요 거래비용 절감
- `settings_fast.yaml`: `rebalance_threshold: 0.05`

### 9.6 LightGBM 크로스-섹셔널 벤치마크 (`models/lgbm_baseline.py` 신규)
- `LGBMBaseline`: LambdaRank / Regression 기반 크로스-섹셔널 모델
- `backtest_cross_sectional()`: Walk-forward Long/Short 포트폴리오 백테스트
- Feature importance 분석으로 실효 피처 식별
- `main.py`: `python main.py lgbm-baseline` 커맨드 추가
- 딥러닝 앙상블이 이 벤치마크를 못 이기면 모델 단순화 근거

| 파일 | 변경 |
|------|------|
| `data/flow_data.py` | **신규** — 수급 수집 + 피처 엔지니어링 |
| `models/lgbm_baseline.py` | **신규** — LightGBM 크로스-섹셔널 벤치마크 |
| `data/collector.py` | KOSPI200 필터 + **KOSDAQ150 수집** 추가 |
| `data/sector_classifier.py` | `classify_kosdaq()` 메서드 추가 |
| `data/dataset.py` | cross_sectional_target 옵션 |
| `data/feature_engineer.py` | 수급 피처 연동 |
| `backtest/engine.py` | risk_parity 블렌딩 + 드리프트 리밸런싱 |
| `pipeline/train_pipeline.py` | 수급 수집/머지, cross-sectional 전달, KOSDAQ 분류 연동 |
| `main.py` | lgbm-baseline 커맨드, rebalance_threshold |
| `config/settings_fast.yaml` | 유니버스 축소, KOSDAQ 추가, cross-sectional, 수급, 시장별 슬리피지 |
| `config/sectors.yaml` | KOSDAQ 마켓 정의 추가 |
| `config/live_config.yaml` | KOSDAQ 섹터 ETF 맵 추가 |
| `requirements.txt` | lightgbm>=4.3.0 추가 |

### 9.7 KOSDAQ 시장 추가 (KOSDAQ150)
- `collector.py`: `get_kosdaq_tickers(kosdaq150_only=True)` — pykrx `상장종목검색("KSQ")` + KOSDAQ150 지수(`1150`) 필터
- `collect_all()`: `include_kosdaq`, `kosdaq150_only` 파라미터 추가
- `DataCollector`: KOSDAQ 시장 설정 시 자동 수집
- `sector_classifier.py`: `classify_kosdaq()` — KOSPI와 동일한 한국어 키워드 매칭
- `flow_data.py`: KOSDAQ 수급 데이터 수집 지원 (`.KQ` suffix 처리)
- `train_pipeline.py`: KOSDAQ 섹터 분류 연동, 수급 수집 대상 확대
- `settings_fast.yaml`: `include_kosdaq: true`, `kosdaq150_only: true`, 시장별 슬리피지 차등
  - KOSPI: 0.3%, KOSDAQ: 0.5%, NASDAQ: 0.1%
- `sectors.yaml`: KOSDAQ 마켓 정의 (`.KQ` suffix, KRX 거래소)
- `live_config.yaml`: KOSDAQ 섹터 ETF 맵 추가 (개별종목 매매 + ETF 대체 헤지)
- **기대 효과**: 바이오/IT 중심의 KOSDAQ150 → 기관 커버리지 낮아 alpha 기회 풍부, KOSPI 대비 상관관계 낮아 분산 효과

---

## 10. v2.3 유휴 현금(미수금) 과다 문제 수정 (2026-03-17)

> 라이브 트레이딩 시 자본의 30~50%가 미투자 상태로 남는 문제 진단 및 수정

### 10.1 원인 진단

`signal_to_order.py` 분석 결과 8개 지점에서 현금 누수가 복합적으로 발생:

| 원인 | 위치 | 영향도 | 메커니즘 |
|------|------|--------|----------|
| TWAP Wave 미완료 | `TWAP_FRACTIONS` | **최대** | Wave 1이 35%만 집행, 후속 Wave 누락 시 65% 유휴 |
| 비중 클리핑 증발 | `_validate_weights()` | **대** | max_sector_wt 초과분이 다른 섹터에 재배분되지 않음 |
| min_order_amount 필터 | `_compute_target_positions()` | 중 | 약한 신호 섹터 소액 배분 전량 스킵 |
| Half-Kelly 과도 축소 | `_kelly_scale()` | 중 | 약한 신호 → 0.3x까지 축소, 70% 미투자 |
| Phase D 약한 신호 차단 | score_cost_threshold | 중 | 중립~약양수 종목 대량 탈락 |
| sector_top_tickers 부족 | split 모드 | 저 | 모델이 양수 score 종목 적게 생성 |
| cash_buffer 고정 5% | `_validate_weights()` | 저 | 의도적 (정상 동작) |
| market_impact 축소 | `MarketImpactModel` | 저 | 대량 주문만 해당 |

### 10.2 수정 내용

**① `_validate_weights()` — 클리핑 초과분 재배분 (가장 큰 효과)**
- 기존: `np.clip(weights, 0, 0.30)` 후 끝 → 초과분 증발
- 수정: 반복적 클리핑 + 잔여 양수 섹터에 비례 재배분 (최대 10회 수렴)
- 총합이 `investable(95%)` 미만이면 비례 스케일업 → 유휴 현금 최소화
- **효과**: 섹터 3개만 양수여도 95%까지 투자 (기존 대비 ~10~15%p 현금 감소)

**② TWAP Wave 1 비율 상향 + 소액 면제 기준 상향**
- Wave 1: 35% → **50%** (후속 Wave 누락 시 미투자분 65% → 50%로 축소)
- TWAP 적용 기준: 50만원 → **100만원** (100만원 미만은 Wave 1에서 전량 실행)
- **효과**: 스케줄러 장애 시에도 최소 50% 투자 보장

**③ Half-Kelly 하한 완화**
- 최소 스케일: 0.3 → **0.5** (약한 신호도 최소 50% 투자)
- **효과**: 기존 대비 약한 신호 주문 금액 ~67% 증가

### 10.3 변경 파일

| 파일 | 변경 |
|------|------|
| `live/signal_to_order.py` | `_validate_weights()` 재배분 로직, TWAP 비율/기준, Kelly 하한 |
| `config/live_config.yaml` | `twap_wave_fractions.wave1: 0.50`, `twap_threshold: 1000000` |

### 10.4 예상 효과

| 시나리오 | 기존 투자율 | 수정 후 투자율 |
|----------|-----------|--------------|
| Wave 1만 실행 + 3섹터 양수 | ~30% | **~48%** |
| 전체 Wave 정상 + 3섹터 양수 | ~65% | **~95%** |
| 전체 Wave 정상 + 8섹터 양수 | ~80% | **~95%** |

---

## 11. 향후 개선 방향

1. **VAE 재학습**: NaN 수정 + cross-sectional 타겟으로 재학습 → alpha > 0 확인
2. **FinBERT 뉴스 감성**: 현재 키워드 기반 → 사전학습 모델로 업그레이드
3. **오더북 데이터**: KIS API 호가창 조회 → 호가 불균형 피처
4. **멀티 타임프레임**: 일봉 + 주봉 + 월봉 시계열 병합
5. **GPU 추론**: 서버 GPU 상시화 (현재 CPU only)
6. **Black-Litterman**: 모델 예측을 view로 사용하는 BL 포트폴리오 최적화
7. **TWAP 완료율 모니터링**: Wave 2/3 미실행 시 자동 보상 매수 (catch-up 주문)
8. **동적 cash_buffer**: 레짐별 현금 버퍼 조정 (bear 10%, bull 3%)
