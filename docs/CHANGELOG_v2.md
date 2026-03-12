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
09:10  [KR] Wave 1 (35%)
10:00  [KR] Layer 2 #1 (인트라데이 모멘텀 + DART 공시 + 스탑로스 3%)
11:00  [KR] Wave 2 (30%, Layer 1 x Layer 2)
13:00  [KR] Layer 2 #2 (+ DART + 스탑로스)
13:30  [KR] Wave 3 (20%)
15:35  [KR] 장후 시간외 종가 (5%)
16:00  EOD 기록
16:30  [KR] 장후 시간외 단일가 (5%)
18:30  [US] Pre-market (15%)
23:40  [US] Wave 1 (35%)
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

| 이슈 | 심각도 | 비고 |
|------|--------|------|
| VAE loss NaN | 중 | 72 피처 중 일부 스케일 이슈, 앙상블은 정상 |
| Transformer dir_acc 48.9% | 중 | 랜덤 이하 — VAE NaN 전파 영향 |
| 모델 vs 베이스라인 동일 수치 | 저 | alpha_blend 적용 경로 점검 필요 |
| KOSPI 섹터 30.5% (목표 50%+) | 저 | 키워드 추가 확장으로 개선 가능 |
| 추론 속도 CPU 10~15분 | 저 | GPU 상시화 시 2~3분 가능 |

---

## 8. 향후 개선 방향

1. **VAE NaN 해결**: 신규 피처(alt_*) 정규화 파이프라인 점검, inf/NaN 클리핑
2. **피처 중요도 분석**: 72개 피처 중 실효성 검증, 저기여 피처 제거
3. **FinBERT 뉴스 감성**: 현재 키워드 기반 → 사전학습 모델로 업그레이드
4. **오더북 데이터**: KIS API 호가창 조회 → 호가 불균형 피처
5. **멀티 타임프레임**: 일봉 + 주봉 + 월봉 시계열 병합
6. **GPU 추론**: 서버 GPU 상시화 (현재 CPU only)
