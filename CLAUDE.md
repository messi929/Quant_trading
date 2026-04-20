# Quant Trading System — CLAUDE.md

## 투자 철학 (Trading Philosophy)

> **"확신 있을 때만, 크게, 빠르게"**
> — 작은 edge × 반복 × Kelly sizing = 수익

이 시스템의 모든 코드, 모델, 실행 로직은 아래 원칙을 벗어나지 않는다.

### 핵심 원칙 3가지

**1. 확신 없으면 안 산다 (Conviction-or-Cash)**
- 모델 신뢰도가 threshold 미만이면 현금 보유 (0~100% 현금 허용)
- "항상 투자해야 한다"는 전제를 거부한다
- Two Sigma의 "sigma-based amplification": 신뢰도에 비례해서만 사이징

**2. 집중 투자한다 (Concentration over Diversification)**
- 소자본(1억)에서 9종목 분산은 수수료에 잠식된다
- 확신 있는 1~3개 종목에 집중 → 개별 2~3% 수익 → 비용 상쇄 후 1%+
- 종목당 충분한 금액이 들어가야 의미 있는 수익이 나온다

**3. 빠르게 청산한다 (Fast Exit)**
- 이익: +2~3% 도달 시 50~100% 이익 실현
- 손실: -2% 도달 시 전량 손절
- 시간: 보유 기간 최대 3일, 기본은 당일 청산
- Medallion 원칙: 보유 기간 ~2일, 리스크 노출 최소화

### 수익 목표

- **월 평균 일 1%** (월 +20%, 거래일 기준)
- 매일 달성이 아닌, 월 누적 기준
- 승률 55%+ × 손익비 1.5:1 이상 → 월 목표 달성 가능

---

## 매수 정책 (Buy Policy)

### 매수 3대 조건 — 모두 충족 시에만 진입

```
조건 1: 모델 신뢰도 (Model Conviction)
  - Rank IC ≥ 0.10
  - 상위 종목의 score가 나머지 대비 확연히 높을 것
  - 전체적으로 약하면 → 매수 안 함 (현금 보유)

조건 2: 시장 환경 (Market Regime)
  - bull 또는 neutral 레짐에서만 매수
  - bear 레짐 → 현금 100% (포지션 진입 금지)
  - 변동성 급등 시 → 사이즈 50% 축소

조건 3: 거래 품질 (Trade Quality)
  - ICR ≥ 3.0 (신호 강도가 거래비용의 3배 이상)
  - 거래량 상위 종목만 (유동성 확보)
  - 장 시작 30분 내 집중 (가장 효율적 시간)
```

### 포지션 사이징 — v2.3 개정

- ~~Half-Kelly 기반~~ → v2.3: 변동성 타겟팅 + 균등비중 + confidence tilt
- 종목 간 균등 비중 (N종목 → 1/N), confidence에 따라 ±50% 조절
- 최대 단일 종목: 포트폴리오의 40%
- 최소 거래 금액: 500만원 (이하는 의미 없음)
- v2.3에서 제거: `score × confidence × 10` (이론적 근거 없는 휴리스틱)

---

## 매도 정책 (Sell Policy) — v2.3 개정

### 핵심 원칙: 예측 기간 동안 보유

모델이 3일 후를 예측 → 3일간 보유하여 예측 실현 기회 확보.
개별 종목 스탑로스/time_exit 제거 → 포트폴리오 레벨 리스크 관리.

### 청산 조건 (4가지만)

```
1. 이상치 이익 실현: +5% → 전량 청산
2. 보유 기간 만료: 3일 도달 → 전량 청산 (prediction_horizon 종료)
3. 신호 반전: 다음 날 신호에 종목이 없으면 → rebalance 청산
4. 포트폴리오 손실 한도: 일간 -1.5% (미실현 포함) → 전 포지션 청산
```

### v2.3 추가: 진입 제한 (Medallion alignment)

```
5. 고스트 포지션 제거: 매도 3회 연속 실패 → open_positions에서 강제 제거
6. 멀티데이 냉각기: 5일 내 2회 이상 손실 종목 → 진입 차단
7. 연속 진입 제한: 동일 종목 3일 연속 진입 불가 (신호 정체 방지)
8. 포지션 영속화: open_positions.json으로 서비스 재시작 시 복구
```

### 제거된 규칙 (v2.0→v2.2)

```
[제거] 개별 종목 -2% 스탑로스 → 노이즈 청산, V자 반등 매도 방지
[제거] 2시간 +1% 미달 time_exit → 3일 예측을 2시간에 판단하는 모순
[제거] +2.5% 부분 이익실현 → 불필요한 복잡도, +5% 전량만 유지
[제거] 장 마감 전 전량 청산 → 오버나이트 보유로 턴오버 감소
```

### 서킷 브레이커 (MDD 기반)

```
MDD > 5%  → 포지션 사이즈 75%
MDD > 10% → 포지션 사이즈 50%
MDD > 20% → 포지션 사이즈 25%
MDD > 30% → 전량 청산, 거래 중단
```

---

## 매매 타이밍 (Execution Timing)

### 원칙

- 신호 생성 후 시장 안정화 대기 후 실행 (09:30, 개장 30분 후)
- Signal decay half-life: **6시간** (24시간은 과도하게 관대)
- 스프레드 안정화 후 진입 (개장 직후 회피)
- 시간외 거래: 최소화 (유동성 부족)

### KR 스케줄 (v2, 간소화)

```
06:00  데이터 수집 (yfinance + pykrx, 증분 10일)
06:10  신호 생성 (AlphaTransformer → ConvictionSignalGenerator)
09:30  매수 (스프레드 안정화 후, 확신 있는 1~3 종목)
09:35~15:15  모니터 (5분 간격, +5% 이익실현/3일 만료/포트폴리오 리스크)
15:20  세션 종료 (만료 포지션만 청산, 나머지 오버나이트 보유)
16:00  EOD 성과 기록
```

### US 스케줄 (v2, 간소화)

```
22:00  US 신호 재생성 (06:10 신호 재사용 금지)
23:40  매수 (1~2 종목)
23:45~04:30  모니터 (5분 간격)
04:30  청산
```

### 폐지 세션 (v1→v2)

- 장전 시간외 (07:30): 유동성 <1%, 시장 충격 과다 → 폐지
- 장후 시간외 (15:35, 16:30): 불필요한 복잡도 → 폐지
- US after-hours (05:10): 신호 23시간 경과 → 폐지
- US pre-market (18:30): 불필요한 복잡도 → 폐지
- TWAP Wave 분할 (09:10/11:00/13:30): 단일 진입으로 통합 → 폐지

---

## 모델 원칙 (Model Principles)

### 알파가 없으면 거래하지 않는다 — v2.3 강화

- Transformer dir_acc < **54.5%** → 거래 중단, 모델 개선 우선 (v2.2: 52%)
- Rank IC < **0.10** → 라이브 배포 금지 (v2.2: 0.05로 완화했으나 복원)
- 백테스트 Sharpe < 1.0 → 라이브 배포 금지
- 백테스트와 라이브 코드 경로가 동일해야 한다
- **게이트를 완화해서 배포하지 않는다** (v2.2 교훈: 7일간 -6.91%)

### 단순한 모델이 복잡한 모델보다 낫다

- **현재 (v2.3)**: AlphaTransformer 단일 모델 (~1.6M params, 35 features)
- v1은 VAE→Transformer→GAN→RL 4단계 → v2에서 단일 모델로 축소
- Confidence head로 예측 확신도를 함께 출력
- **v2.3**: ListMLE ranking loss (순위 학습), Huber delta 1.5 (아웃라이어 로버스트)

### 피처 원칙

- 모든 피처는 z-score 정규화 후 모델에 입력 (학습셋 통계 기준)
- 원본 스케일 그대로 모델에 넣지 않는다
- `saved_models/normalizer_stats.json`에 통계 저장, 추론 시 로드
- **v2.3**: 71 → 35 피처 (상관관계 기반 중복 제거, max_corr=0.85)
- 리턴 클리핑: ±30% (KRX 일일 제한가), v2.2의 [-100%, +1000%] 수정
- 피처 선별은 `select_features()` 함수로 학습 시 자동 수행

---

## 백테스트-라이브 정합성 (Consistency Rule)

### 절대 규칙

- 백테스트와 라이브에서 **동일한 함수**로 신호 정규화
- 백테스트와 라이브의 **비용 가정 동일** (commission, slippage)
- 백테스트에 **이익실현/손절 규칙** 반드시 포함
- top_k, max_position, threshold 등 **모든 파라미터 통일**
- **v2.3**: 백테스트 day_trade_default=False (3일 보유 반영)
- **v2.3**: 매도 슬리피지 추가 (기존: 매수만 적용 → 양방향)

### v2.3 비용 파라미터 (현실화)

```yaml
transaction_cost_rate: 0.010   # 왕복 1.0% (v2.2: 0.6%)
commission_rate: 0.0005        # 편도 0.05% (v2.2: 0.02%)
slippage_by_market:
  KOSPI: 0.003                 # 매수+매도 각각 적용
  KOSDAQ: 0.005
  NASDAQ: 0.001
```

---

## 리스크 관리 원칙 (Risk Management)

### Medallion 원칙 적용

- 승률 50.75%로도 수익 — **edge × 반복 × sizing**
- ~~Half-Kelly sizing~~ → v2.3: 변동성 타겟팅 + 균등 + confidence tilt
- 거래량 5% 참여율 제한 (시장 충격 관리)
- **v2.3 거래비용**: 왕복 1.0% (수수료 0.05%×2 + 세금 0.18% + 슬리피지 0.3%×2 + 스프레드)
- **v2.3 레짐**: bear 임계값 -8% (v2.2의 -3%는 정상 조정에서 과민 반응)

### 현금은 포지션이다

- 확신 없는 날 현금 100%는 **올바른 판단**이다
- "투자 안 하면 기회비용"이 아니라 "투자하면 비용"이다
- bear 레짐에서 현금 보유는 수익이다
- **게이트 미통과 모델로 거래하는 것은 확신이 아니라 도박이다** (v2.3 교훈)

---

## 코드 원칙 (Code Principles)

### 설정과 코드 분리

- 매직 넘버 금지: 모든 상수는 config YAML에서 로드
- TWAP fractions, cost rate, threshold 등 하드코딩 금지

### 중복 금지

- 신호 정규화: 단일 함수 (`strategy/signal.py`)에서만 수행
- 백테스트/라이브 동일 함수 호출

### 검증 가능해야 한다

- 모든 거래에 이유(score, regime, conviction)가 로그에 남아야 한다
- 백테스트 결과 재현 가능해야 한다

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
- **교훈: 게이트를 완화해서 배포하면 안 된다**

### Exit Strategy 비교 백테스트 (120일, 2025-09~2026-03)

| 전략 | 수익률 | Sharpe | 승률 |
|------|--------|--------|------|
| **V2.2 (3일, 스탑없음)** | **-7.65%** | **-1.38** | 39.9% |
| 3일+SL2% | -18.43% | -6.52 | 25.0% |
| 당일청산 | -14.48% | -3.76 | 31.1% |
| 당일+SL2% (V2.0) | -17.26% | -7.43 | 25.3% |

**결론:** 스탑 제거+3일 보유가 상대적으로 최적이지만, 모든 전략이 마이너스.

---

## Phase 22 현황 (2026-04-01 완료)

Phase 22~23을 통합 실행하여 V2 재구축 + 서버 배포 완료.

### 완료 항목
- [x] Feature 정규화 (z-score per column, 학습셋 통계)
- [x] AlphaTransformer 단일 모델 (d_model=192, 5 layers, mean pooling)
- [x] Confidence head (방향 정확도 확률 출력)
- [x] 백테스트/라이브 파라미터 완전 통일 (system_config.yaml 1개)
- [x] Conviction threshold (0~3종목, 0~100% 현금)
- [x] ~~Profit taking (+2.5% 절반, +5% 전량)~~ → v2.2: +5% 전량만
- [x] ~~Stop loss (-2% 전량)~~ → v2.2: 제거 (포트폴리오 레벨)
- [x] ~~시간 기반 청산 (2시간 내 +1% 미달)~~ → v2.2: 제거 (3일 보유)
- [x] 거래 세션 간소화 (7세션 → 2세션)
- [x] US 신호 22:00 재생성
- [x] 데이터 수집 파이프라인 연결
- [x] 서버 배포 + E2E 검증

### 모델 성능 이력

| 버전 | Dir Acc | Val IC | Test IC | 피처수 | Loss | 게이트 |
|------|---------|--------|---------|--------|------|--------|
| V2.2 | 52.51% | 0.1044 | 0.0529 | 71 | Pairwise+Huber(0.5) | Val PASS/Test FAIL(완화) |
| **V2.3** | **53.07%** | **0.0748** | **0.0383** | **35** | **ListMLE+Huber(1.5)** | **모두 FAIL(정직)** |

### V2.2 변경 (2026-04-03)

실거래 2일(4/2~4/3) 분석 결과, 전략 구조 문제 확인:
- 기대값 -1.14%/거래 (승률 37.5%, 손익비 0.23:1, Kelly 음수)
- 근본 원인: prediction_horizon=3일 vs 실제 보유 2시간 (time_exit)
- 043260.KQ: 스탑 -3.32% → 종가 -0.11% (V자 반등에서 바닥 매도)

변경 사항:
- [x] 개별 스탑로스 제거 → 포트폴리오 daily_loss_limit -1.5% (미실현 포함)
- [x] time_exit 제거 → 3일 보유 만료까지 유지
- [x] 이익실현: +5% 전량만 (부분 청산 제거)
- [x] 진입 09:10 → 09:30 (스프레드 안정화)
- [x] session_close: 만료 포지션만 청산, 오버나이트 보유
- [x] 서킷 브레이커: MDD 5/10/20/30% 단계별 스케일링
- [x] 매도 실패 시 KIS 잔고 확인 + 포지션 동기화
- [x] score_history JSON 영속화

### V2.3 변경 (2026-04-09)

V2.2 실거래 7일(4/1~4/8) 분석: -6.91%, MDD -9.97%.
세계적 퀀트 트레이더 관점에서 전면 재진단 → 10가지 개선.

**모델/학습 변경:**
- [x] PairwiseRankingLoss → ListMLE (listwise 순위 학습)
- [x] Huber delta 0.5 → 1.5 (아웃라이어 로버스트)
- [x] 피처 71 → 35개 (상관관계 기반 중복 제거)
- [x] Walk-forward 학습 파이프라인 구현 (4 fold)
- [x] 리턴 클리핑 [-1,10] → [-0.3,0.3] (KRX 현실)

**비용/게이트 변경:**
- [x] 거래비용 0.6% → 1.0% (세금+슬리피지 현실화)
- [x] 수수료 0.02% → 0.05% (한투 실제 비용)
- [x] 배포 게이트: dir_acc 52%→54.5%, rank_ic 0.05→0.10
- [x] bear_threshold: -3%→-8% (과민 반응 방지)
- [x] ranking_loss_weight: 0.5→0.6 (ranking이 핵심)

**실행 인프라 변경:**
- [x] 고스트 포지션 강제 제거 (매도 3회 실패 → 삭제)
- [x] 포지션 디스크 영속화 (open_positions.json)
- [x] 멀티데이 냉각기 (5일 내 2패 → 진입 차단)
- [x] 연속 진입 제한 (동일 종목 3일 연속 → 차단)
- [x] Kelly → 변동성 기반 + 균등 + confidence tilt 사이징

**백테스트 정합성:**
- [x] day_trade_default: True → False (3일 보유 반영)
- [x] 매도 슬리피지 추가 (기존: 매수만)
- [x] 백테스트 exit strategy 비교 스크립트 (7가지 전략)

**결과:** 게이트 미통과 (Val IC 0.0748, Test IC 0.0383). 배포 불가.
**교훈:** OHLCV만으로는 Rank IC 0.10 달성 어려움. 피처 품질이 병목.

### Phase 24: 알파 개선 실험 (2026-04-11) — 실패, 전면 재설계 결정

**Phase A: 피처 추가 실험**

| 실험 | 추가 피처 | Test IC 변화 | 결과 |
|------|----------|-------------|------|
| V2.3 (기준) | OHLCV 35개 | 0.038 | FAIL |
| V2.4 (+수급) | +flow 1개 | 0.057 (+0.019) | FAIL |
| V2.5 (+섹터+베타) | +sector 3, beta 3 | 0.041 (-0.016) | FAIL |

- 수급: 네이버 금융 크롤링 (`data/flow_data.py`), 175종목 2년치 수집
- 외국인 순매수 IC=+0.025 (175종목), 대형주만(61종목)에서는 IC≈0
- 섹터/베타 피처 추가 시 오히려 IC 하락 (과적합 또는 노이즈)

**Phase B: 예측 대상 실험**

| 타겟 | 최고 피처 IC | 비고 |
|------|------------|------|
| 1일 상대수익률 | -0.040 (return_1d) | 3일보다 약간 강함 |
| 3일 상대수익률 (현행) | -0.036 (volatility_20d) | 현행 |
| **변동성 예측** | **+0.487 (volatility_20d)** | **13배 강한 신호** |

**백테스트 최종 검증 (V2.5, 120일)**

| 전략 | 수익률 | Sharpe | 승률 | Cash일 |
|------|--------|--------|------|--------|
| 3일 보유 | **-53.2%** | -9.86 | 26.6% | **0일** |
| 3일+SL2% | -59.4% | -14.86 | 24.3% | 0일 |
| 당일청산 | -54.6% | -16.49 | 16.1% | 0일 |
| 5일 보유 | -56.4% | -7.39 | 28.1% | 0일 |

**결론: 모든 전략 -50% 이상 손실. 피처 추가로 해결 불가.**

### 근본 원인 진단

V1→V2→V2.5 전체를 관통하는 **불변 전제 4가지**가 문제:

```
1. 입력: 공개 OHLCV 가격 데이터 (모두가 접근 가능 → edge 없음)
2. 목표: 3일 수익률 순위 예측 (가장 어려운 예측 문제 중 하나)
3. 실행: 매일 거래 (비용 240%/년 → IC 0.30+ 필요, 현재 0.04)
4. 시장: KRX 왕복 1% (US 0.1%의 10배 비용)
```

**V1→V2 "재설계"는 모델 간소화였을 뿐, 위 전제는 동일했다.**
모델 아키텍처가 아니라 전제 자체를 바꿔야 한다.

---

## V3 — Vol Expansion Trader (Phase 25, 2026-04-11)

> **"변동성 팽창 예측 → 방향은 규칙 → 조건부 진입"**

V2의 4가지 불변 전제를 모두 변경한 전면 재설계.

### 전제 변경 내역

| 전제 | V2 | V3 | 결과 |
|------|-----|-----|------|
| 무엇을 예측 | 수익률 순위 (IC=0.04) | **변동성 팽창** (IC=0.70) | **17.5× 강한 신호** |
| 언제 거래 | 매일 (연 240% 비용) | **월 5회 이하** (연 6% 비용) | **비용 40× 감소** |
| 어디서 거래 | KRX (왕복 1%) | **NASDAQ** (왕복 0.1%) | **비용 10× 저렴** |
| 무엇을 보고 | OHLCV 가격만 | **OHLCV + vol 구조 + 수급 + 이벤트** | 다차원 신호 |

### V3 아키텍처

```
v3/
├── config/        Pydantic 타입 검증 설정
├── data/          OHLCV 수집, vol 피처, z-score 정규화
├── model/         VolTransformer (2.26M params, d=192, h=8, L=5)
├── rules/         방향 판단(규칙), 진입 필터(10조건), 청산 규칙
├── strategy/      통합 신호, 공분산 사이징, 적응형 리스크, 레짐 감지
├── backtest/      종목 레벨 이벤트 드리븐, walk-forward
├── execution/     KIS API, 포지션 영속화, 페이퍼 브로커
├── pipeline/      data, train, inference, live (E2E)
└── scripts/       run_data, run_train, run_backtest, run_live
```

### V3 전략 로직

```
1. VolTransformer: "이 종목, 향후 5일 변동성 팽창할 것인가?"
   - target = vol_5d_forward / vol_20d_current - 1
   - Huber(0.4) + ListMLE(0.6) + BCE(0.1)

2. Direction Engine (규칙 기반, ML 아님):
   - 모멘텀 (50%): 20일 수익률 방향 + vol 레짐 조정
   - 수급 (30%): 외국인/기관 순매수, 지속성 체크
   - 이벤트 (20%): DART 공시 스코어링
   - 평균회귀: KRX 종목 2σ 이격 시 반전 보조
   - 레짐 적응: bull→모멘텀↑, bear→평균회귀↑, volatile→이벤트↑
   - 신호 충돌 감지: 방향 불일치 시 clarity 감소

3. Entry Filter (10가지 조건 모두 충족):
   - vol_expansion > 5%, confidence > 30%, direction_clarity > 20%
   - 월 거래 < 동적한도(MDD/승률 기반), 포지션 < 3개
   - 기대수익 > 비용×1.75 (vol조정 슬리피지 반영)
   - 유동성 500억+, 섹터 집중 2개 이내, 서킷브레이커 OFF

4. 사이징: 공분산 포트폴리오 + Half-Kelly
   - 예측 vol 역수 × confidence × 상관관계 drag
   - Ledoit-Wolf 축소 상관행렬 (과적합 방지)
   - 유동성 제약 (거래량 5% 이내)

5. 청산 (시간감쇠 TP + 포트폴리오 스탑):
   - 시간감쇠 TP: Day1 5%, Day2 4%, Day3 3%, Day4 2%, Day5 1.5%
   - 신뢰도 적응: high conf → 0.7× TP, low → 1.3× TP
   - vol 수축: 진입 vol의 70% 이하 (3일 지속 확인)
   - 보유 만기: 5일
   - 포트폴리오 스탑: 배포 비례 (-1.0~2.0%)

6. 리스크:
   - 적응형 서킷 브레이커: model IC + 집중도 + 회복 반영
   - Vol-of-Vol 충격: z>3 → 사이즈 30%
   - 99% VaR: 스트레스 시나리오 (vol 1.5×, 상관 +0.5)
   - 레짐 히스테리시스: 3일 확인 후 전환, 비용 인식

7. 레짐: bull / bear / volatile / neutral / mean_reversion (5개)
   - 적응형 모멘텀 윈도우: 고vol→30d, 저vol→10d
   - 비용 인식 전환: 마진한 bear 콜에 부분 감축만
```

### 매수/매도 정책 (V3)

```
매수 조건:
  - VolTransformer vol 팽창 예측 > 5%
  - 방향 판단 clarity > 20% (momentum + flow + event 합의)
  - 기대수익 > 거래비용의 1.75배 (vol 조정 슬리피지 포함)
  - 월 거래 한도 미도달 (동적: 2~8회)
  - 포지션 < 3개, 유동성 500억+, 섹터 집중 < 2개
  - long-only (NASDAQ, 공매도 미사용)

매도 조건 (4가지만, 개별 스탑 없음):
  1. 시간감쇠 이익실현: Day1 +5%, Day5 +1.5% (신뢰도 조정)
  2. vol 수축: 진입 vol의 70% 이하 (3일 지속)
  3. 보유 만기: 5일
  4. 포트폴리오 일간 -1.0~2.0% (배포 비례)
```

### 비용 구조 (V3)

```yaml
NASDAQ:
  roundtrip: 0.1% (수수료 0.01%×2 + 슬리피지 0.1%×2)
  월 5회 × 0.1% = 월 0.5% = 연 6%

KRX (제외):
  roundtrip: 1.0% (수수료+세금+슬리피지)
  KOSPI+NASDAQ 혼합 백테스트 결과 -9% → KRX 부적합 확인
```

### V3 성과

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

### V3 실행 명령

```bash
PYTHON=/c/Users/wogus/miniconda3/envs/quant/python.exe

# 데이터 수집 + 피처 (NASDAQ, flow 제외)
$PYTHON v3/scripts/run_data.py --no-flow

# 모델 학습
$PYTHON v3/scripts/run_train.py

# 백테스트
$PYTHON v3/scripts/run_backtest.py

# 라이브 (페이퍼, 1회)
$PYTHON v3/scripts/run_live.py --mode once

# 라이브 (데몬)
$PYTHON v3/scripts/run_live.py --mode daemon

# 페이퍼 계좌 확인
$PYTHON -c "from v3.execution.paper_broker import PaperBroker; PaperBroker().print_summary()"
```

### V3 서버 배포

```bash
# 배포
bash deploy_v3.sh 77.42.78.9

# 상태 확인
ssh root@77.42.78.9 "systemctl status quant-trading-v3"

# 로그
ssh root@77.42.78.9 "tail -f /var/log/quant-v3.log"

# V2 복원 (긴급)
ssh root@77.42.78.9 "systemctl stop quant-trading-v3 && rm -rf /opt/quant && mv /opt/quant_v2_backup /opt/quant && systemctl start quant-trading"
```

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

## V3.2 — Phase 2 Regime/Alpha 재설계 (Phase 26, 2026-04-18)

> **"Regime은 임계값 조작자가 아니라 알파 가중치 선택자"**
>
> Phase 1의 `threshold_multiplier`, `engine toggle`, 상속 변이 등 원칙
> 위배 요소를 전면 제거하고, Two Sigma/AQR 컨벤션에 맞춰 재설계.

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

### 알파 분류 체계 (새로 도입)

Two Sigma/AQR 컨벤션에 맞춰 **두 축으로 분리**:

| 축 | 역할 | 출력 범위 | 현재 구성 |
|----|------|----------|----------|
| **DirectionalAlpha** | 수익률 예측 (signed) | [-0.1, 0.1] (5일 기대 초과수익률) | `trend`, `reversion` |
| **ConvictionSource** | 확신도 예측 (unsigned) | [0, 1] | `vol` (VolTransformer 재분류) |

**핵심 인식**: VolTransformer는 **Risk model**이지 Alpha model이 아니다.
변동성 팽창은 "크기 예측"이지 "방향 예측"이 아니므로 signed return과 IC ≈ 0.
직접 수익률 예측에 합치 않고, 대신 **다른 알파의 확신도를 modulate**.

#### 수식

```
direction(ticker)   = Σ_a  w_a(regime) · α_a(ticker)       ∈ [-0.1, 0.1]
conviction(ticker)  = Π_c  c_s(ticker)                      ∈ [0, 1]
opportunity(ticker) = direction · conviction                ∈ [-0.1, 0.1]

enter_if:  opportunity > cost × k    (k = 1.75)
```

### 매수 정책 — v3.2 개정

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

#### Regime별 진입 방식 차이

**이전**: regime이 `threshold_multiplier`로 entry filter의 `min_vol_expansion`
등을 동적 수정 (뮤테이션) → 원칙 위배.

**이후**: regime은 **알파 가중치**만 결정. 게이트 자체는 고정:
- strong_bull: alpha_weights 학습 결과 (예: reversion 1.0) → reversion 알파가
  주도하는 opportunity → 그 기준 진입
- bull/neutral/caution/bear: 각 regime이 자기 가중치로 opportunity 계산
- bear: `position_scale=0 → CASH` 단락

### 매도 정책 — V3 유지 (변경 없음)

1. 시간감쇠 이익실현 (Day1 +5% ~ Day5 +1.5%)
2. Vol 수축 (진입 vol의 70% 이하 3일 지속)
3. 보유 만기 (5일)
4. 포트폴리오 일간 -1.0~2.0% (배포 비례)

### 포지션 사이징 — v3.2 개정

#### position_scale 연속화

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

#### Alpha weights (regime별)

`v3/config/alpha_weights.json`에 저장 (S2 학습 결과). 최신 (2026-04-18, 3년 학습):

```
strong_bull: {trend: 0.0, reversion: 1.0}   # reversion IC=0.113 유일 양수
bull:        {trend: 0.5, reversion: 0.5}   # IC 모두 게이트 미달 → uniform
neutral:     {trend: 1.0, reversion: 0.0}   # trend IC=0.028 유일 유의미
caution:     {trend: 0.5, reversion: 0.5}   # uniform
bear:        {trend: 0.5, reversion: 0.5}   # uniform (CASH로 우회)
```

### Regime 시스템 — v3.2 재정의

#### 역할 변경
- **이전**: bull/bear 판정 → threshold 조작 (잘못된 설계)
- **이후**: 5 state 분류 → **알파 가중치 선택 + position_scale 결정**

#### Regime 5 state + continuous score

| State | Score 범위 | 의미 | position_scale |
|-------|----------|------|---------------|
| strong_bull | ≥ 0.75 | 강한 위험선호 | ~1.10 |
| bull | 0.55~0.75 | 위험선호 | ~0.90 |
| neutral | 0.40~0.55 | 중립 | ~0.60 |
| caution | 0.25~0.40 | 경계 | ~0.30 |
| bear | < 0.25 | 위험회피 | 0 (CASH) |

#### Composite score 계산

7개 macro feature의 5년 rolling percentile을 **학습된 weights + signs**로 가중합.

```
score = Σ  w_f × (pctl_f  if sign_f == +1  else  1 - pctl_f)
```

#### 학습된 feature_signs (3년 NASDAQ 2023~2026, contrarian 패턴)

```
+1 (percentile 그대로 기여):    vix_ratio, hy_level, gold_spy_mom_60d
-1 (1 - percentile 반전):       yc_slope, hy_change_60d, dxy_mom_60d,
                                hyg_tlt_mom_60d, breadth
```

**해석**: 최근 3년은 NASDAQ 상승장 → "vix 높고 HY 넓으면 반등"(contrarian),
"breadth 높으면 과열"(mean-revert) 패턴이 학습됨. 데이터 기반 결과이므로 존중
하되, 시장 국면 변화 시 월 재학습에서 자동 수정됨.

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

### 백테스트-라이브 정합성 강화

#### 이전 (V3.1) — 다른 코드 경로
- `live_pipeline` → `RegimeDetector` + `DirectionEngine` + `EntryFilter`
- `backtest/engine.py` → **동일하지만 재구현** (불일치 리스크)

#### 이후 (V3.2) — 단일 코드 경로
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
v3/config/alpha_weights.json                           (latest)
v3/config/alpha_weights_history/alpha_weights_YYYY-MM.json (monthly version)
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

### 운영 명령 (V3.2)

```bash
PYTHON=/c/Users/wogus/miniconda3/envs/quant/python.exe
FRED_API_KEY=... # .env에서 로드

# 월 재학습 (매월 1일 실행 권장)
PYTHONPATH=. $PYTHON v3/backtest/alpha_weight_trainer.py --lookback-years 3

# 로컬 smoke test
PYTHONPATH=. $PYTHON -c "
from v3.pipeline.live_pipeline import LivePipeline
p = LivePipeline()
df = p.collect_data()
print(p.generate_signal(df))"

# 서버 배포 (파일별 scp, rsync 없음)
scp v3/strategy/alpha_sources.py v3/strategy/regime_v2.py \
    v3/strategy/opportunity.py v3/strategy/signal.py \
    root@77.42.78.9:/opt/quant/v3/strategy/
scp v3/rules/entry.py root@77.42.78.9:/opt/quant/v3/rules/
scp v3/pipeline/live_pipeline.py root@77.42.78.9:/opt/quant/v3/pipeline/
scp v3/backtest/engine.py v3/backtest/alpha_weight_trainer.py \
    root@77.42.78.9:/opt/quant/v3/backtest/
scp v3/config/schema.py v3/config/v3_config.yaml v3/config/alpha_weights.json \
    root@77.42.78.9:/opt/quant/v3/config/

# 서비스 재시작
ssh root@77.42.78.9 "systemctl restart quant-trading-v3"

# Paper trading 관찰 (일일)
ssh root@77.42.78.9 "tail -100 /opt/quant/v3/logs/v3_$(date +%Y-%m-%d).log | \
    grep -E 'Regime|Signal|Opportunity'"
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

## V3.2.1 — 실행 인프라 보강 (2026-04-20)

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

**정책**
- `reason == "profit_take"` 트리거 시:
  - `opportunity(ticker) > cost × 1.75` 성립 → **보유 유지** (veto + 로그)
  - 그 외 → 기존대로 청산
- 다른 청산 사유(`max_hold`, `vol_contraction`, `dynamic_stop_mae`,
  `portfolio_stop`)는 **veto 금지** (무조건 체결)
- Phase 2의 진입 수식(`opportunity > cost × k`)을 **유지 판단에도 재사용**
  → 단일 진입/유지 기준 (설계 통일)

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

### 매도 정책 — v3.2.1 개정

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

### V3.2.1 커밋

```
<hash>  feat: V3 실행 보강 — PaperBroker 와이어링 + conditional TP
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

### V3.2.1 핫픽스 — hold_days 달력 일수 (2026-04-21)

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

## 개발 워크플로우 (V3.2.1+) — **검증 필수**

Generator(코드 작성) / Evaluator(독립 검증) 분리 체계. **세션마다 일관되게 적용할 것.**

### 자동 검증 — Hook Layer B (PostToolUse)

V3 Python 파일 편집 시 pytest 회귀 suite(`v3/tests/`)가 자동 실행됨. 설정:
`.claude/settings.local.json` PostToolUse + `.claude/hooks/post_edit.py`.

- **Hook 실패(`[v3-evaluator hook] ✗`) 시 즉시 중단**. 다음 작업 금지.
- 먼저 실패 원인 파악 → 코드 수정 → 재편집 → 통과 확인 후 진행.
- 테스트 자체가 틀렸다고 판단되면 테스트부터 수정하되, "테스트가 까탈스러워서
  끄겠다" 금지. 테스트를 완화하려면 더 구체적인 대체 테스트를 먼저 추가.

### 명시적 검증 — Subagent Layer A (`v3-evaluator`)

다음 조건 중 **하나라도** 해당하면 커밋 전 `Agent(subagent_type="v3-evaluator")`
를 호출한다:

1. 3파일 이상 동시 변경 (리팩터링, 다중 모듈 수정)
2. 매수·매도·리스크·regime 등 **정책 로직** 변경
3. `SignalGenerator` / `ExitRules` / `RegimeDetectorV2` 등 core 재작성
4. 모델 재학습 후 live 파이프라인 수정
5. 서버 배포 직전

Evaluator 리포트가 3섹션(regression / invariant / silent-failure) 모두 clean이
아니면 해결 전까지 커밋 금지.

### 새 버그 발견 시 루틴 — **테스트 선행 원칙**

```
버그 제보 → v3/tests/test_regression.py에 회귀 테스트 추가 → 실패 재현 확인
         → 코드 수정 → 테스트 통과 → 커밋 (테스트 + 수정 동일 커밋)
```

- **테스트 추가 없이 "수정했습니다" 보고 금지.**
- 테스트 파일 내 분류는 버그 출처별 `TestXxx` 클래스. 기존 클래스에 추가하거나
  새 클래스 생성.
- 테스트는 invariant를 인코딩한다. 완화는 신중하게, 추가는 적극적으로.

### 금지 사항

- `pytest.skip`, `pytest.mark.skip`으로 실패 우회 금지
- `--no-verify` / `--no-gpg-sign` 등 훅 우회 금지 (사용자 명시 지시 없이는)
- 테스트 통과 못 한 상태에서 "일단 커밋하고 나중에 고치자" 금지
- Hook 끄기(`disableAllHooks: true`) 금지. 해당 세션만 임시 해제도 금지.

### 검증 도구 위치 참고

- pytest 설정: `pytest.ini`
- 회귀 테스트: `v3/tests/test_regression.py`
- Evaluator 정의: `.claude/agents/v3-evaluator.md`
- Hook 스크립트: `.claude/hooks/post_edit.py`
- Hook 등록: `.claude/settings.local.json` `hooks.PostToolUse`
