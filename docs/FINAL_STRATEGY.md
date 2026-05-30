# 최종 전략 — 시장별 상세 정리

**작성**: 2026-05-30
**상태**: NASDAQ(V3.3) LIVE 유지 · KOSDAQ(V4) paper 자동 가동(2026-06-01~)
**핵심**: 2개 시장 × 2개 독립 엔진. 시장마다 통하는 메커니즘이 다르므로 **같은 전략을 강제하지 않는다.**

---

## 0. 개요 — 2-엔진 포트폴리오

| 시장 | 엔진 | 메커니즘 | 상태 | 시간대 |
|------|------|----------|:----:|--------|
| **NASDAQ** | V3.3 Vol-Expansion Trader | 변동성 팽창 예측 → 확신도 사이징 (방향 무관) | ✅ LIVE (paper) | 미국 밤 (KST 22:00~04:30) |
| **KOSDAQ** | V4 Momentum Engine | multi-lb ensemble 추세 + regime gate (retail herding) | 🟢 paper 자동 (2026-06-01~) | 한국 낮 (KST 09:00~15:30) |
| KOSPI (후보) | vol-expansion (V3.3 메커니즘 이식) | 변동성 conviction × 약한 direction (효율적 대형주) | 🟡 검증 중 (알파 real, §4.1) | 한국 낮 |

**왜 2-엔진인가**:
- 두 메커니즘이 반대(미국=변동성 크기, 한국=추세 herding) → 상관 낮음 → 결합 분산효과
- 시간대 분리(미국 밤/한국 낮) → 24시간 자본 회전
- **NASDAQ-100이 막힌 건 "시장 선택" 문제였음**: 효율적 시장에선 개별종목 방향 예측 불가 → 변동성(크기)만 수익화. 한국은 덜 효율적(개인 herding) → 추세 수익화.

### 목표 (2회 리셋됨)
- ~~"연 20% 신화"~~ (V1/V2 시절, 폐기)
- **현재: "무위험 + α + 내 시스템"** — 무위험(~3%)을 진짜 엣지로 이기고 risk 제어
- 배포 게이트: 백테스트 Sharpe ≥ 1.0 권장(KOSDAQ 0.66은 "진짜 +α"로 수용), **가짜 엣지(mirage) 배포 거부**

---

## 1. 투자 철학 — "확신 있을 때만, 크게, 빠르게"

> 작은 edge × 반복 × Kelly sizing = 수익

| 원칙 | 의미 | NASDAQ 구현 | KOSDAQ 구현 |
|------|------|-------------|-------------|
| **1. 확신 없으면 안 산다** | 0~100% 현금 허용 | opportunity < gate → 현금 | regime gate OFF → 전량 현금 |
| **2. 집중 투자** | 1~3종목 집중 (분산 ≠ 안전) | max 3 포지션, 종목당 15~40% | top-20 동일가중 |
| **3. 빠르게 청산** | "thesis 깨지면" 청산 (시간 아님) | 리스크 기반 무조건 청산 | regime/vol-target 동적 조정 |

**현금은 포지션이다** — 확신 없는 날 현금 100%는 올바른 판단. "투자 안 하면 기회비용"이 아니라 "투자하면 비용".

---

## 2. NASDAQ 엔진 — V3.3 Vol-Expansion Trader (LIVE)

### 2.1 핵심 thesis
**효율적 시장(NASDAQ-100 대형주)에서 방향 예측은 죽었다(direction IC ≤ 0.03). 하지만 변동성 팽창은 예측 가능하다(IC 0.70).** 방향을 안 맞히고, 변동성을 확신도로 변환해 사이징한다.

### 2.2 알파 체계 (Two Sigma/AQR 컨벤션)

```
opportunity(ticker) = direction · conviction        ∈ [-0.1, 0.1]
  direction  = Σ_a w_a(regime) · α_a(ticker)        (signed, [-0.1,0.1])
  conviction = Π_c c_s(ticker)                        (unsigned, [0,1])

enter_if: opportunity > cost × 1.75    # cost=0.001(왕복0.1%), gate=0.00175
```

| 축 | 역할 | 구성 |
|----|------|------|
| **DirectionalAlpha** | 수익률 예측 (signed) | `trend`, `reversion`, `volume_surprise`(+0.028), `breakout_fade`(+0.020) |
| **ConvictionSource** | 확신도 예측 (unsigned) | `vol` (VolTransformer) — **risk model이지 alpha 아님** |

- **VolTransformer**: d_model=192, 5 layers, 2.26M params. 변동성 팽창 magnitude amplifier.
  - `vol_predicted`를 signed alpha로 변환 실험 → IC 0.007 (방향 예측 불가 데이터 확정)
- 알파 가중치: regime별, 매월 1일 06:00 KST 자동 재학습 (`alpha_weight_trainer.py`, 3년 lookback)
  - `ic_to_weights`: `sqrt(max(IC−0.02,0))` + min_weight 0.10 floor (winner-take-most 완화)

### 2.3 진입 — OpportunityScorer + EntryFilter (5제약)
- 포지션 한도 max 3 · 월 거래 한도(dynamic, unique-ticker) · circuit breaker · 유동성(일거래 ≥ 5억) · 섹터 집중(≤ 2/sector)

### 2.4 사이징 — position_scale = max_gross_exposure (2026-05-13 재해석)
**`position_scale`은 종목당 곱셈이 아니라 포트폴리오 노출 한도.** (이 재해석으로 원칙②"크게" 3/10→7/10, ABNB 자본 38% 실증)

```
raw_weights = inv-vol × Half-Kelly × correlation drag → min(max_weight, raw)
종목당 floor 0.15 미달 → drop | 총합 > position_scale → 약한 종목부터 drop | 정규화 안 함(미달=현금)

POSITION_SCALE_CURVE (regime score → max_gross_exposure):
  0.00→0.00(bear=CASH)  0.25→0.30  0.40→0.60  0.55→0.90  0.75→1.10(cap 1.0)  1.00→1.20(cap 1.0)
```
- 최대 단일 0.40 · 최소 단일 0.15(절대 floor) · 최소 거래 500만원 · long-only
- predicted_vol = `vol_cc_20d`(annualized) — vol_score는 ranking signal이지 vol 값 아님

### 2.5 청산 — conditional veto (시간/리스크 분리)
```
시간기반(opportunity 재평가 트리거):
  profit_take: Day1 +5% → Day5 +1.5%  → opportunity > gate면 유지(veto)
  max_hold: 5일                        → 동일 재평가
리스크기반(무조건 청산, veto 금지):
  vol_contraction: 진입 vol 70% 이하 3일 지속
  dynamic_stop_mae · portfolio_stop: 일간 -1.0~2.0%
```
- **승자 자르기 방지 = 손절만큼 중요** | 서킷브레이커: MDD 5%→75% / 10%→50% / 20%→25% / 30%→전량청산

### 2.6 V3.3 활성 상태 (Phase 1/3 ON, Phase 2/4 OFF since 2026-05-13)
| Phase | Features | 상태 | 사유 |
|-------|----------|:----:|------|
| 1 진단 | no_trade_logger/tc_monitor/execution_quality | ON | read-only |
| 2 Edge | edge_calibrator/engine/tier/allocation | **OFF** | calibration FAIL(top-bottom −0.0001), 5d return alpha 아님 |
| 3 Exit | exit_thesis/partial_exit/signal_decay | ON | alpha 가정 무관 |
| 4 Capital | pyramid/rotation | **OFF** | Edge layer 의존 |

### 2.7 성과 / 스펙
- BT: Sharpe 1.65, +38%, MDD 4%, Win 64%, PF 3.93 | paper(4/11~5/8): +1.39%, 승률 71%, 손익비 12:1
- universe: 99 NASDAQ-100 | cost: 0.1% 왕복 | 빈도: 월 ~5회 | 데이터: yfinance

---

## 3. KOSDAQ 엔진 — V4 Momentum (paper 자동 가동 2026-06-01~)

### 3.1 핵심 thesis
**한국 시장은 개인 herding으로 덜 효율적 → 추세(time-series momentum)가 진짜 엣지.** 단, raw momentum은 MDD -89%(momentum crash) → regime gate + vol-target으로 risk 제어.

### 3.2 확정 SPEC (6중 게이트 통과 — V4 유일)
```
시장        : KOSDAQ 단독 (KOSPI ensemble 0.15 → 기각)
신호        : multi-lb ensemble momentum
              lookback {40,60,90,120} 거래일 랭크평균 blend + TS trend(past 평균 > 0)
universe    : PIT 거래대금 top 100 (point-in-time, look-ahead 제거)
holding     : 20 거래일, N=20, long-only (공매도 제약)
regime gate : 200d SMA — 지수 하회 시 전량 현금 (크래시 보험)
vol-target  : 0.15 target, cap 1.5 (실운영 무레버리지 ≈ cap 1.0)
비용        : 0.4% 왕복 + 상폐 50% 패널티
```

### 3.3 검증 결과 (full-cycle 2014-2026, survivorship-free)
| 지표 | 값 |
|------|-----|
| Sharpe | **0.66** |
| annual | **+10.5%** |
| MDD | **-17%** |

**6중 게이트**: survivorship-free(FDR 상폐 포함) + beta 제거 + 다중 하락기 + look-ahead 제거 + walk-forward 생존 + param robustness

### 3.4 설계 근거 (검증 narrative)
- **단일 창(2021-26) Sharpe 1.0은 period luck**(2025-26 폭발장) → full-cycle 기준 Sharpe ~0.5가 현실
- **regime gate**: 느린 약세장(2015·2018·2022)에 강력(MDD 반감), 빠른 V자(2020 COVID)엔 whipsaw
- **vol-target keeper**: MDD -40%→-21%, COVID whipsaw 부분개선. cap 1.0이라 annual은 소폭 하락
- **trailing stop 기각**: Sharpe 0.41→0.17 (추세 전략에 스탑 해로움 — V3 "개별스탑 제거" 교훈 재확인)
- **multi-lb ensemble**: 단일 lb fragility 제거 (lb40 0.93은 신기루, lb60 0.52가 robust → ensemble 0.66)

### 3.5 실행 — `v4/` 패키지 (backtest=live 단일 코드 경로)
```
v4/engine.py    : ensemble_picks / regime_on / vol_target_exposure / target_book (순수함수)
v4/data.py      : PIT 패널 load | v4/backtest.py : run_backtest (research 수치 정확 재현)
v4/live/        : runner(rebalance-or-hold) + state(영속화) + data_live(FDR)
v4/execution/   : kis_broker(KIS 국내) + executor(reconcile: target→청산/신규/조정)
테스트 38개 통과 (invariant + parity)
```
- 데이터: FinanceDataReader(무료, 상폐 포함) | 체결: KIS 국내 sandbox paper
- **별도 venv_v4 격리** (FDR numpy 2.x ↔ V3 torch numpy 1.26 충돌 방지)

### 3.6 운영
- systemd `quant-v4-korea.timer`: **평일 09:05 KST** (직전 종가 신호 → 개장 직후 체결, sandbox 장외 주문 거부 회피)
- 첫 세션: executor reconcile이 잔여 V1/V2 종목 자동 청산(=리셋) + KOSDAQ 매수
- ⚠️ 미검증: T+2 결제(sandbox 매수여력) / 첫 실주문 경로 — paper라 안전·자동복구
- 관찰: `tail -f /var/log/quant-v4.log` | 중단: `systemctl disable --now quant-v4-korea.timer`

---

## 4. 기각된 시장 / 접근 (반복 방지 — 전부 데이터로 확인)

> "수익 올리자 / 새 alpha 시도" 시 **먼저 여기 참조.** 막힌 경로 반복 금지.

### 4.1 KOSPI — 가격/펀더멘털 기각, **vol-expansion은 알파 real (검증 중)**
| 접근 | 결과 | 판정 |
|------|------|:----:|
| 가격신호(reversion/low-vol/momentum) | full-cycle Sharpe ~0 (효율적, NASDAQ-100과 동일) | ❌ 기각 |
| 펀더멘털(value/quality, DART) | clean ROE IC 0.15(약함), 0.5 근처는 survivor-bias mirage | ❌ 기각 |
| **vol-expansion** (V3.3 메커니즘) | **conviction marginal alpha +2.8%/yr, market-neutral (β0.67)** | 🟡 검증 중 |

**핵심 (2026-05-30 발견)**: KOSPI = NASDAQ-100의 구조적 쌍둥이(효율적 대형주) → 올바른 무기는 momentum(KOSDAQ용)이 아니라 **vol-expansion**. 과거 기각은 "KRX 1% 비용" 가정 탓인데 KOSPI 대형주 현실 왕복 ~0.3%(거래세 인하).
- `kospi_vol_expansion_probe.py` + `_v2.py` (full-cycle 2014-2026, top100 거래대금, 200d gate)
- **vol-expansion conviction이 진짜 market-neutral 알파 생성** — opp-only α+2.9% → opp×conv α+5.7% (동일 β0.67). 베타틸트 아님.
- 단 **deployable 롱온리+gate Sharpe 0.49 < buy&hold 0.58** (gate가 상승장 β 업사이드 깎음 + 한국 리테일 숏 제약으로 알파 분리 불가).
- **다음 결정(미정)**: VolTransformer(IC 0.70 ≫ crude proxy) KOSPI 재학습 시 알파 배가 → 0.58 넘을 가능성 = 3번째 엔진 후보. 단 멀티데이 작업, 기대 페이오프 modest.
- **결론**: KOSPI 완전 기각 아님. **메커니즘(vol-expansion) 전이 확인 = real 알파.** 풀엔진 베팅 여부만 미결.

### 4.2 미국 — V3.3 외 전부 기각 (2026-05-27~30 전수 탐색)
| 접근 | 결과 |
|------|------|
| 개별종목 방향 alpha (8개) | direction IC ≤ 0.03, 비용 후 net ≈ 0 |
| reversion (broad universe) | **survivorship bias** (survivor +0.64 → SF -0.50) |
| momentum (broad) | -0.16 (음수) |
| 페어/stat-arb | gross Sharpe ~0 (수렴 엣지 소멸) |
| overnight anomaly | 성립 안 함 + turnover에 죽음 |
| 섹터 ETF 로테이션 | 선택 무가치, 추세게이트=방어 overlay(수익 아님) |
| sentiment (EODHD 구조화) | IC < 0.02 (priced-in) |
| 뉴스 LLM 이벤트추출 | fade만, LLM ≈ polarity, 기존 alpha와 중복 |
| retail attention (Wikipedia) | 5d IC 0.0145 persistent but sub-threshold, 비용 후 net ≈ 0 |

**메타 교훈**:
- 효율적 시장에선 개별종목 엣지(방향/relative-value/대체데이터)가 무료 데이터로 소진됨
- 유일 faint 신호 = retail herding → monetize엔 **유료 order-flow 데이터(Quiver $30+)** 필요 (보수적 prior)
- **vol IC 0.70조차 직접 monetize 불가** (옵션 VRP) → V3.3처럼 확신도로만 사용

---

## 5. 리스크 관리 (공통 원칙)

- **개별 스탑 제거**: V2.2~V4 일관 결론. MAE 스탑 제거 시 Sharpe 0.65→1.65 (추세에 스탑 해로움)
- **regime 기반 현금화**: bear/하락기 0% 노출 (NASDAQ position_scale=0, KOSDAQ gate OFF)
- **사이징**: 공분산 + Half-Kelly(NASDAQ) / vol-target(KOSDAQ)
- **백테스트-라이브 단일 코드 경로**: 차이는 데이터 공급 방식뿐. 게이트 완화 배포 금지(V2.2 교훈: -6.91%)
- **검증 우선**: 가짜 엣지를 라이브 전 차단. "1.0 신기루 < 0.66 진실"

---

## 6. 운영 인프라

| 구성 | 내용 |
|------|------|
| 서버 | 77.42.78.9 (Asia/Seoul KST) |
| V3 서비스 | `quant-trading-v3.service` (daemon), venv(torch+numpy1.26) |
| V3 타이머 | alpha-retrain(월1 06:00) · v33-daily-report(16:00) · v33-rollback-check(16:30) · calibration-retrain(월1 07:00) |
| V4 서비스 | `quant-v4-korea.service` (oneshot), **venv_v4 격리**(FDR+numpy2.x), TimeoutStartSec=1800 |
| V4 타이머 | `quant-v4-korea.timer` 평일 09:05 KST |
| 배포 | `deploy_v3_git.sh`(V3 전체) · `deploy_v4.sh`(V4 격리) |

---

## 7. 성과 요약

| 엔진 | 검증 기간 | Sharpe | Return | MDD | 비고 |
|------|----------|:------:|:------:|:---:|------|
| V3.1 BT | 2026-04 | 1.65 | +38% | -4% | vol 팽창 |
| V3.2.1 paper | 4/11~5/8 | n/a | +1.39% | n/a | 승률 71%, 손익비 12:1 |
| V3.3 부분활성 paper | 5/20~ | 관찰중 | 관찰중 | 관찰중 | sizing 6.5배(ABNB 38%) |
| **V4 KOSDAQ BT** | 2014-2026 | **0.66** | **+10.5%** | **-17%** | full-cycle, 6중 게이트 |
| V4 KOSDAQ paper | 2026-06-01~ | 가동예정 | — | — | 자동 |

---

## 8. 핵심 교훈 (이 시스템이 배운 것)

1. **예측 대상 전환** — 수익률(IC 0.04) → 변동성 팽창(IC 0.70). 가장 중요한 결정.
2. **시장이 전략을 결정** — KRX 비용 1%, NASDAQ 0.1%. 효율적 시장은 방향 불가, 덜 효율적 시장은 추세 가능.
3. **"잘 예측"과 "수익화"는 다름** — vol IC 0.70도 옵션 시장이 이미 반영하면 monetize 불가.
4. **단일 창 검증은 위험** — KOSDAQ 2021-26 Sharpe 1.0은 period luck. full-cycle 필수.
5. **단일 파라미터 튜닝은 fragile** — lb 단일 튜닝 신기루, multi-lb ensemble만 robust.
6. **추세 전략에 개별 스탑은 해롭다** — 일관 재확인.
7. **survivorship bias가 가장 흔한 mirage** — 미국 reversion(+0.64→-0.50)을 죽인 함정.
8. **정직한 negative > 가짜 positive** — 막힌 경로를 데이터로 확인하고 기록하는 것이 mirage 배포보다 가치 있다.

---

## 참고 문서
- `CLAUDE.md` — 운영 정책 본문
- `docs/V4_DUAL_ENGINE_DESIGN.md` — V4 설계/검증 상세 (§1.9 SPEC)
- `docs/CHANGELOG.md` — Phase 이력
- `docs/FOLLOW_UPS.md` — 후속 과제
- `memory/v4_edge_exploration.md` — 기각 경로 전수 기록
