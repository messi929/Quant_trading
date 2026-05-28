# V4 Roadmap — 트레이딩 의사결정 결함 종합 수정 계획

**작성일**: 2026-05-27
**근거**: 3 subagent 종합 리서치 + 외부 quant standard 결합 결과 (24개 결함 식별)
**전제 원칙** (CLAUDE.md):
- 동시 다발 수정 금지. 한 번에 한 가지 변경, 1~2주 검증, 다음.
- 데이터 없이 정책 건드리지 않는다 (백테스트-라이브 parity 보장).
- 게이트 완화로 배포하지 않는다 (V2.2 −6.91% 교훈).
- 한 lever당 단독 backtest 검증 후 deploy.

**Baseline (2026-05-27 측정)**:
- Backtest: Sharpe **2.92** / Return +23.77% / MDD 4.98% / Win 63% / PF 2.88 / 52 trades
- Paper (4/11~5/22, 6주): +0.85% — 표본 부족으로 결론 보류

---

## 카테고리 A — 단기 적용 가능 (한 세션, 각 backtest 검증)

각 항목은 단독 변경 → backtest → baseline 대비 효과 측정 → 통과 시 적용, 악화 시 revert.

### A1. Trailing stop 도입 ⭐ P0
**문제**: BT Top 5 trade (MDB +21.97%, INTC +10.70% 등) 모두 1일 만에 `no_fresh_signal` exit. winner riding 효과 0. profit_take time decay (5% → 1.5%)가 winner 강제 청산.

**구현** (`v3/strategy/exit_rules.py`):
- 진입 후 peak price 추적
- 진입 +3% 도달 후, peak 대비 -2% drop 시 trigger
- conditional veto와 무관 (무조건 청산 — 손실 컷 보호)
- priority: `dynamic_stop_mae` 다음, `profit_take` 이전

**예상 효과**: BT Sharpe 2.92 → 3.5+? (Top trade들이 +10% 이상 hold 시) — 데이터로 검증
**위험**: trailing stop trigger 빈도 ↑ → 거래 빈도 ↑ → cost drag ↑
**검증**: backtest 1회

### A2. Cost 0.1% → 0.25% recalibration ⭐ P0
**문제**: subagent C 분석: NASDAQ 실제 spread + impact 0.15~0.35%. config 0.1%는 underestimate. BT 23.77% 중 ~1%p inflation 가능.

**구현** (`v3/config/v3_config.yaml:107-108`):
```yaml
us_roundtrip: 0.0025   # was 0.001 — realistic spread + impact
```
→ alpha gate (`cost × 1.75`)는 자동으로 0.00175 → 0.004375로 강화됨 (선별성 ↑)

**예상 효과**: BT Sharpe 2.92 → 2.5? (cost drag 추가) — paper-real gap 줄임
**위험**: 진입 빈도 ↓ (gate 강화로) → 거래 0인 달 가능
**검증**: backtest 1회 + paper realism 비교

### A3. monthly_trades 5 → 10 🟡 P0
**문제**: 5/25~5/27 paper 3 거래일 entries=0 직접 측정 (rejections={monthly_trades: 34/39}). 직접 입증된 손실.

**구현** (`v3/config/v3_config.yaml:95`):
```yaml
max_trades_per_month: 10   # was 5
```

**예상 효과**: BT 영향 0 (BT는 4.3/월), paper 진입 차단 해소
**위험**: 매우 낮음 (알파 게이트가 quality 통제)
**검증**: backtest로 변화 없음 확인 (sanity)

### A4. dynamic_stop_mae 로직 수정 🔴 P1
**문제**: subagent A: MAE -3% 트리거 후 회복 +0.5% 상태에서도 stop 안 풀림. 최악 이후 회복 시 추가 수익 기회 상실.

**구현** (`v3/strategy/exit_rules.py:81-86`):
- MAE -3% 트리거 발생 → tightened stop 활성화
- 단 현재가 > entry × 1.01 시 tightened stop 해제 (회복 신호)
- 또는 trailing peak로 재정의 (A1과 통합)

**예상 효과**: stop 평균 손실 -5.09% → -3% 수준
**위험**: false positive 회복 (-3% 후 -8% 가는 케이스 보호 약화)
**검증**: backtest, stop 발동 분포 분석

### A5. Correlation 동적 측정 🟡 P1
**문제**: subagent B: `sizing.py:89` default correlation 0.3 고정. 실제 NASDAQ tech sector 0.7+. VaR 과소 추정.

**구현** (`v3/strategy/sizing.py`):
- `estimate_correlation` 호출을 항상 시도 (이미 있는 함수, rolling 60d)
- default 0.3은 데이터 부족 시 fallback만
- portfolio risk 계산에 실제 correlation 반영

**예상 효과**: 다종목 진입 시 사이즈 적절히 감소 (현재는 max_positions=1이라 영향 작음)
**위험**: 낮음
**검증**: backtest, max_positions=3으로 시뮬

### A6. 환율 동적 반영 🟡 P1
**문제**: subagent B: `paper_broker.py:26` `usd_krw = 1400.0` 고정. 실제 1380~1420 변동. paper PnL ±2%p 오차.

**구현** (`v3/execution/paper_broker.py`):
- yfinance `KRW=X` ticker로 실시간 환율 fetch
- fetch 실패 시 last known 또는 1400 fallback
- 진입 시 환율 기록, 청산 시 환율로 PnL 계산

**예상 효과**: paper PnL 정확도 ±2%p 개선
**위험**: yfinance FX 지연/실패 가능
**검증**: 단위 테스트 + paper 결과 환율 변환 검증

### A7. Alpha gate cost × 1.75 → × 3.0 🔴 P1
**문제**: subagent A: gate 0.00175은 비용 회수 1.75배만 요구. "확신" 측정 아님. 매 세션 99 종목 중 30~60개 통과 — 선별성 매우 낮음.

**구현** (`v3/strategy/opportunity.py`):
```python
gate_multiplier = 3.0   # was 1.75
```
→ gate = 0.001 × 3.0 = 0.003 (NASDAQ 기준)

**예상 효과**: BT 진입 빈도 ↓ (52 → 20~30?), Sharpe 변화 (개선 또는 악화)
**위험**: 거래 너무 적어 자본 idle
**검증**: backtest, win rate / Sharpe 변화

### A8. MDD peak_value 초기화 로직 검증 + fix ⚠ P2
**문제**: subagent B 추정 (코드 직접 미검증): `peak_value` 초기화 없음 가능. 첫 loss 후 peak=100M 고정 → MDD 과대 계산.

**구현** (`v3/strategy/risk_manager.py`):
- `peak_value` 명시적 초기화 (account creation 시점 또는 매월 1일 reset)
- 코드 검증 후 필요시 fix

**예상 효과**: MDD 정확성, circuit breaker false trigger 감소
**위험**: 낮음
**검증**: 단위 테스트 + 시나리오 테스트

**카테고리 A 누적 예상 시간**: 1~2 세션 (각 1~3시간)

---

## 카테고리 B — 별도 세션 (각 며칠~1주)

### B1. Walk-forward backtest 구현 ✅ 완료 (2026-05-28)
**문제**: subagent C: 현재 single train/test split, 1년만 backtest. Lopez de Prado CPCV 미적용. Sharpe 2.92 robustness 미검증.

**구현 완료**:
- `v3/scripts/run_walk_forward.py` 신설 (CLI)
- `v3/backtest/walk_forward.py` 정비: `max_folds` 인자 추가, 구 API
  (`engine.entry_filter`) 제거 — Phase 2 재설계 후 SignalGenerator 내부로 이동
- 252d train / 63d test / 63d step rolling

**결과 (8 folds × 30 epochs, 2021-04 ~ 2024-08)**:
- Avg return per fold (3개월): **+0.98%**
- Std return: 1.34%
- Sharpe: 0 (trades 16건 noise, std 의미 없음)
- Profitable folds: **5/8 = 62.5%**
- Avg trades / fold: 2.0 (총 16)
- 연 외삽: **+4%/년**

**핵심 진단**: Baseline Sharpe 4.03 (Return +41%/년)은 **over-fit 강한 신호**.
- Baseline: 단일 train (3.5년) + 단일 test (1년) + 60 epochs
- WF: 8 fold × train 1년 + test 3개월 + 30 epochs
- WF의 +4%/년이 **paper +7%/년 외삽 (6주)에 훨씬 가까움**
- 즉 진짜 expected performance는 baseline의 10분의 1 수준

**Limitation 인정**:
1. WF train period 짧음 (1년 vs baseline 3.5년)
2. Epochs 30 vs baseline 60 (학습 quality 차이)
3. 16 trades = noise 표본
4. 공정 비교는 train=1000d + epochs=60 (4시간+) 필요

**결론**: 알파 자체는 존재 (5/8 profitable), 단 Sharpe 4.03만큼 강하지는
않음. Paper +0.85% (6주)는 noise 범위 내 정상. **장기 paper 관찰 필수**.

**남은 구현 (별도 세션)**:
- 252d train / 63d test rolling window
- 각 fold에서 VolTransformer + alpha_weights 재학습
- 결과 aggregate (Sharpe distribution, MDD distribution)

**소요**: 1주 (engine 재작성 + 학습 시간)
**예상 효과**: backtest 신뢰성 회복. Sharpe 분포 측정으로 overfitting 판정.

### B2. Multi-timeframe filter (주봉/4시간) 🟡 P2
**문제**: subagent A: 일봉만 사용. 주봉 약세장에서 일봉 반등 진입 위험.

**구현**:
- 주봉 trend 피처 추가 (5주 SMA, ADX)
- 진입 confirmation: 주봉 trend가 일봉과 일치할 때만
- 4시간봉은 monitor 루프 단축 후 추가

**소요**: 1주 (피처 + alpha 통합 + backtest)

### B3. Earnings 회피 진입 차단 🟡 P2
**문제**: earnings 전후 가격 변동성 극대. earnings_collector.py 이미 데이터 있음 (5/13 추가).

**구현**:
- EntryFilter에 earnings_proximity 체크 추가
- earnings ± 3 거래일 → 진입 차단
- 또는 sizing × 0.5

**소요**: 2일

### B4. Partial exit / Signal decay logic 정비 🟡 P2
**문제**: subagent A: `partial_exit.py` residual_edge 계산 불명, signal_decay 통합 안 됨.

**구현**:
- residual_edge 계산 명확화
- decay multiplier 적용 통합
- 단위 테스트 추가

**소요**: 3일

### B5. Monitor 5분 간격 + intraday peak tracking 🟡 P2
**문제**: subagent A: 15분 간격 — intraday peak 활용 0. Trailing stop이 intraday high를 못 잡음.

**구현**:
- monitor_interval_min 15 → 5
- intraday high water mark 별도 tracking
- A1 Trailing stop이 intraday peak 사용 가능

**소요**: 3일 (load test 포함)

### B6. Failure recovery 절차 자동화 🟡 P2
**문제**: subagent B: entry_history.json 비어있음 가능, systemd 재시작 시 state 복구 불완전.

**구현**:
- 재시작 시 state validation 추가
- inconsistent 시 alert + safe mode
- manual override 절차 문서화

**소요**: 3일

### B7. Conviction floor / multi-source veto 🟡 P2
**문제**: subagent A: conviction 곱셈만 사용, floor 없음. 약한 conviction에서도 진입.

**구현**:
- `conviction > 0.50` floor 추가
- 또는 multi-source veto (모든 directional alpha 양수 요구)

**소요**: 3일 + backtest

### B8. Stress test framework 🟡 P2
**문제**: subagent B: 2008/2020 같은 black swan 시나리오 테스트 없음.

**구현**:
- historical stress scenario (2008-09, 2020-03, 2018-12) replay
- VaR 99 / CVaR 측정
- portfolio_stop 발동 시뮬

**소요**: 1주

**카테고리 B 누적**: 6~8주 (1 세션당 1 항목)

---

## 카테고리 C — R&D / 장기 (1~3개월+)

### C1. Edge layer wrapper script ⭐ 보류 중
**문제**: 2026-05-27 세션 발견. `build_edge_dataset.py` CLI stub. wrapper 필요.

**상세**: `docs/FOLLOW_UPS.md` 1순위 1.0 항목 참조.

**소요**: 2주 (wrapper 200~400 line + 테스트 + calibration 검증)

### C2. 새 alpha R&D 🔴 본질적 개선
**문제**: subagent A + 5/13 IC 실험: trend +0.003, reversion -0.001 (둘 다 noise). volume_surprise +0.028만 marginal.

**5/28 시도 #1: AlphaPriceAcceleration**:
- 구현: 2차 미분 (recent 5d return − previous 5d return)
- 측정 (3년 lookback, panel 14,207): vanilla IC **+0.0052** (FAIL)
- Regime 최대: strong_bull +0.0562 (n=198), caution +0.0377
- **Verdict: REGIME_ONLY** (REJECT)
- Report: `v3/research/reports/experimental_alpha_ic_20260528_074444.json`

**5/28 시도 #2: AlphaBreakoutFade** ✅ PROMOTE_VANILLA:
- 초기 구현 AlphaBreakout (20d high 돌파 + 추세 추종): vanilla IC **−0.0200**
  → **부호 반대!** NASDAQ-100 대형주에서 20d high 돌파는 단기 exhaustion
  (pop & drop) 신호로 측정됨
- 부호 negation 후 AlphaBreakoutFade로 rename: vanilla IC **+0.0200** PASS
- Regime conditional:
  · **bear: +0.0815** (가장 강함, breakdown 후 회복)
  · bull: +0.0585, neutral: +0.0266
  · **caution: −0.0109** (paper 주력 regime, 약한 음수 — 효과 의문)
  · strong_bull: +0.0019
- **DEFAULT_DIRECTIONAL에 추가** (2026-05-28 c5c0853 다음 commit)
  · test_regression.py `test_default_directional_promoted_set` invariant 갱신
  · alpha_weights.json 5/13 학습이라 breakout_fade weight = 0 (production 0)
  · **6/1 06:00 KST 자동 retrain 후** regime별 weight 부여, production 효과 발생
- Report: `v3/research/reports/experimental_alpha_ic_20260528_080352.json`

**5/28 시도 #3, #4: AlphaRSIReversal + AlphaGapFade** (contrarian):
- RSIReversal (RSI14 과매수 fade): vanilla IC +0.0151 (FAIL, 근접) → REGIME_ONLY
  · bear +0.094, bull +0.050, strong_bull −0.156 (강세장 momentum), caution −0.002
- GapFade (overnight gap reversal): vanilla IC +0.0084 (FAIL) → REGIME_ONLY
  · strong_bull +0.228 (n=198 noise), bear +0.107, caution −0.047 (paper 주력 음수)
- 둘 다 DEFAULT 추가 안 함. EXPERIMENTAL candidate 보존.
- Report: `v3/research/reports/experimental_alpha_ic_20260528_125049.json`

**5/13 + 5/28 누적 (8 candidate 측정)**:
| Alpha | Vanilla IC | 결과 |
|-------|-----------:|------|
| volume_surprise | +0.030 | ✅ DEFAULT (5/13) |
| breakout_fade | +0.020 | ✅ DEFAULT (5/28) |
| vol_term | +0.020 | REGIME_ONLY |
| rsi_reversal | +0.015 | REGIME_ONLY |
| earnings_proximity | +0.009 | REGIME_ONLY |
| gap_fade | +0.008 | REGIME_ONLY |
| vol_predicted | +0.005 | REGIME_ONLY |
| price_acceleration | +0.005 | REGIME_ONLY |

**핵심 결론**: 8 candidate 중 2 PASS. **단일 가격/거래량 기반 alpha의 vanilla IC
ceiling ≈ 0.02~0.03**. 큰 개선은 **새 정보 소스 필요** (options flow, sentiment,
institutional positioning, multi-asset). 단순 OHLCV derivative로는 한계 명확.

#### 외부 데이터 edge 시도 (2026-05-28) — earnings surprise (PEAD)

A+B+D 배치 시도 → 데이터 가용성:
- **A. Earnings surprise**: ✅ yfinance get_earnings_dates에 EPS Estimate/Reported/
  Surprise(%) 5년 quarterly. collector `--with-surprise` 확장 + 재수집 (99/99).
- **B. Short interest**: ❌ yfinance snapshot만, historical 없음 → backtest IC 불가
- **D. Insider Form 4**: 🟡 SEC EDGAR 3년치 수집 큰 작업 → 별도 세션

**AlphaEarningsSurprise (PEAD) IC**: vanilla **+0.0007** (REJECT)
- bull +0.048, **bear −0.111**, caution −0.012, neutral −0.003
- panel 2505 (earnings window 15d 내 종목만, 전체의 18%)

**메타 통찰 — universe가 edge ceiling을 결정**:
- PEAD는 학술적으로 robust anomaly인데 NASDAQ-100에서 ~0
- 이유: PEAD는 **small-cap에서 강하고 large-cap에서 약함** (analyst coverage
  많아 earnings 정보 즉시 반영 = efficient)
- 종합: 단순 alpha (ceiling 0.02~0.03) + 외부 데이터 PEAD (~0)
  → **문제는 alpha가 아니라 universe**. NASDAQ-100은 세계에서 가장 efficient한
  large-cap 집합. 어떤 alpha든 ceiling 낮음.

**철학 재검토 결론 — 새 edge는 alpha가 아닌 universe/시장에서**:
| 방향 | 근거 | 트레이드오프 |
|------|------|-------------|
| 비효율 universe (small/mid-cap, 신흥국) | PEAD/momentum/breakout 강해짐 | 유동성↓ 비용↑ 데이터 |
| vol-self trade (옵션 straddle) | 우리 IC 0.70 활용, direction 무관 | 옵션 인프라 (큰 작업) |
| 현 수준 수용 | WF +4%/년, paper +7%/년 | 페르소나 목표(월1%) 미달 |

B(short interest)/D(insider)도 large-cap efficiency로 비슷한 결과 예상.
**단일 alpha 추가 트랙 졸업 — universe 전환 또는 vol-self trade가 본질적 lever.**

#### Multi-horizon IC 실험 (2026-05-28) — forward_horizon 5d 유지 결론

alpha별 vanilla IC를 1d/3d/5d/10d horizon에서 측정:

| Alpha | 1d | 3d | 5d | 10d | 최적 |
|-------|---:|---:|---:|----:|:----:|
| volume_surprise | +0.009 | +0.029 | +0.030 | +0.018 | 3~5d |
| breakout_fade | +0.018 | +0.024 | +0.020 | +0.013 | 3d |
| rsi_reversal | +0.023 | +0.020 | +0.015 | +0.019 | 1d (1d만 PASS) |
| vol_term | +0.012 | +0.014 | +0.020 | +0.011 | 5d |

**발견 1**: alpha마다 최적 horizon 다름 (rsi 1d, breakout 3d, volume 5d).
**발견 2 (가설)**: backtest/paper avg hold = 1d인데 alpha는 5d로 학습 → mismatch.
1d로 학습하면 caution regime에서 reversion/breakout_fade가 음수→양수 전환.

**검증 (alpha weight 직접 비교, production write 차단)**:
- caution (paper 주력): 5d {trend .33, vol_surp .47} vs 1d {trend .40, vol_surp .40}
  → **거의 동일, production 효과 미미**
- 극단 regime (strong_bull/bear): 1d는 floor 균등 .25 → **변별력 손실**.
  5d는 strong_bull trend=.70로 명확
- bull/neutral에서 차이 크지만 paper 진입 빈도 낮음

**결론**: forward_horizon **5d 유지**. mismatch 가설은 흥미로웠으나 paper 주력
caution에서 weight 차이 미미 + 극단 regime 변별력 손실. 데이터로 "변경 안 함"
확인 — V3가 이미 5d로 적절히 calibrated. multi-horizon 발견은 미래
multi-horizon alpha 설계 참고용 보존.
- Reports: `ic_horizon1.json`, `ic_horizon3.json`, `ic_horizon10.json`

**다음 탐색 영역** (각 1~3개월):
- AlphaBreakout: 20d high 돌파 + 거래량 confirmation
- AlphaMACD: 12-26 cross + histogram momentum
- AlphaRelativeStrength vs SPY (sector neutral) — SPY 데이터 추가 필요
- Multi-timeframe alpha (주봉/4시간봉 confirmation, 별도 데이터)
- Options flow alpha (P/C ratio, gamma exposure) — 외부 데이터 필수
- Sentiment alpha (news NLP, NAAIM, Fear & Greed) — 외부 데이터
- Cross-asset alpha (bond → equity flow) — macro 확장
- Insider trading (Form 4), Institutional 13F — 외부 데이터

**핵심 통찰**: 시스템이 작은 edge (volume_surprise 0.03)로 작동 중. **큰 개선은 새 정보 소스 필요** — 단순 가격/거래량 derivative로는 ceiling 확인됨.

**소요**: 각 alpha당 1~3개월 (데이터 수집 + IC 측정 + production)

### C3. Survivorship-free historical universe 🔴 P1
**문제**: subagent C: 현재 99 NASDAQ을 5년 retroactive 적용. delisted 종목 누락. BT inflation +5~10%.

**구현**:
- Historical NASDAQ-100 membership 데이터 (Bloomberg / S&P Capital IQ)
- delisting dates 매핑
- build_edge_dataset.py `historical_universe` callable 활용
- bias-adjusted backtest 재실행

**소요**: 2주 (데이터 acquisition + integration)

### C4. Hedging mechanism (옵션 / short) 🟡 P3
**문제**: subagent B: long-only NASDAQ, systemic risk full exposure. SPY put hedging 가능.

**구현**:
- 옵션 pricing 모듈 (Black-Scholes)
- regime caution/bear 시 자동 SPY put 매수
- delta-neutral / partial hedge

**소요**: 1개월 (옵션 chain 데이터 + pricing + execution)

### C5. Multi-timeframe alpha 전체 재설계 🔴 P3
**문제**: 현재 일봉 cross-sectional. 시간 horizon diversification 없음.

**구현**:
- Time-series alpha (1d / 3d / 5d / 10d horizon)
- 각 horizon 별 IC 측정 → 최강 horizon으로 OpportunityScorer 재학습
- regime-conditional horizon selection

**소요**: 2개월

### C6. 양도세 / 환위험 통합 모듈 🟢 P3
**문제**: subagent B: 한국 해외주식 양도세 22% (250만원 공제). 환위험 미반영. net return 실제로 더 낮음.

**구현**:
- 거래 시점마다 양도세 누적 추적
- 연말 최적화 (loss harvesting)
- 환위험 헤징 (KRW 선물 또는 ETF)

**소요**: 1주 (계산 모듈) + 옵션 헤징은 별도

### C7. Tax-loss harvesting + portfolio rebalance 🟢 P3
**문제**: 양도세 최적화 위해 12월 loss-making position 의도적 청산.

**구현**:
- 연말 손실 position 자동 청산 (wash sale rule 회피)
- 30일 후 재진입 옵션

**소요**: 2주

### C8. International equity 확장 🟢 P3
**문제**: NASDAQ만 → diversification 0. 유럽 / 일본 / 신흥국 추가 가능.

**구현**:
- Universe 확장 (EAFE, EM ETF)
- 시차 처리 (각 시장 close 시간 다름)
- 환위험 (multi-currency)

**소요**: 1개월

**카테고리 C 누적**: 3~6개월

---

## 진행 순서 권장

### Phase 1 (이번~다음 세션, 2주 이내)
1. **A1 Trailing stop** ← 가장 큰 lever
2. **A2 Cost 0.25%**
3. **A3 monthly_trades 10**
4. **A4 dynamic_stop_mae 수정**

각 단독 적용 + backtest 검증. 누적 Sharpe / MDD 변화 측정.

### Phase 2 (다음 1개월, 1주에 1~2 항목)
5. A5 Correlation 동적
6. A6 환율 동적
7. A7 Alpha gate × 3.0
8. A8 MDD peak 검증
9. B1 Walk-forward (가장 중요한 B)
10. B3 Earnings 차단
11. B7 Conviction floor

### Phase 3 (1~3개월, R&D 영역)
12. C1 Edge layer wrapper
13. C3 Survivorship correction
14. C2 새 alpha R&D (option flow 우선)

### Phase 4 (3~6개월+)
15. C4 Hedging
16. C5 Multi-timeframe alpha 재설계
17. C6/C7 Tax 통합
18. C8 International 확장

---

## 검증 절차 (각 lever 적용 시 공통)

1. **코드 변경 + 회귀 테스트** (`pytest v3/tests/`, 598+ tests)
2. **Backtest 실행 + Baseline 비교** (Sharpe / Return / MDD / PF / Win)
3. **악화 시 즉시 revert** (오늘 Concentrated mode 사고 학습)
4. **개선 시 paper deploy + 1~2주 관찰**
5. **paper 결과 확인 후 다음 lever**

---

## 의사결정 게이트 — 각 lever 적용/유지 기준

| Metric | Baseline | 적용 가능 임계 | 즉시 revert |
|--------|---------:|--------------:|------------:|
| Backtest Sharpe | 2.92 | ≥ 2.0 | < 1.5 |
| Return | +23.77% | ≥ +15% | < +5% |
| MDD | 4.98% | ≤ 7% | > 10% |
| Win Rate | 63% | ≥ 55% | < 45% |
| PF | 2.88 | ≥ 2.0 | < 1.5 |
| 거래 빈도 | 52/년 | ≥ 30/년 | < 20/년 |

위 6개 중 1개라도 "즉시 revert" 임계 도달 → 변경 제거.

---

## 위험 매트릭스

| 위험 | 확률 | 영향 | 대응 |
|------|:----:|:----:|------|
| 변경 후 backtest 악화 | 중 | 큰 | 즉시 revert, 다음 lever 시도 |
| paper에서 backtest와 다른 결과 | 큰 | 큰 | survivorship + cost recalibration |
| 동시 다발 수정으로 효과 분리 불가 | (이미 1회 발생) | 큰 | **금지 — 순차만** |
| Edge layer 재활성 같은 silent failure | 중 | 큰 | calibration 검증 후만 |
| paper 손실 누적 −5% 도달 | 낮 | 큰 | 자동 rollback timer + 수동 점검 |

---

## 다음 세션 시작 시 참고

1. 이 문서 + `docs/FOLLOW_UPS.md` 1순위 (Edge layer wrapper) + `CLAUDE.md`를 먼저 읽고 시작
2. 진행 중인 lever (in_progress task) 확인 후 다음 진행
3. 각 lever 완료 시 이 문서의 해당 섹션에 **결과 + 적용/revert 기록** 추가

---

**문서 갱신 정책**: 각 카테고리 A/B 항목 완료 시 결과 추가. C 영역은 분기별 재평가.
