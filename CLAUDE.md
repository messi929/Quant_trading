# Quant Trading System — CLAUDE.md

현재 운영 정책(V3.3 active, 12 features ON since 2026-05-10) + 개발 워크플로우만
다룬다. Phase별 이력은 `docs/CHANGELOG.md`, 후속 과제 추적은
`docs/FOLLOW_UPS.md` 참조.

> ⚠️ 2026-05-10 사용자 결정 — V3.3 12개 features 한 번에 ON (페르소나 무시,
> ROADMAP §5 주차별 promotion 건너뜀). LivePipeline ctx.actions 통합 +
> Edge layer 활성 (calibration validation FAIL 상태로 수동 publish).
> 자세한 내용 + 위험 + 관찰 + rollback 절차는 CHANGELOG "V3.3 전체 활성화"
> 섹션 참조. 페르소나 점수 측정은 1~2주 paper 데이터 누적 후.

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
- V3 재해석: "빠르게" = "thesis 깨지면 빠져나오기"이지 "시간 됐으니 팔기"가 아님
- 리스크 기반 청산(`vol_contraction`, `dynamic_stop_mae`, `portfolio_stop`)이
  실질적 구현체. 시간 기반은 재평가 트리거.

### 수익 목표

- **월 평균 일 1%** (월 +20%, 거래일 기준) — 매일 달성이 아닌 월 누적 기준
- 승률 55%+ × 손익비 1.5:1 이상 → 월 목표 달성 가능

---

## 매수 정책 (Buy Policy) — V3.2 단일 알파 게이트

### 알파 게이트 — OpportunityScorer (단일 수식)

```
direction(ticker)   = Σ_a  w_a(regime) · α_a(ticker)       ∈ [-0.1, 0.1]
conviction(ticker)  = Π_c  c_s(ticker)                      ∈ [0, 1]
opportunity(ticker) = direction · conviction                ∈ [-0.1, 0.1]

enter_if:  opportunity > cost × 1.75       # cost = 0.001 (NASDAQ 왕복 0.1%)
                                            # gate = 0.00175
```

- DirectionalAlpha: `trend`, `reversion` (signed return 예측)
- ConvictionSource: `vol` (VolTransformer, 확신도 modulate)
- Phase 2 원칙: VolTransformer는 risk model이지 alpha model 아님 (Two Sigma 컨벤션)

### 운영 제약 — EntryFilter (5가지)

```
✓ 포지션 한도 (max_positions = 3)
✓ 월 거래 한도 (dynamic, win_rate/mdd 반영) — Phase 25.1 이후 unique-ticker 카운트
✓ Circuit breaker halt
✓ 유동성 (daily_volume ≥ 5억)
✓ 섹터 집중 (≤ 2 per sector)
```

### Regime별 진입 — 게이트 고정, 가중치만 변경

regime은 **알파 가중치**만 결정. 게이트(`opportunity > cost × 1.75`) 자체는 고정.

```
strong_bull: {trend: 0.0,  reversion: 1.0}   # reversion IC=0.113 유일 양수
bull:        {trend: 0.5,  reversion: 0.5}   # uniform fallback
neutral:     {trend: 1.0,  reversion: 0.0}   # trend IC=0.028 유일 유의미
caution:     {trend: 0.5,  reversion: 0.5}   # uniform
bear:        position_scale=0 → CASH 단락
```

### 포지션 사이징 — 공분산 + Half-Kelly + position_scale 연속화

- 예측 vol 역수 × confidence × 상관관계 drag (Ledoit-Wolf 축소)
- 유동성 제약 (거래량 5% 이내)
- position_scale: 연속 score → smooth scale (Bridgewater discrete + Medallion continuous 하이브리드)

```
POSITION_SCALE_CURVE (piecewise linear):
  score 0.00 → 0.00  (bear = CASH)
  score 0.25 → 0.30
  score 0.40 → 0.60
  score 0.55 → 0.90
  score 0.75 → 1.10
  score 1.00 → 1.20
```

- 최대 단일 종목: `max_single_weight = 0.40`
- 최소 단일 종목: `min_position_weight = 0.15` (sizer floor, Phase 25.2)
  · caution 최저 scale 0.35에서도 5M 통과 보장 (0.15 × 0.35 = 5.25M)
  · bull 1종목 13.5%, 3종목 균등 40.5% — "1~3종목 집중" 부합
- 최소 거래 금액: 500만원 (이하는 의미 없음)
- long-only (NASDAQ, 공매도 미사용)

---

## 매도 정책 (Sell Policy) — V3.2.1 conditional veto

### 청산 4가지 — 시간/리스크 분리

```
1. 시간감쇠 이익실현 (profit_take): Day1 +5% → Day5 +1.5%
   → opportunity(ticker) > cost × 1.75 이면 **유지** (veto + 로그)
   → 그 외 청산
2. 보유 만기 (max_hold): 5일
   → opportunity 재평가 후 동일 룰 적용
3. Vol 수축 (vol_contraction): 진입 vol의 70% 이하 (3일 지속) → 무조건 청산
4. 포트폴리오 일간 -1.0~2.0% (배포 비례) → 전 포지션 무조건 청산
```

### 핵심 원칙

- **시간 기반 청산**(profit_take, max_hold)은 **opportunity 재평가 트리거**일 뿐.
  Phase 2의 진입 수식(`opportunity > cost × k`)을 유지 판단에도 재사용 → 단일 기준.
- **리스크 기반 청산**(vol_contraction, dynamic_stop_mae, portfolio_stop)은
  **veto 금지** (무조건 체결). 리스크 신호는 시간 경과와 무관.
- 승자 자르기 방지 = 손절만큼 중요한 edge 보존 메커니즘.

### 진입 제한 (Medallion alignment)

- 고스트 포지션 제거: 매도 3회 연속 실패 → open_positions에서 강제 제거
- 멀티데이 냉각기: 5일 내 2회 이상 손실 종목 → 진입 차단
- 연속 진입 제한: 동일 종목 3일 연속 진입 불가 (신호 정체 방지)
- 포지션 영속화: open_positions.json으로 서비스 재시작 시 복구

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
- Signal decay half-life: **6시간**
- 스프레드 안정화 후 진입 (개장 직후 회피)

### KR 스케줄 (V3, NASDAQ 기준 일부 활용)

```
06:00  데이터 수집 (yfinance)
06:10  신호 생성 (VolTransformer → SignalGenerator)
09:30  매수 (스프레드 안정화 후, 확신 있는 1~3 종목)
09:35~15:15  모니터 (15분 간격, conditional TP / max_hold / 리스크 청산)
15:20  세션 종료 (만료 포지션만 청산, 나머지 오버나이트 보유)
16:00  EOD 성과 기록
```

### US 스케줄

```
22:00  US 신호 재생성 (06:10 신호 재사용 금지)
23:40  매수 (1~2 종목)
23:45~04:30  모니터 (15분 간격)
04:30  청산
```

---

## 모델 원칙 (Model Principles)

### 알파 분류 체계 (Two Sigma/AQR 컨벤션)

| 축 | 역할 | 출력 범위 | 현재 구성 |
|----|------|----------|----------|
| **DirectionalAlpha** | 수익률 예측 (signed) | [-0.1, 0.1] | `trend`, `reversion` |
| **ConvictionSource** | 확신도 예측 (unsigned) | [0, 1] | `vol` (VolTransformer) |

- **VolTransformer는 Risk model**이지 Alpha model이 아니다.
- 변동성 팽창은 "크기 예측"이지 "방향 예측"이 아니므로 signed return과 IC ≈ 0.
- 직접 수익률 예측에 합치 않고, **다른 알파의 확신도를 modulate**.

### 알파 가중치 학습 정책

- **주기**: 매월 1일 재학습 (분기는 regime shift 반영 느림)
- **Lookback**: 3년 (더 길면 구체제 편향, 짧으면 noise)
- **파일**: `v3/config/alpha_weights.json` (latest) +
  `v3/config/alpha_weights_history/alpha_weights_YYYY-MM.json`
- **수동 실행**: `python v3/backtest/alpha_weight_trainer.py --lookback-years 3`

#### 3-step bootstrap

```
A. Vanilla IC — 각 directional alpha의 전체 기간 IC (MIN_VANILLA_IC = 0.02)
B. Regime 분류 — 7개 macro feature stand-alone IC로 composite weights 산출
C. Conditional IC — 각 regime 내 alpha × return IC, max(IC - 0.02, 0) shrinkage
```

### 알파가 없으면 거래하지 않는다

- VolTransformer dir_acc < 55% → 거래 중단, 모델 개선 우선
- Vol IC < 0.30 → 라이브 배포 금지
- 백테스트 Sharpe < 1.0 → 라이브 배포 금지
- 백테스트와 라이브 코드 경로가 동일해야 한다
- **게이트를 완화해서 배포하지 않는다** (V2.2 교훈: 7일간 -6.91%)

### 피처 원칙

- 모든 피처는 z-score 정규화 후 모델에 입력 (학습셋 통계 기준)
- `saved_models/normalizer_stats.json`에 통계 저장, 추론 시 로드
- 리턴 클리핑: ±30% (NASDAQ 일일 제한가)

---

## 백테스트-라이브 정합성 (Consistency Rule)

### 절대 규칙 — 단일 코드 경로

- 백테스트와 라이브 모두 동일한 `SignalGenerator`(v3/strategy/signal.py) 호출
- 동일한 `OpportunityScorer`, `EntryFilter`, `RegimeDetectorV2`
- 차이점은 오직 "데이터 공급 방식"(live=실시간, backtest=과거 replay)
- top_k, max_position, threshold 등 모든 파라미터 통일
- 매수+매도 양방향 슬리피지 적용

### V3 비용 파라미터

```yaml
NASDAQ:
  roundtrip: 0.001            # 0.1% (수수료 0.01%×2 + 슬리피지 0.1%×2)
  월 5회 × 0.1% = 월 0.5% = 연 6%

KRX (제외):
  roundtrip: 0.010            # 1.0% — vol 전략에 부적합 확인
```

---

## 리스크 관리 원칙 (Risk Management)

### Medallion 원칙 적용

- 승률 50.75%로도 수익 — **edge × 반복 × sizing**
- V3 사이징: 변동성 타겟팅 + 균등 + confidence tilt + Half-Kelly
- 거래량 5% 참여율 제한 (시장 충격 관리)
- bear 임계값: position_scale=0 (CASH 단락)

### 현금은 포지션이다

- 확신 없는 날 현금 100%는 **올바른 판단**이다
- "투자 안 하면 기회비용"이 아니라 "투자하면 비용"이다
- bear 레짐에서 현금 보유는 수익이다
- **게이트 미통과 모델로 거래하는 것은 확신이 아니라 도박이다**

---

## 코드 원칙 (Code Principles)

### 설정과 코드 분리

- 매직 넘버 금지: 모든 상수는 config YAML에서 로드
- TWAP fractions, cost rate, threshold 등 하드코딩 금지

### 중복 금지

- 신호 정규화: 단일 함수 (`v3/strategy/signal.py`)에서만 수행
- 백테스트/라이브 동일 함수 호출

### 검증 가능해야 한다

- 모든 거래에 이유(opportunity, regime, alpha_weights)가 로그에 남아야 한다
- 백테스트 결과 재현 가능해야 한다

### Immutable data flow

- frozen dataclass, pure functions, 뮤테이션 제거 (Phase 2 원칙)

---

## 운영 명령 (V3)

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

# 월 재학습 (수동, 정상은 systemd timer가 매월 1일 06:00 자동 실행)
PYTHONPATH=. $PYTHON v3/backtest/alpha_weight_trainer.py --lookback-years 3
```

### 자동화된 작업 (서버 systemd)

```
quant-trading-v3.service       — 라이브 데몬 (run_live.py --mode daemon)
alpha-retrain.timer            — 매월 1일 06:00 KST 자동 재학습 (Phase 25.2)
  → service: alpha-retrain.service
  → 로그: /var/log/alpha-retrain.log
  → artifact: v3/config/alpha_weights_history/alpha_weights_YYYY-MM.json
```

### 서버 배포

```bash
# 배포
bash deploy_v3.sh 77.42.78.9

# 상태 확인
ssh root@77.42.78.9 "systemctl status quant-trading-v3"

# 로그
ssh root@77.42.78.9 "tail -f /var/log/quant-v3.log"

# Paper trading 관찰 (일일)
ssh root@77.42.78.9 "tail -100 /opt/quant/v3/logs/v3_$(date +%Y-%m-%d).log | \
    grep -E 'Regime|Signal|Opportunity'"

# V2 복원 (긴급)
ssh root@77.42.78.9 "systemctl stop quant-trading-v3 && rm -rf /opt/quant && \
    mv /opt/quant_v2_backup /opt/quant && systemctl start quant-trading"
```

---

## 개발 워크플로우 — **검증 필수**

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

### 동시 다발 수정 금지

한 번에 한 가지 변경, 1~2주 검증, 다음. 데이터 없이 정책 건드리면
백테스트-라이브 parity 깨짐.

---

## V3.3 Profitability Engine — 12 features ON (2026-05-10 활성)

V3.3은 V3.2.1 위에 13개 신규 모듈 + `BookOptimizer` orchestrator를 추가해
"Vol Expansion Trader"를 "기대수익 기반 자본 재배분 엔진"으로 진화시킨다.

**현재 상태**: 12개 features 모두 ON (2026-05-10), 547 tests pass.
서버 deploy 완료, F2 hook이 12개 신규 활성 기록 (`feature_activations.jsonl`).
LivePipeline `ctx.actions` 소비 통합 (`TradingExecutor.execute_actions()` 5종
핸들러). Edge layer는 `v3/config/edge_calibration.json` 수동 publish로 활성
(validation FAIL 상태 — top-bottom -0.0001).

### 신규 모듈

| 영역 | 모듈 |
|------|------|
| Edge layer | `edge_calibrator`, `edge_engine`, `edge_tier` |
| Exit policies | `exit_thesis` (+ Conditional Veto), `signal_decay`, `partial_exit` |
| Capital | `allocation` (leverage 영구 거부), `pyramid` (winner-only), `rotation` |
| Diagnostics | `diagnostics` (NoTradeReason + TC), `execution_quality` |
| Orchestrator | `book_optimizer` (BacktestEngine + LivePipeline 모두 wiring) |
| Research | `build_edge_dataset`, `calibrate_edge`, `validate_edge` |
| Ablation | `ablation_specs/runner/report`, `promotion_decision` |
| CLI | `scripts/run_ablation_sweep.py` |

### V3.3 활성/비활성 제어 (현재 모두 ON)

`v3/config/v3_config.yaml` `features:` 섹션. ROADMAP §5의 주차별 promotion
일정은 2026-05-10 사용자 결정으로 우회 — 12개 모두 ON 됨. 롤백:

```yaml
features:
  # 한 줄씩 false 전환 후 deploy_v3_git.sh 77.42.78.9
  # Edge layer만 비활성: rm v3/config/edge_calibration.json + service restart
```

자동 rollback (`v33-rollback-check.timer`)은 매일 16:30 KST 1주 PnL -2% 시
flag OFF — 안전망. 자세한 rollback 절차 + 위험 분석은 CHANGELOG "V3.3 전체
활성화" 섹션 H 참조.

### V3.3 신규 운영 명령

```bash
# Calibration pipeline (server, 매월)
$PYTHON v3/research/build_edge_dataset.py --start ... --end ... --output ...
$PYTHON v3/research/calibrate_edge.py --panel ... --train-end ... --output ...
$PYTHON v3/research/validate_edge.py --panel ... --calibration ... --output ...

# Ablation sweep
$PYTHON v3/scripts/run_ablation_sweep.py --synthetic           # sanity
$PYTHON v3/scripts/run_ablation_sweep.py --panel <parquet>     # production

# Feature 활성화 후 paper 검증
ssh root@77.42.78.9 "tail -f /opt/quant/v3/saved_models/no_trade_logs/no_trade_*.jsonl"
ssh root@77.42.78.9 "tail -f /opt/quant/v3/saved_models/tc_history.jsonl"
```

### 페르소나 영향

| 원칙 | V3.2.1 | V3.3 처방 |
|------|--------|----------|
| 1. 확신 있을 때만 | 5/10 | EdgeCalibrator → 진짜 expected_return 측정. EdgeTier → S/A/B/C 등급화. |
| 2. 크게 | 3/10 (가장 심각) | AllocationEngine net_edge / risk sizing + Pyramid winner add-on. Phase 25.2 sizer floor 0.15와 시너지. |
| 3. 빠르게 | 8/10 | Conditional Veto 정상화 (FOLLOW_UPS 1순위) + ExitThesis 4-way (HOLD/REDUCE/ROTATE/EXIT). |

### 참고 — V3.3 docs

- `docs/V3.3_DESIGN.md` — 13 모듈 책임·데이터 흐름
- `docs/V3.3_ROADMAP.md` — 5단계 일정·합격 기준
- `docs/V3.3_INTERFACES.md` — 함수 시그니처·protocol·CLI
- `docs/V3.3_AB_PLAN.md` — A1~A9+Full ablation 실험
- `docs/V3.3_CHECKLIST.md` — PR별 작업 체크리스트

---

## 참고 문서

- **이력**: `docs/CHANGELOG.md` — Phase 22~26 (V3.3) 전체 변경 narrative
- **후속 과제**: `docs/FOLLOW_UPS.md` — 페르소나 정합성 점검 + active 보류 항목
- **V3 아키텍처**: `memory/v3_architecture.md`
- **Phase 2 재설계 상세**: `memory/phase2_plan_detailed.md`
- **V3.3 코어 thesis**: `docs/CORE.md` — 토론용 self-contained reference
