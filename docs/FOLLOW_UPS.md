# 후속 과제 (Follow-ups)

CLAUDE.md 정책 본문에서 분리된 active tracking. 페르소나 정합성 점검(2026-04-21) +
Phase 25.1 옵션 C 적용 후(2026-05-03) 잔존 항목 + V3.3 부분 활성 후 신규 항목
(2026-05-13).

**철칙**: 동시 다발 수정 금지. 한 번에 한 가지 변경, 1~2주 검증, 다음.

---

## 페르소나 원칙 점수 (2026-05-13 update)

"확신 있을 때만, 크게, 빠르게" 3대 원칙 기준 시스템 정합성 평가.

| 원칙 | 4/21 baseline | 5/13 (V3.3 부분 활성 후) | 상태 |
|------|:-------------:|:-----------------------:|------|
| 1. 확신 있을 때만 | 5/10 | 5/10 (`volume_surprise` +0.028 marginal) | 🔄 추가 알파 후보 탐색 |
| 2. 크게 | 3/10 | **7/10** (ABNB 자본 38% 실증, 5/13) | ✅ sizing 재해석 완료 |
| 3. 빠르게 | 8/10 | 8/10 (ExitThesis 16h staleness 유지) | ✅ Phase 3 ON 유지 |

**5/13 핵심 lever**: 원칙 ②"크게"가 3/10 → 7/10으로 4점 점프. Edge layer 재활성
또는 추가 알파 없이도 `position_scale` 의미 재정의(곱셈 → 포트폴리오 노출 한도)
만으로 6.5배 사이즈 확보. 페르소나 진단의 본질이 sizing 구조에 있었음.

### 원칙 3 "빠르게" 재해석 — 종결 (수정 없음)

초안에선 "max_hold veto로 무기한 보유 가능 → hard ceiling 10d 필요"
제안했으나 철회. 이유:

- V3의 "빠르게" = "thesis 깨지면 빠져나오기"이지 "시간 됐으니 팔기"가 아님
- Thesis 깨짐 신호(`vol_contraction`, `dynamic_stop_mae`, `portfolio_stop`)
  가 **무조건 청산** 담당 → 이미 구현됨
- 시간 기반(`profit_take`, `max_hold`)은 **opportunity 재평가 트리거** 역할
- `opportunity_map >8h stale guard`가 세션 간 fail-safe 담당
- 회전율 자체가 수익 공식 아님. Positive expectancy × 반복. 무의미한 churn =
  expectancy 음수
- V2(일단위 return 예측)의 "빠르게"를 V3(월 5회 vol 팽창)에 그대로 적용하면
  principle conflict

**결론**: 현재 conditional veto 정책 유지. 리스크 기반 청산이 "빠르게"의
실질적 구현체.

---

## 적용 순서 (2026-05-07 업데이트)

```
[완료]   2026-05-03  옵션 C: monthly cap → unique-ticker 카운트
[완료]   2026-05-07  Phase 25.2 — observation tools + sizer floor 0.05→0.15
                     · alpha-retrain systemd timer (monthly, 6/1 첫 실행)
                     · recommendation_log JSONL
                     · sizer floor 5/7 23:40 boundary 보정 (0.12 → 0.15)
[관찰중] 5/8~        통합 효과 1~2주 측정 (옵션 C + 사이즈 확대)
[다음]   미정        Conditional Veto 작동 수정 (스테일 16h or 정책 재설계)
[그다음] 미정        원칙 1 opp gate 이원화 (신규 strict, 유지 loose)
[저우선] 미정        변수명 정리, weekend monitor guard, 옵션 D
```

**Phase 25.2 진단 reframe**: 4/27~5/7 진입 0의 진짜 1차 단속점은 옵션 C가
풀려고 했던 monthly cap이 아니라 **사이저 floor**였음. 옵션 C는 정책으로
옳지만 cap이 단속점 아니었어서 효과 측정 거리 없음. Phase 25.2 사이즈 변경이
페르소나 원칙 2 (3/10)의 직접 처방.

---

## 신규 발견: Conditional Veto가 작동하지 않고 있음 (높은 우선순위)

4/11~5/3 paper 로그 전수 조사 결과, V3.2.1에서 도입한 conditional TP/max_hold
veto 정책이 **단 한 번도 발동된 적 없음**.

### 증상

```
TP 청산 5회 모두 다음 패턴:
  WARN  Opportunity cache stale (8.2~14h old) — dropping for TP veto
        (safer to let TP fire unconditionally)
  INFO  EXIT FANG: profit_take ret=+1.65% hold=4d
```

### 원인

- 8h staleness threshold < 14h (KR 09:30 generate ↔ US 23:40 generate)
- US 세션 시작 시점에 cache는 항상 stale
- 모든 시간 기반 청산이 "fire unconditionally" 경로로 빠짐
- **결과: V3.2.1 정책 자체가 죽은 코드**

### 영향 평가

- 4/21 ADI +9.02%, AMZN +3.30% 같은 큰 winner도 무차별 청산됨
- 다만 그 청산이 손해였는지 이득이었는지는 별개 평가 필요
- 정책이 의도대로 작동 안 한다는 사실 자체는 확정

### 검토할 개선안

1. **Staleness threshold 8h → 16h**: KR↔US 14h 간격 + 여유. 가장 단순.
2. **세션 시작 시점에 즉시 generate_signal 호출**: cache freshness 보장.
3. **정책 폐기**: veto 자체를 제거하고 무조건 청산으로 일관 (V2 회귀 위험 평가 필요).

**적용 시점**: 옵션 C 효과 1~2주 검증 후. 동시 변경 금지 원칙 준수.

**Phase 25.2 후 재평가**: 5/7 사이즈 변경 후 진입이 발생하면 자동으로
TP/max_hold 트리거 가능 → conditional veto 발동 여부 처음으로 측정 가능해짐.
즉 사이즈 수정이 이 항목의 사전 조건. 진입 누적 후 재진단.

---

## 신규 발견 (Phase 25.2): alpha_weights 재학습 자동화 ✅ 완료 (2026-05-07)

### 발견

- `alpha_weights.json` 4/18 19:40 freeze 후 미갱신
- `alpha_weights_history/`에 `alpha_weights_2026-04.json` 1개만 존재
- crontab 비어있음, systemd timer 없음
- CLAUDE.md/CHANGELOG에 "매월 1일 재학습" 정책 명시되었으나 수동 only

### 적용

`/etc/systemd/system/alpha-retrain.service` + `alpha-retrain.timer` 신설.
매월 1일 06:00 KST 자동 실행. 다음 실행: **2026-06-01 06:00 KST**.

5/1 catch-up은 미실행 (3년 lookback이라 20일 차이 영향 미미).

### 후속 모니터링

- 6/1 첫 자동 실행 결과 `/var/log/alpha-retrain.log`에서 확인
- `v3/config/alpha_weights_history/alpha_weights_2026-06.json` 생성 확인
- 학습 결과 `vanilla_ic` 변화 추이 (3년 rolling이라 1개월 단위 변화는 작지만 trend 추적)

---

## 보류 1: 원칙 1 "확신 있을 때만" — opp gate 이원화 (중기)

### 현재 실태 (변경 없음)

- `opp gate = cost × 1.75 = 0.00175` → "비용 회수 1.75배" ≠ "확신"
- 진입 gate와 유지 veto gate가 동일 수식
- Phase 2 "단일 수식" 원칙은 우아하나, 신규 진입 vs 지속 보유의 필요 conviction은 다름
- Regime=caution인데 게이트 불변 (가중치만 변경)
- 편입된 ADI/AMZN는 `confidence=0.5` 합성값 — 실제 학습 시점 conviction 아님

### 검토할 개선안 (옵션 C 효과 검증 후)

1. **유지 veto gate 상향**: 유지는 `cost × 3.0`, 진입보다 높은 bar
   - 이유: "계속 들고 있겠다" > "신규로 들어가겠다"의 conviction 요구
   - 이 항목은 위 "Conditional Veto 작동 안 함" 해결 후에 의미 있음
2. **Regime multiplier**: `caution → gate × 1.5`, `strong_bull → gate × 0.8`
3. **Reconciled 포지션의 veto 제외**: `confidence=0.5` 합성값 포지션은 max_hold 도달 시 강제 청산

**적용 시점**: 옵션 C 검증 + Conditional Veto 수정 이후.

---

## 보류 2: 원칙 2 "크게" — 사이즈 확대 ✅ Phase 25.2에서 적용 (2026-05-07)

### 적용 내역

5/4~5/7 13세션 연속 entries=0 데이터 분석 결과, 진입 부재의 1차 단속점이
**사이저 floor 0.05 + min_order_amount 5M의 수학적 미달**임이 확인됨
(regime 무관 — neutral에서도 동일). 우선순위 상향 후 즉시 적용.

채택안: **sizer.min_position_weight 0.05 → 0.15** (`v3/strategy/sizing.py:23`)

이유:
- 단일 lever, 단일 파일 변경
- 백테스트-라이브 parity 자동 유지 (양쪽 default 사용)
- caution 최저 scale 0.35에서도 5M 통과 (0.15 × 0.35 = 0.0525 = 5.32M)
- bull 1종목 13.5%, 3종목 균등 40.5% — 페르소나 "1~3종목 집중" 부합

미채택안 + 사유:
- SignalGenerator min_weight 0.02 → 0.10: cutoff만 변경, 통과 사이즈 무영향
- position_scale 곡선 상향: caution regime 의미 약화 (페르소나 충돌)
- 동적 capital deployment: 사이저 철학 대체, 변경 표면 큼

### 적용 과정의 boundary 보정

1차 0.12 적용 (5/7 23:33): caution scale 0.42 가정. 실제 5/7 23:40 세션에서
scale 0.411로 산출되어 PYPL weight 0.0493 = 4,997,999 KRW → 1,001 KRW
차이로 SKIP 발생.

2차 0.15 재적용 (5/7 23:47): caution 영역 관측 최저 scale 0.35 anchor.
회귀 테스트도 0.42 → 0.35로 강화. 5/8~ 모든 caution 세션에서 통과 보장.

### 4/11~5/3 데이터 (참고용 보존)

- Deployed capital 32% / Cash 68% — "집중 투자"와 거리 멀음
- 최대 포지션 FANG 가중치 0.18 (max_single_weight 0.40의 절반도 안 씀)
- ADI 5.9M, AMZN 7.0M, FANG 18~21M (1억 기준)
- ADI +9.02% 수익이 531k원에 그침 (사이즈 2배였다면 1M+)
- 4월 +1.39% 누적 수익률의 직접 원인 = 사이즈 부족

### 검증 단계 (5/8~5/14)

1주 paper 관찰. 측정 항목:
- 진입 발생 횟수 (목표: ≥3건/주)
- 평균 deploy 비율 (목표: ≥10% caution / ≥30% bull)
- 평균 실현 PnL/거래 (참고)
- vol-target 15% 위배 여부 (sanity 한도 30%)

결과 미흡 시 (예: caution 지속 + 진입 0 패턴 재현) 후속 검토:
- `min_position_weight` 추가 인상
- position_scale 곡선 동시 조정 (보류 2의 미채택안 #2 재검토)

---

## 보류 3: 변수명/메서드명 정리 (저우선순위)

옵션 C 적용 시 변경 표면 최소화를 위해 의미만 바꾸고 이름은 유지했음:

- `state.monthly_trades` (의미: unique tickers this month)
- `monthly_trade_count()` (의미: unique-ticker count)

### 검토할 정리 (영향 작은 정비 작업으로 별도 PR)

- 변수명 → `monthly_unique_tickers`
- 메서드명 → `monthly_unique_ticker_count()`
- 호출처 7곳 일괄 변경
- 회귀 테스트 동시 갱신

---

## 보류 4: 옵션 D (portfolio turnover) — 장기

진정한 회전율 기반 cap. 옵션 C로 충분치 않을 때 (예: 진짜 다양한 종목으로
churn하는 패턴이 등장할 때) 검토. 현재 우선순위 낮음.

---

## V3.3 F3 — BookOptimizer ctx.actions consumption ✅ 완료 (2026-05-09)

**Status**: 풀 통합 완료. BacktestEngine이 features.exit_thesis ON 시
ctx.actions의 EXIT/TRIM/ROTATE/ADD_NEW/ADD_TO_WINNER 모두 dispatch.

추가된 helpers (v3/backtest/engine.py):
- `_convert_to_position_states`: dict → PositionState
- `_check_triggers_v33`: ExitRules.check → ticker→trigger map
- `_handle_exit_action` / `_handle_trim_action` / `_handle_add_new_action`
  / `_handle_pyramid_action`
- `_process_book_actions`: multi-action dispatch + portfolio update

run() loop:
- features.exit_thesis OFF → V3.2.1 path (parity 보장, 547 tests)
- features.exit_thesis ON → V3.3 path (ctx.actions 사용)

테스트 (test_backtest_v33_integration.py, 21 신규):
- _convert_to_position_states (4)
- _check_triggers_v33 (3): no_exit / exit / missing_today
- _handle_exit_action (2): trade 생성, unknown 안전
- _handle_trim_action (2): partial trade, no-op when target == current
- _handle_add_new_action (3): 신규, 중복 차단, min_order 차단
- _handle_pyramid_action (3): weighted-avg, unknown 안전, target<current 안전
- _process_book_actions (4): KEEP, EXIT+ADD_NEW, NO_ACTION, BLOCKED

검증:
- pytest v3/tests/ → 547/547 통과 (이전 526 + F3 신규 21)
- features OFF default → 모든 기존 회귀 통과
- features ON 시 V3.3 path 모든 helper 동작

원본 issue (Evaluator):

---

(원본 항목 archive — 위 ✅ 완료 narrative 참조)

## V3.3 F3 — BookOptimizer ctx.actions consumption (Week 4 전 필수)

V3.3 evaluator (2026-05-09) 발견 critical issue. F1+F2는 즉시 fix됐으나
F3은 backtest engine 깊이 통합이 필요해 별도 PR로 deferred.

### 문제

BacktestEngine + LivePipeline은 `BookOptimizer.decide_with_context()`
호출 후 `ctx.signal`만 사용 (`ctx.actions`는 무시). features OFF 시
parity 100%이지만 features.exit_thesis / partial_exit / pyramid /
rotation 활성화 시 BookOptimizer가 emit한 EXIT/TRIM/ROTATE/ADD_TO_WINNER
가 무시 → backtest 결과가 baseline과 동일. ablation 데이터 무효.

### 영향 범위

- Paper Week 0 (진단 3개): 무영향. read-only.
- Paper Week 1~3 (edge_calibrator/engine/tier): 무영향. enrichment-only.
- **Paper Week 4 (exit_thesis): 영향 발생** — Conditional Veto 정상화
  효과가 backtest에서 측정 안 됨. ablation A4_exit_thesis 결과가
  A3_tier와 동일 → promotion 결정 잘못될 수 있음.
- Week 5~8 (decay/partial/allocation/pyramid/rotation): 동일 영향.

### Fix 작업 추정

1. BacktestEngine.run() loop 순서 재배치:
   - SignalGenerator + BookOptimizer 먼저 호출
   - ctx.signal로 entry 처리 (parity 유지)
   - features.exit_thesis ON 시 ctx.actions의 EXIT/TRIM/ROTATE를
     trade로 변환 (기존 _check_exits ExitRules 결과와 통합)
2. Position state ↔ PositionState 변환 helper
3. Trade unwind / cash 재배분 / daily_pnl 보정 로직
4. parity 회귀 테스트 (features OFF → V3.2.1 결과 100% 동일)
5. Conditional Veto 통합 회귀 (4/21 ADI scenario backtest 결과 KEEP)

추정 분량: ~400 line + ~250 line test. 별도 PR.

### Deadline

**Paper Week 4 (대략 6/12) 활성화 전 완료 필수**. Week 0 paper 활성화는
F3 없이 안전. Week 1~3까지도 무영향. Week 4 활성화 직전 별도 세션에서
처리.

### 현재 상태

- 코드: V3.3 전체 main 머지됨 (a69ea68 + 4f5834b 포함 526 tests pass)
- F1: conditional_veto dead key 제거 완료
- F2: feature_activations.jsonl 자동 기록 완료
- F3: deferred to "before Week 4 activation"
- Deploy: features OFF default라 즉시 가능 (parity 보장)

---

## 보류 5: 5/2~5/3 monitor의 weekend stale warning (cosmetic)

`monitor` 루프에 `weekday() < 5` 가드 없음 → 주말에도 매 15분 깨고 cache
staleness warning 누적. 동작에는 영향 없으나 로그 노이즈 + "신호 죽었나?"
오해 유발.

### 수정안

monitor에 weekday 가드 추가 또는 stale 경고를 주말에만 INFO로 demote.

**우선순위**: 낮음. 다른 변경과 동시 묶지 말 것.

---

## 관찰 기간 + 데이터 수집 (5/8부터 본격화)

### 관찰 도구 (Phase 25.2 신설)

- **`v3/saved_models/recommendation_log.jsonl`**: 매 세션 1줄 누적
  - 필드: regime, opp_gate, top_opportunities (top10), selected_positions,
    rejections, entries/exits, open_positions
  - 6개 관찰 포인트 4·5·6번을 직접 측정 가능
- **`/var/log/alpha-retrain.log`**: 6/1 자동 재학습 결과 (이후 매월)
- **`/opt/quant/v3/saved_models/paper_account.json`**: 누적 trade history
- **`/opt/quant/v3/logs/v3_YYYY-MM-DD.log`**: 일별 trace

### 관찰 포인트

1. Conditional veto가 실제 발동하는가 (TP/max_hold 트리거 시점 & 결과)
   — Phase 25.2 사이즈 변경 후 진입 발생해야 측정 가능
2. 8h staleness guard가 세션 간격(KR↔US=14h)에서 의도대로 작동하는가
3. Regime 전환(caution → neutral/bull) 시 position_scale이 자연스럽게 오르는가
4. opportunity 분포: gate(0.00175) 대비 실제 분포 (recommendation_log top10)
5. 포지션 크기 분포: floor 0.15 적용 후 실제 sized weight 추이
6. monthly cap 도달 빈도, churn에 의한 budget 잠식 비율

### 의사결정 게이트

위 6개 포인트 데이터 쌓인 후, 원칙 1·2 + Conditional Veto 개선안 중
**가장 영향 큰 1개씩만 선택 적용**. 동시 다발 수정 금지 (변경 효과 측정 불가).

**절대 금지**: 관찰 기간 중 임의 튜닝. "그냥 느낌으로 올려보자" 금지.
데이터 없이 정책 건드리면 백테스트-라이브 parity 깨짐.

---

## V3.3 전체 활성화 추적 (2026-05-10 ~)

> CHANGELOG.md "V3.3 전체 활성화" 참조. 사용자 "페르소나 무시, 즉시 활성"
> 결정으로 12개 features 동시 ON. 동시 다발 수정 금지 원칙 위배 — 효과
> 측정 분리 불가능. 자동 rollback (`v33-rollback-check.timer`)이 안전망.

### Active 추적 항목

| # | 항목 | 위치 | 빈도 |
|---|------|------|------|
| 1 | Validation FAIL 재현 | `research/reports/validation_YYYY-MM.md` | 매월 1일 07:00 |
| 2 | Edge layer ctx.actions 발생 빈도 | `/var/log/quant-v3-error.log` grep `EdgeTier\|net_edge` | 매 세션 |
| 3 | V3.3 action 발생 (EXIT/TRIM/ROTATE/ADD) | 위 로그 grep `V3.3 (EXIT\|TRIM\|ROTATE\|ADD)` | 매 세션 |
| 4 | Feature 활성 이력 | `feature_activations.jsonl` | startup |
| 5 | 1주 PnL -2% 자동 OFF 트리거 | `/var/log/v33-rollback.log` | 매일 16:30 |
| 6 | Daily diagnostic report | `research/reports/daily/` | 매일 16:00 |

### 1순위 — Calibration validation FAIL 추적

**현재 (2026-05-10)**: top-bottom -0.0001 → publish 차단됐으나 수동 publish.
- Decile 0 anomaly: 가장 음의 opportunity인데 fwd_5d mean +0.72%
- 원인 가설:
  - V3.2.1 `trend`/`reversion` alphas의 OOS 6개월 cross-sectional 약화
  - NASDAQ 99종목 표본 한계 (10K OOS rows)
  - sector="unknown" 단일 분류 — sector_map 미주입

**6/1 재실행 시 확인**:
- OOS 윈도우 자동 갱신 (5/10 → 11/10 → 5/11)
- 자연 해소 (OOS 다른 6개월 → top-bottom 차이 발생)?
- 같은 패턴 반복 시 → 알파 자체 재설계 필요 (V3.4 후보)

**조치 후보** (validation 계속 FAIL 시):
1. `validate_edge.py` 기준 완화 (top-bottom ≥ -0.002 허용) — 보수적 신호
   품질 저하
2. Sector map 주입 (build_edge_dataset에 sector_map 인자 활용) — 분류 개선
3. Alpha 재설계 (`reversion` 5d window 조정, `trend` shorter momentum) —
   V3.4 scope

**우선순위**: 6/1 데이터 보고 결정. 그 전 임의 튜닝 금지.

### 2순위 — LivePipeline ↔ Monitor exit 통합

**현 한계** (CHANGELOG §V3.3 활성화 G.5):
- generate_signal 시점 (KR 09:30 / US 23:40): V3.3 ctx.actions path
- 15분 monitor 루프: V3.2.1 ExitRules + executor.py inline veto

**증상**: ExitThesis "HOLD" 결정이 보유 중 monitor에서 무효. 예) HOLD
target이지만 monitor가 profit_take TP 발동 시 V3.2.1 inline veto만 체크
(8h stale guard).

**조치 시나리오**:
- A. monitor에서 ExitThesisEngine 직접 호출 (positions snapshot + last
  candidates)
- B. ctx.actions에 시간 만료 만들어서 next session generate_signal 때만
  의사결정 (현재 동작) — 단순하지만 reactive 늦음
- C. 통합 안 함 (monitor V3.2.1 유지) — 안정성 우선, 정책 효과 부분적

**우선순위**: live data 1~2주 누적 후 결정. ExitThesis trigger와
ExitRules trigger 충돌 케이스 발생하면 A로 진행.

### 3순위 — V3.3 features 효과 측정

**의사결정 게이트** (5/24 즈음 평가 가능):

| 측정 | V3.2.1 baseline | 1주 V3.3 paper 결과 | 판정 |
|------|----------------|---------------------|------|
| Avg deployed | ~60% | ? | ≥ 75% (allocation 효과) |
| 종목당 weight | floor 0.15 | ? | 큰 종목 증가 (sizer ↔ allocation) |
| Pyramid 발생 | 0 | ? | winner-only invariant 위배 0 |
| Rotation 발생 | 0 | ? | 월 cap 내 |
| TP 잘리기 (4/21 ADI 케이스) | 발생 | ? | conditional veto 16h fix 효과 |

데이터 부족 시 측정 보류. 주간 paper 결과 vs V3.2.1 동기간 baseline 비교.

### 절대 금지

- features 12개 ON 상태에서 한 개씩 OFF로 효과 분리 시도 (rollback timer
  triggered = 자연 OFF만)
- Calibration FAIL 무시 후 게이트 완화 — V2.2 교훈 (게이트 완화 → -6.91%)
- monitor 루프에 ExitThesis 통합 + 타 변경 동시 — 한 번에 하나
- "어차피 paper니까" 안전장치 비활성

---

## V3.3 부분 활성화 후 추적 (2026-05-13 ~)

5/11~12 4 거래일 entries=0 silent failure 분석 + 5/13 4 commits 안정화 이후
잔존 추적 항목. 자세한 narrative는 `docs/CHANGELOG.md` "V3.3 부분 활성화 +
sizing 재해석".

### ✅ 5/13 작업으로 closed된 항목

| 항목 | 해결 방식 | 검증 |
|------|----------|------|
| Conditional Veto 16h staleness | Phase 3 `exit_thesis` ON 유지 → ExitThesisEngine 16h + signal_refresh 적용 | `test_exit_thesis.py` invariant |
| Calibration validation FAIL 추적 | Phase 2 OFF → Edge layer 무력화 차단. 데이터 진단으로 **OpportunityScorer가 5d return alpha 아님 확정** (vol_predicted IC 0.007) | `experimental_alpha_ic_20260513_*.json` |
| 사이저 floor 곱셈 무력화 | `position_scale` 의미 재정의 (5/13 commit `5ffcae6`) | `test_regression.py` Bug 11 |
| `flush_diagnostics` 호출 누락 | `LivePipeline.run_session` + `BacktestEngine.run` try/finally (5/13 commit `fdb9eb0`) | `test_regression.py` Bug 12 (AST enforcement) |
| 원칙 ②"크게" 3/10 | sizing 재해석으로 7/10 (ABNB 자본 38% 실증) | paper 5/13 09:30 KR 세션 |

### 1순위 — Edge layer 재활성 조건 (현재 보류)

Phase 2 (`edge_calibrator / edge_engine / edge_tier / allocation`) + Phase 4
(`pyramid / rotation`) OFF 상태. 재활성 가능 조건:

1. **Calibration top-bottom 의미값 확보** — 현재 −0.0001 (noise). 목표 > 0.01
   (decile 9 평균 return − decile 0 평균 ≥ 1%).
2. **`validate_edge.py` PASS** — out-of-sample window에서 in-sample IC 일관성
   유지 확인.
3. **paper 1~2주 검증** — 부분 활성 환경에서 5/13 sizing 효과 누적 (사이즈
   확대가 sharpe / 손익비 유지하는지).

위 3개 동시 만족 시 Phase 2/4 단계적 ON 검토.

#### 1.0 (선행 과제) — Calibration build wrapper script 작성

2026-05-27 사용자 재활성 시도 중 발견:
- `v3/research/build_edge_dataset.py` CLI는 **stub** ("CLI is a stub.
  Production usage: 1. Run from server with v3/data/raw/ populated 2. Ensure
  v3/saved_models/vol_transformer_best.pt exists 3. Provide ohlcv_data,
  macro_pctl, vol_predictions to build_edge_dataset()")
- 즉 panel 생성을 위해 OHLCV 로드 + VolTransformer 전체 기간 추론 + macro
  percentile 계산을 직접 호출하는 Python wrapper가 필요
- alpha-retrain.timer는 `alpha_weight_trainer.py`만 자동화하지 Edge
  calibration pipeline은 자동화 미구현

**필요 작업** (별도 세션):
- `v3/research/run_calibration_pipeline.py` 신설 — end-to-end orchestrator:
  1. `Collector` 또는 OHLCV parquet 로드
  2. `MacroCollector` + `MacroFeatures` percentile 계산
  3. `VolTransformer` 로드 + walk-forward 추론 (lookback window 전체)
  4. `build_edge_dataset()` 함수 호출 + parquet 저장
  5. `calibrate_edge.py` 호출 또는 함수 직접 호출
  6. `validate_edge.py` 호출 + PASS/FAIL 보고
  7. PASS면 `/opt/quant/v3/config/edge_calibration.json` 갱신
- 회귀 테스트 추가 (`v3/tests/test_calibration_pipeline.py`에 wrapper 통합 케이스)
- alpha-retrain.timer와 별도로 `calibration-retrain.timer` 활용 또는 신설
  (CLAUDE.md "calibration-retrain.timer — 매월 1일 07:00 KST (Edge layer
  활성 대비, 현재 무의미)" 이미 등록되어 있음 — service script 작성)

**서버 calibration JSON 현재 상태** (2026-05-27 14:00 KST 확인):
- `/opt/quant/v3/config/edge_calibration.json` 84KB 존재 but
  `top_bottom_spread=None, train_end_date=MISSING, table_entries=0`
  → SF-1 (silent-failure) 100% 활성 상태. wrapper 완성 + 정상 calibration
  생성 전까지 Phase 2/4 ON 절대 금지.

**v3-evaluator SF-3 발견** — `v33-rollback-check.timer` 안전망 한계:
- `rollback.py:150` `max(eligible, key=activated_at)` 한 번에 1 feature만 OFF
- 4 features 동시 ON 시 완전 rollback에 **4 거래일 × -2% PnL 사이클** 필요
- wrapper 완성 후 재활성 시에도 단계적 ON (A → 1주 → B → 1주 → C ...) 필수
- 또는 `rollback.py` 수정: 같은 startup timestamp의 features 일괄 OFF

#### 1.1 단계 순서 (선행 1.0 완료 후) 단계 순서:
- A. `edge_calibrator` + `edge_engine` ON (Edge tier 분류 활성, allocation은 OFF)
- B. paper 1주 후 `edge_tier` ON
- C. paper 1주 후 `allocation` ON (sizing 가중치 override 시작)
- D. paper 2주 후 `pyramid` + `rotation` ON

선결 조건이 calibration 품질이라 단기 (1~2개월) 내 충족 어려움. **OpportunityScorer
자체가 5d return alpha 아니라는 데이터적 결론** — 단순 calibration 재학습으로
해결 안 될 수 있음. 새 알파 입력 필요 (아래 2순위).

### 2순위 — 추가 directional alpha 후보 promotion 조건

5/13 IC 실험에서 4 candidate 측정:

| Alpha | Vanilla IC | 최대 regime IC | 현재 상태 |
|-------|----------:|--------------:|----------|
| `volume_surprise` ✅ | +0.028 | caution +0.059 | DEFAULT 적용 |
| `vol_term` | +0.019 | caution +0.041 | EXPERIMENTAL 보류 |
| `earnings_proximity` | +0.012 | strong_bull +0.114 (n=99) / neutral +0.027 | EXPERIMENTAL 보류 |
| `vol_predicted` | +0.007 | strong_bull +0.061 (n=99) | EXPERIMENTAL 보류 |

**Promotion 조건** (다음 검토 시점):
- Vanilla IC ≥ MIN_VANILLA_IC (0.02) — 1회 IC 측정 통과
- 2~3개월 추가 panel data 누적 후 IC 시계열 robust 확인 — single point IC는
  noise일 수 있음
- `test_new_alphas.py` 재실행으로 verdict 안정성 확인

`vol_term`이 가장 가까움 (vanilla 0.019, threshold 0.001 미달). 6/13 panel
재측정 시 통과 가능성 있음.

### 3순위 — `volume_surprise` 실효성 검증 (5/13~5/27)

DEFAULT 추가됐으나 vanilla IC 0.028은 marginal. paper에서 실제 효과가
나타나는지 1~2주 누적 측정.

**관찰 포인트**:
- caution regime 진입 시 `volume_surprise weight 0.60` 적용 — 종목 선택 패턴
  변화 (이전: trend/reversion uniform 0.5 → 신규: volume-driven)
- 진입 종목의 거래량 ratio (vs 20d MA) 분포
- 진입 후 5일 return 분포 — IC 0.028 effect size 실측

**비교 baseline**: 4/11~5/8 paper (V3.2.1 sizing 곱셈 + trend/reversion only)
승률 71%·손익비 12:1. 5/13~5/27 (V3.3 부분 활성 + volume_surprise) 동일/유사
승률·손익비 유지 + 사이즈 6.5배 효과로 누적 수익률 5~6배 가능성 검증.

### 4순위 — OpportunityScorer 자체 재설계 (장기, 6/1 이후)

5/13 follow-up #2 연구가 확정한 진단:
- 기존 trend/reversion 둘 다 vanilla IC FAIL (+0.003 / −0.001)
- vol_predicted (VolConviction을 signed alpha로 변환) IC 0.007 — VolTransformer
  신호는 magnitude amplifier이지 directional alpha 아님
- 4 candidate 중 volume_surprise만 marginal pass — 단일 알파로는 IC 0.028이
  ceiling 가능성

**근본 한계 가설**: OpportunityScorer 수식 (`direction × conviction`)이 단일
종목 cross-sectional signed return을 예측하기에 본질적으로 부족. 검토할 방향:

A. **Multi-horizon alpha** — 1d / 3d / 5d / 10d 별 alpha 측정 → 가장 강한
horizon으로 OpportunityScorer 재학습 (현재 5d 고정)
B. **Time-series momentum** — cross-sectional 대신 ticker-self time-series IC
(현재 cross-sectional rank IC만 측정)
C. **External data alpha** — sentiment / options flow / institutional ownership.
가장 비용 큼. Tier C 후보 (Follow-up #2 옵션 4).
D. **VolConviction을 OpportunityScorer 곱셈에서 분리** — conviction을 sizer
input으로만 사용, opportunity = direction만으로 정의 (수식 단순화)

이 항목은 paper 1~2주 누적 + 6/1 calibration 재시도 결과 본 후에야 의미.

### 5순위 — `vol_term` 등 EXPERIMENTAL alpha 활용도

`vol_term`은 caution regime에서 +0.041로 보조 가능. 그러나 vanilla 미달 →
DEFAULT 추가 보류. 다음 옵션:
- 6월 panel 재측정 후 vanilla 통과하면 DEFAULT 추가
- regime-conditional weight 직접 부여 (`alpha_weights.json` 수동 편집) —
  단 자동 cron이 덮어쓰니 의미 없음
- `OpportunityScorer` 자체에 conditional alpha (regime별로 다른 알파 set
  사용) 도입 — 정책 변경 큼

대기 상태.

### 6순위 — `compute_directional` `vol_scores` forwarding latent trap

5/13 commit `ebdecc6` v3-evaluator 발견: `signal.py:122`에서 `vol_scores`를
forward 안 했음. 같은 commit에서 fix됨 (compute_directional에 vol_scores
kwarg 전달). 단 미래에 `AlphaVolPredicted` 같이 vol_scores 의존 알파가
DEFAULT_DIRECTIONAL에 promote되면 silent empty Series 위험. 현재 fix로 안전.

확인됨, 액션 없음.

### 7순위 — `live_pipeline` hold_days `+1` 미세 버그 — ✅ 해소 (5/21, `31cb85f`)

5/13 evaluator pre-existing 발견 (commit `4ced695` 도입): `_check_triggers_v33`
가 hold_days = `int(pos.get("hold_days", 0)) + 1` (counter 증가)로 계산하고
있는데 `_monitor_v33` / `executor.py:178` 는 calendar diff 사용. 트리거 1일
일찍 fire 가능 (예: day-4 profit_take boundary 도달).

영향: Phase 2/4 OFF라 dormant이었으나 Edge layer 재활성 시 pyramid/rotation/
exit_thesis 타이밍 오류 유발 가능.

**해소 (`31cb85f`, 5/21)**: 3곳(`_check_triggers_v33` + `_convert_to_position_states`
+ `_convert_one_position_state`)이 V3.3 BookOptimizer 입력으로 `hold_days + 1`을
넘기던 것을, 공유 staticmethod `_hold_days(pos, today)` (entry_date calendar diff,
canonical exit path와 동일)로 통일. 회귀 `TestV33HoldDaysCalendarBasis` 6건. Edge
layer 재활성 선결 조건(1순위)에서 제거됨.

---

## 다음 의사결정 게이트 (2026-05-27 즈음)

5/13 부분 활성 + sizing 재해석 + volume_surprise promotion 2주 paper 검증
결과로 다음 단계 결정:

| 측정 | 5/13 시점 baseline | 5/27 target | 미달 시 조치 |
|------|-------------------|-------------|--------------|
| 종목당 weight | 0.06 (이전) → 0.15~0.40 (현재) | 평균 ≥ 0.20 | sizing 추가 조정 |
| 진입 빈도 | 4/11~5/8 BUY 7회 / 한 달 | 5/13~5/27 BUY ≥ 4회 / 2주 | regime 트리거 점검 |
| 승률 | 71% (4월) | ≥ 65% | volume_surprise 효과 평가 |
| 손익비 | 12:1 (4월) | ≥ 2:1 | conditional veto 작동 점검 |
| MDD | < 4% (BT) | ≤ 5% | 사이즈 다시 조절 |
| 누적 수익률 | +1.4% (4월) | ≥ +3% (사이즈 6.5배 효과 일부 발현) | 진단 |

미달 시 단계적 rollback (`docs/V3.3_OPERATIONS.md` §4). 자동 rollback
(`v33-rollback-check.timer`) 1주 PnL −2% 시 features OFF 안전망 작동.

---

## V4 KOSDAQ 메인 엔진 보강 로드맵 (2026-05-30 ~)

> 사용자 전략 리뷰(2026-05-30): "KOSDAQ을 메인 엔진으로 두되, 알파를 더 찾기보다
> 체결·노출·섹터·상대강도 필터를 보강하는 단계." 큰 그림 동의. 단 제안된 5개 보강
> 항목은 **위험 프로파일이 정반대인 두 부류로 갈리며, 섞으면 위험**하다.
> KOSPI 3경로 기각 확정은 `docs/FINAL_STRATEGY.md` §4.1.

**철칙 (이 프로젝트의 진짜 엣지)**: V4 검증 이력은 lever 추가의 연속 기각이었다 —
trailing stop 기각(Sharpe 0.41→0.17), 단일 lb 튜닝=신기루, best-fit param=overfit,
**keeper는 vol-target 하나뿐**. 보강 항목의 합격 바는 "백테스트 숫자 개선"이 아니라
**a-priori 파라미터 + walk-forward 생존 + param plateau**. 대부분 탈락을 예상한다.

### 부류 분류 — A(전방 검증) vs B(새 fitted lever)

| # | 항목 | 부류 | 판정 |
|---|------|:----:|------|
| 1 | 체결품질 측정 | **A** | 최우선·정당. 단 LIVE 데이터로만 측정(백테스트 불가) |
| 5 | NASDAQ↔KOSDAQ 통합 상관 | **A** | legit 측정. 단 현재 live 히스토리 미겹침(아래) |
| 2 | KOSDAQ/KOSPI RS 필터 | B | KOSPI 재활용 아이디어 OK. overlay는 a-priori 1회 테스트, prior 낮음 |
| 4 | 단계 노출 0/50/100 | B | 동기(COVID whipsaw) real, 그러나 **표본 크래시 n=1 과적합 최대 위험** |
| 3 | sector/liquidity cap | B | 아래 effective-N 데이터로 부분 정당화 (soft cap만) |

- **A부류**: 측정이지 fitting이 아님 → overfit 위험 0, 가치 높음. 단 둘 다 **지금은 데이터가
  없다.** #1은 6/1+ live 필요. #5는 V3 paper(4/11~, 미국 밤) ↔ V4(6/1~, 한국 낮)
  히스토리가 안 겹쳐 백테스트 상관은 약한 proxy. 둘 다 paper 누적 후 측정.
- **B부류**: 같은 2014-26 창에 knob 추가 → 정확히 이 프로젝트가 반복 기각해온 함정.
  진입 전 paper가 알파의 *체결 생존*을 먼저 증명해야 의미.

### 데이터로 검증한 #3 — "top20 = 사실상 2~3 테마 베팅?"

`v3/research/kosdaq_concentration_probe.py` (production `ensemble_picks` 그대로 호출,
regime-on 78 바스켓, full-cycle). sector 라벨 없이 픽들의 실현 수익률 동조성으로 실효
분산 측정:

| 지표 | 값 |
|------|-----|
| pairwise 상관 ρ̄ | +0.14 (median 0.13, p90 0.29) |
| **실효 종목수 N_eff** | **7.1 / 명목 20** (median 5.9, **p10 3.1**) |
| 바스켓 수익 집중 | top1 18.7%, top3 41.2% (\|기여\|) |

**판정: 명목 20이 실효 ~7개 베팅.** "2~3"은 평균적으론 과장이나 **p10 꼬리(3.1)**에서
실제 발생 → 우려는 *real but moderate*. 
- avg case(7.1)는 **hard sector cap 정당화 못 함** (복잡도만 추가).
- 꼬리(3.1)는 **soft single-name/liquidity weight cap**을 *리스크 통제*로 지지
  (수익 최적화 아님). a-priori 후보: 단일종목 ≤ 7%, 거래대금 대비 주문금액 상한.
  "best 비중/섹터한도 찾기"는 B부류 함정 → 금지.

### 시퀀싱 (확정)

```
[지금~6/1]  모델에 아무것도 추가하지 않는다.
            역설적이지만 KOSDAQ 최선. 6/1 전 lever를 얹으면 #1(깨끗한 forward
            체결 검증)을 오염시킨다. 엣지는 알파가 아니라 추가를 거부하는 규율.
[6/1]       첫 자동 세션 관찰 (journalctl -u quant-v4-korea / quant-v4.log).
            정상 작동 확인.
[6/1+]      execution-quality 로거를 v4 live 경로에 보강 (테스트 동반).
            6/1 한 세션 돌려 logs 확인 후 적용 (가동 직전 live 경로 변경 금지).
[paper 3~6M] 체결 데이터 수집·측정: 신호종가 대비 체결가(슬리피지) / 09:05 체결률
            / 미체결 비율 / turnover / 종목별 PnL 기여 / 실전 MDD vs 백테스트.
            ★ 이게 모든 걸 결정한다. 알파 추가 아님.
[증명 후]   B부류 진입. 각각 a-priori + walk-forward, 탈락 예상하며 1개씩.
            soft single-name/liquidity cap(#3 꼬리 근거)만 리스크 통제로 선행 가능.
[자본 배분 전] #5 통합 상관 — V3·V4 둘 다 live 누적 후. "growth risk 동시 노출"
            (미 기술주 급락 ↔ 한 성장주 급락) 동반 손실일 비율 측정 필수.
```

### KOSPI 재활용 (#2 reframe) — 동의

KOSPI는 *독립 알파 엔진*으로 기각(§4.1)이지만 *regime/RS 지표*로 재활용은 다른·더
싼 용도라 동의. 단 RS overlay도 fitted lever → 120d 표준 a-priori 1회 테스트, walk-forward로만
판단. 절대 best-fit band 탐색 금지.

### 절대 금지

- 6/1 가동 전 live 경로(`v4/live/`, `v4/execution/`) 변경 — 첫 자동 세션 깨질 위험
- B부류 lever를 백테스트 숫자 최대화로 채택 (best-fit = mirage, V4 전체 교훈)
- COVID whipsaw 하나(n=1)에 맞춘 단계 노출 band 튜닝
- paper 체결 검증 전 "알파 더 찾기" — 현 단계는 alpha가 아니라 execution이 bottleneck
