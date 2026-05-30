# 전략 전체 리뷰 + 비판적 평가 (토론용)

**작성**: 2026-05-28 (V4 탐색 종료 직후)
**목적**: 현재 전략 전체를 한 문서로 정리 + Claude의 솔직한 "부족한 부분" 평가.
토론 기반 문서 — 결론이 아니라 논의 출발점.

---

## Part 1 — 현재 전략 전체

### 1.1 투자 철학

> "확신 있을 때만, 크게, 빠르게" — 작은 edge × 반복 × Kelly sizing

- 확신 없으면 현금 (0~100% 현금 허용, bear → CASH)
- 1~3종목 집중 (소자본 분산은 수수료에 잠식)
- thesis 깨지면 빠르게 청산 (시간이 아니라 리스크 기반)
- 목표: 월 평균 일 1% (연 ~20%)

### 1.2 시스템 파이프라인

```
데이터(yfinance NASDAQ-100)
  → VolTransformer (vol expansion 예측, IC 0.70)         [conviction]
  → 규칙 alpha (trend/reversion/volume_surprise/breakout_fade)  [direction, IC 0.02~0.03]
  → OpportunityScorer: opportunity = direction × conviction
  → 진입 게이트: opportunity > cost × 1.75 (= 0.00175)
  → RegimeDetectorV2 (8 macro features → 5단계 regime → alpha weight + position_scale)
  → VolTargetSizer (inv-vol × Half-Kelly × regime scale)
  → EntryFilter (max_positions, monthly cap, sector, liquidity)
  → PaperBroker (KIS sandbox 옵션 미지원 → yfinance 가격 시뮬)
  → 청산: profit_take/max_hold (conditional veto) + vol_contraction/dynamic_stop_mae/portfolio_stop (무조건)
```

### 1.3 핵심 수치 (2026-05-28)

| 지표 | 값 | 출처 |
|------|-----|------|
| Backtest Sharpe | 4.03 (A3 적용 후) | **over-fit 의심** |
| Walk-forward | +4%/년, 5/8 fold profitable | 진짜 expected에 가까움 |
| Paper (6주) | +0.85% (≈+7%/년 외삽) | 표본 부족 |
| direction alpha IC | 0.02~0.03 | volume_surprise, breakout_fade만 PASS |
| vol 예측 IC | 0.70 | 가장 강한 신호 |

### 1.4 현재 deployed (commit 9f285ad)

A3 monthly_trades 10 + A1 trailing + A6 환율 동적 + A10 conf 반전 +
A11 Day1 7% + breakout_fade alpha(6/1 retrain 후 활성).

---

## Part 2 — Claude의 솔직한 "부족한 부분" 평가

비판 강도순. 1~3이 근본, 나머지는 파생.

### 🔴 1. edge 자체가 너무 약하다 (근본 중의 근본)

direction alpha IC 0.02~0.03. 이건 거의 noise 수준이다. 학술적으로 IC 0.05+
가 "쓸만한" alpha, 0.10+가 "강한" alpha. 우리는 그 절반~1/3.

**모든 정교한 엔지니어링(regime, sizing, exit veto, conditional TP)이 이 약한
edge 위에 쌓여 있다.** edge가 약하니 아무리 잘 만들어도 ceiling이 낮다. V4에서
8개 alpha를 시도했지만 2개만 marginal PASS — 이 도메인(NASDAQ daily OHLCV)에
짜낼 edge가 거의 없다는 뜻.

### 🔴 2. 가장 강한 자산(vol IC 0.70)을 monetize 못한다

VolTransformer는 vol expansion을 IC 0.70으로 예측 — 진짜 강한 신호. 그런데:
- 이걸 conviction **보조**로만 쓴다 (direction × **conviction**)
- 직접 trade하려면 옵션인데, Phase 0 검증 결과 straddle은 VRP에 짐 (옵션
  시장이 이미 implied vol에 반영)

**"잘 예측하는 것"과 "수익으로 바꾸는 것"은 완전히 다른 문제.** 우리의 단 하나
강점이 monetize 안 되는 구조. 이건 전략의 가장 아픈 역설이다.

### 🔴 3. backtest가 over-fit이라 의사결정 기반이 흔들린다

baseline Sharpe 2.92~4.03 vs walk-forward +4%/년 (Sharpe ~0). 10배 차이.
지금까지 모든 lever 결정(A3 채택 등)이 **over-fit된 backtest metric**에
기반했다. A3가 진짜 좋은지조차 paper로 확인 안 됐다 (6/1 이후에나).

single train/test split + 모델 1회 학습. walk-forward retraining 없음. 즉
backtest는 "이 특정 1년, 이 특정 모델"에 최적화된 숫자일 수 있다.

### 🟡 4. paper 검증이 사실상 안 됐다

6주, 14 round-trip trades. 통계적으로 무의미한 표본. 게다가 그 6주 중
- 4 거래일 entries=0 (silent failure)
- 1주 데이터 동결 버그 (-1.15%)
- 5/25~ monthly cap으로 거래 0

**실제로 "정상 작동한 paper"가 며칠 안 된다.** +0.85%는 noise 범위. 우리는
사실상 검증 안 된 시스템을 운영 중.

### 🟡 5. universe가 edge ceiling을 가둔다

NASDAQ-100은 세계에서 가장 efficient한 large-cap. PEAD 같은 검증된 anomaly조차
여기선 ~0. small-cap은 alpha 있을지 몰라도 비용(0.5~1%)이 잠식 (V4에서 기각).
**efficient universe를 고른 순간 edge ceiling이 정해졌다.**

### 🟡 6. direction-vol 철학적 mismatch

vol(IC 0.70) 잘 예측 / direction(IC 0.02) 못 예측인데, **수익을 direction에
건다.** 잘하는 걸 안 쓰고 못하는 걸 쓴다. V3가 "vol expansion 예측"으로
출발했지만 결국 direction trade로 귀결된 게 근본 설계 모순.

### 🟡 7. 회전율 vs 비용

avg hold 1d, 월 10회 거래. 비용 0.1% × 회전율 누적. WF +4%/년에서 비용이 상당
부분 차지. 작은 edge에 높은 회전율 = 비용이 alpha를 갉아먹는 구조.

### 🟢 8. regime detection 신뢰성 미검증

8 macro features → 5단계 regime. hysteresis 2일. 이 분류가 실제로 alpha
weight를 올바르게 바꾸는지 독립 검증 없음. strong_bull은 n=99로 표본 극소.

### 🟢 9. 실거래 0 — 모든 게 paper/sim

PaperBroker는 yfinance 가격 + 고정 슬리피지. 실제 체결, 실제 스프레드,
세금(한국 해외주식 양도세 22%), 환위험 미반영. 실거래 이관 시 또 다른 갭 가능.

### 🟢 10. 목표-현실 격차

페르소나 목표 월 1% (연 20%) vs 데이터가 말하는 현실 연 4~7%. **3~5배 격차.**
이 격차를 메울 방법이 V4 탐색에서 안 나왔다.

---

## Part 3 — 메타 비판 (가장 불편한 질문)

### "우리는 없는 edge를 짜내려 한 것 아닌가?"

V4 3 세션 = 약한 edge를 더 짜내려는 시도였고, 거의 다 실패했다. 데이터는
일관되게 "이 도메인에 edge가 거의 없다"고 말한다. 그런데도 계속하는 건:
- sunk cost (이미 만든 정교한 시스템에 대한 애착)
- 페르소나 목표(월 1%)에 대한 집착
일 수 있다.

**정직한 질문 3가지**:
1. NASDAQ daily OHLCV + macro로 연 20%가 가능한 edge가 **애초에 존재하는가?**
   (데이터는 "아니오"에 가깝다고 답하는 중)
2. 그렇다면 (a) 목표를 현실(연 4~7%)로 낮추거나, (b) 완전히 다른 도메인
   (alternative data, HFT, 다른 자산군, 비효율 시장)으로 가야 하는 것 아닌가?
3. 현재 시스템의 정교함(regime/sizing/exit)이 **약한 edge를 가리는 착시**를
   만드는 건 아닌가? (backtest over-fit이 그 증거)

### 그럼에도 잘한 것

- 백테스트-라이브 parity, 검증 워크플로우(회귀 598개), silent failure 진단
  체계 — **엔지니어링은 견고**하다.
- "안 되는 것"을 데이터로 빠르게 기각하는 규율 (V4 전체) — 많은 퀀트가
  over-fit에 속는데 우리는 walk-forward로 자각했다.
- 즉 **문제는 실행력이 아니라 edge의 부재**다.

---

## Part 4 — 토론하고 싶은 논점

1. **edge 존재 여부**: NASDAQ daily에 연 20% edge가 있다고 믿는가, 없다고
   보는가? 데이터는 "거의 없다"인데, 그래도 시도할 이유가 있나?
2. **목표 재설정**: 연 4~7%를 수용하고 안정 운영할 것인가, 아니면 목표를
   위해 도메인을 바꿀 것인가?
3. **vol IC 0.70의 활용**: monetize 못하는 강점을 버릴 것인가, long-short
   vol arb 같은 복잡한 경로를 팔 것인가?
4. **다음 도메인 후보**: 만약 바꾼다면 — 비효율 시장(small-cap micro,
   신흥국, crypto)? alternative data? 다른 빈도(intraday/HFT)? 다른 전략
   (event-driven, stat-arb)?
5. **현 시스템의 운명**: 계속 paper 운영하며 6/1 retrain 효과를 볼 것인가,
   아니면 여기서 멈출 것인가?
