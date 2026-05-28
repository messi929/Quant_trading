# V4 Dual-Engine 설계 — 한국 momentum + 미국 reversion (24시간)

**작성**: 2026-05-29
**상태**: 한국 엔진 검증 완료(설계 단계), 미국 엔진 윤곽 완료(EODHD 확정 대기)
**다음 세션 시작점**: EODHD API 키 받으면 미국 엔진 survivorship-free 확정부터

---

## 0. 배경 — V4 도메인 탐색 결론

paper +0.85%/6주 정체 → "수익 어떻게 올리나" → 9+번 검증.

**핵심 결론**: NASDAQ-100 large-cap은 too efficient (direction IC ≤ 0.03, 모든
단순 alpha 막힘). 진짜 edge는 **시장 선택 + 시장별 맞는 전략**에 있었음:
- **한국**: 개인 herding/덜 efficient → **momentum** (추세 추종)
- **미국 중소형**: 과민반응 → **reversion** (과매도 반등)

**같은 전략을 두 시장에 강제하면 둘 다 실패** (NASDAQ에 momentum = IC −0.002).
시장마다 독립 엔진 필요.

검증 도구 (모두 무료, `v3/research/`):
- `test_momentum_by_marketcap.py` — 시총별 momentum
- `test_korea_ts_alpha_final.py` — 한국 survivorship-free + beta + 하락기
- `backtest_kosdaq_pit.py` — point-in-time universe (KOSPI/KOSDAQ, `--market`)
- `test_nasdaq_reversion.py`, `test_nasdaq_engine.py` — 미국 reversion + VIX

데이터: **FinanceDataReader 무료** (한국 상폐 포함 survivorship-free 가능).
미국 상폐는 무료 불가 → **EODHD $19.99/월 결제 예정**.

---

## 1. 한국 엔진 (검증 완료 — 설계 단계)

### 1.1 확정된 조건

| 항목 | 값 | 근거 |
|------|-----|------|
| 시장 | KOSPI + KOSDAQ | 둘 다 momentum real |
| 신호 | **time-series momentum** (past 60일 수익률 > 0) | CS는 noise, TS만 alpha |
| holding | 20 거래일 | sweet spot |
| universe | 각 시점 **거래대금 상위** (point-in-time) | look-ahead 제거 |
| 방향 | long-only | 한국 공매도 제약 |
| regime | bear/하락기 현금 | 하락기 alpha 약함(+0.0048) |

### 1.2 검증 결과 (4중 게이트 통과 — V4 유일)

```
survivorship-free ✓ (FDR 상폐 포함 386종목)
beta 제거 ✓ (KOSDAQ 지수 대비 market-neutral)
하락기 ✓ (2022: 시장 −2.21% vs 추세 −1.74%, alpha +0.0048 방어적)
look-ahead 제거 ✓ (point-in-time 거래대금 universe)
```

**net 백테스트 (비용 0.4% + 상폐손실 반영, point-in-time)**:
| 시장 | N=20 annual | Sharpe | MDD |
|------|------------:|-------:|----:|
| KOSDAQ | +20.5% | 0.61 | **−49.9%** |
| KOSPI | +16.1% | 0.47 | **−39.9%** (N=10은 +28.9%) |

지수 benchmark +3.8%. → momentum edge 진짜 존재 (지수 대비 +12~17%p 초과).

### 1.3 설계할 부분 (다음 작업) — MDD가 핵심 과제

**문제**: raw momentum MDD −40~50%. 운영 불가 수준. edge는 real, risk 제어 필요.

**설계 항목** (MDD 50% → 25%, Sharpe 0.6 → 1.0+ 목표):
1. **Regime filter** — 시장 하락기 현금 (MDD 상당분이 하락장 동반). 한국 macro
   또는 지수 추세 filter. **1순위** (MDD 가장 크게 줄 듯).
2. **KOSPI + KOSDAQ 결합** — KOSPI(안정 MDD −40%) + KOSDAQ(고수익) 분산.
3. **Trailing stop** — winner riding + loser cut (20d hold라 작동, NASDAQ 1d와 달리).
4. **Vol targeting** — 변동성 큰 종목 비중 축소.
5. **분산 (N↑)** — N=30이 N=10보다 MDD 작음.

**기존 V3 자산 재활용**: VolTransformer(conviction), regime 프레임워크,
sizing(inv-vol/Kelly), 백테스트/검증 엔진, PaperBroker. universe·신호만 교체.

---

## 2. 미국 엔진 (윤곽 완료 — EODHD 확정 대기)

### 2.1 윤곽 잡힌 조건 (survivor-only 탐색 기준)

| 항목 | 값 | 근거 |
|------|-----|------|
| 시장 | NASDAQ 중소형 | reversion은 중소형에서 강함 |
| 신호 | **reversion** (past 10일 과매도 → long) | 미국 = 단기 mean-reversion |
| holding | 5 거래일 (단기) | reversion은 빠름 |
| 방향 | **long-only** (short 기각) | 과매수 short은 손실 (성장주 momentum 잔존) |
| **regime** | **VIX 고변동성 filter** | 고변동성=과민반응=reversion 강함. MDD 제어 핵심 |
| 비용 | 0.1% (저비용 → 고빈도 OK) | |

### 2.2 탐색 결과 (survivor-only — 부풀림 주의)

| lookback | mode | annual | Sharpe | MDD |
|----------|------|-------:|-------:|----:|
| 10 | long_only | +30.0% | 0.64 | −41.6% |
| 10 | long_short | −2.4% | −0.07 | −48% (**short 기각**) |
| 10 | **long_short+VIX** | +14.7% | **0.67** | **−22.9%** |

**핵심 발견**:
- **long-short 기각** — short side 손실 (미국 성장주 과매수 momentum 잔존).
  reversion은 long(과매도 반등)만 작동.
- **VIX filter가 MDD 절반** (−42% → −23%) — 나스닥 고유 조건. 고변동성 구간에서만
  진입. 한국엔 없던 regime 신호.

### 2.3 미확정 — EODHD survivorship-free 확정 필요

survivor-only는 reversion 부풀림 (과매도 중 상폐된 것 누락 → 반등한 것만 보임).
**EODHD $20로 미국 상폐 포함 데이터 확보 후 재검증 필수.**

예상: annual +15% → +8~12%로 하락 (한국 수준), 단 VIX filter MDD 제어는 유지 →
Sharpe 살아남을 것.

---

## 3. 24시간 Dual-Engine 구조

| | 한국 엔진 | 미국 엔진 |
|---|----------|-----------|
| **시간 (KST)** | 09:00~15:30 (낮) | 23:30~06:00 (밤) |
| **신호** | 60d momentum | 10d reversion |
| **방향** | long-only | long-only |
| **regime** | 하락기 현금 | VIX 고변동성 filter |
| **universe** | KOSPI+KOSDAQ point-in-time | NASDAQ 중소형 |
| **데이터** | FDR 무료 ✓ | EODHD $20 (예정) |
| **상태** | 검증완료, risk 설계 | 윤곽완료, 확정 대기 |

**왜 dual-engine** (이상적 퀀트 — 약한 edge 여럿 결합):
- momentum vs reversion = **메커니즘 정반대 → uncorrelated**
- regime도 다름 (한국 macro vs VIX)
- 시간대 분리 → 자본 24시간 회전
- uncorrelated 두 엔진 결합 → 분산으로 Sharpe↑, MDD↓ (√2 효과)

---

## 4. 다음 세션 액션 (우선순위)

1. **[내일] EODHD 가입 + $19.99 결제 + API 키** → 미국 상폐 포함 데이터
2. **미국 엔진 survivorship-free 확정** — EODHD로 universe(상폐 포함) + 가격 수집,
   reversion lb10 + VIX filter 재검증. 부풀림 제거 후 annual/Sharpe/MDD 확인.
3. **한국 엔진 risk management 설계** — regime filter(1순위) + KOSPI+KOSDAQ 결합
   + trailing stop. MDD 50% → 25%, Sharpe → 1.0+ 목표. (미국과 병렬 가능)
4. **dual-engine 통합 백테스트** — 두 엔진 결합 시 분산 효과(상관/Sharpe/MDD) 측정.
5. **운영 설계** — V3 인프라(PaperBroker/regime/sizing) 재활용, 24시간 스케줄.

---

## 5. 미해결 / 주의

- **numpy 2.x 업그레이드** — finance-datareader 설치로 numpy 1.x → 2.4.6.
  v3 production 코드 호환성 **미점검**. `pytest v3/tests/` 회귀 돌려서 깨지면
  numpy 핀 고정 필요. (한국/미국 엔진 검증과 별개, 기존 V3 운영 영향 가능)
- **survivor-only 부풀림** — 미국 모든 수치는 EODHD 확정 전까지 낙관적.
- **표본 한계** — KOSDAQ n=62 rebalance, 미국 76종목. 더 넓은 universe로 보강 여지.
- **실거래 미검증** — 전부 백테스트. 실제 체결/슬리피지/세금/환율 별도.
- **목표 재설정** — "연 20% 신화" 버리고 "무위험+α + 내 시스템". 한국 +20%(raw),
  risk 제어 후 Sharpe 1.0+면 목표 달성.

---

## 6. 검증 결과 아카이브 (reports/)

- `korea_ts_alpha_final.json` — 한국 4중 게이트 alpha
- `kosdaq_pit_backtest.json`, `kospi_pit_backtest.json` — point-in-time net
- `nasdaq_reversion.json`, `nasdaq_engine.json` — 미국 reversion + VIX
- `momentum_by_marketcap.json` — 시총별 비교
- 데이터 캐시 parquet은 .gitignore (재생성 가능)
