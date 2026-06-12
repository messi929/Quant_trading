# V4 Dual-Engine 설계 — 한국 momentum + 미국 reversion (24시간)

**작성**: 2026-05-29
**상태**: ✅ 한국 엔진 검증 완료 (최종 SPEC = §1.9 하단). 미국 엔진 기각.
**최종 엔진**: KOSDAQ multi-lb ensemble momentum + 200d SMA gate + vol-target =
full-cycle Sharpe 0.66 / annual +10.5% / MDD −17% (2014-2026, survivorship-free, robust)
**다음 세션 시작점**: Phase B 엔진 구현 (또는 modest Sharpe 0.66 수용 여부 결정)
**paper 가동 (2026-06-12 update)**: ✅ KIS 계좌 교체(50169471→**50192869**, 옛 계좌
`40910000` 주문차단)로 첫 실거래 준비 완료 — 잔고 1억·양방향 실주문 검증, state 리셋.
**월 2026-06-15 09:05 타이머가 fresh KOSDAQ 20종목 rebalance 자동 실행 = 실거래 첫 시작.**
상세 = CHANGELOG "V4 KOSDAQ paper — 주문차단 해소 (2026-06-12)".

> ⚠️ **2026-05-29 업데이트 — 미국 엔진 기각**: EODHD 상폐 포함 데이터(NASDAQ
> common 6,753종목, survivorship-free)로 재검증한 결과 reversion edge가 **전부
> survivorship bias**였음. survivor-only +30%/Sharpe0.64 → survivorship-free
> -16%/-0.50 (현실적 상폐패널티). 가장 관대한 가정(상폐 무손실)에서도 +3%/0.11.
> price floor($3~$20)·pool크기(100~300)·VIX filter 전 조합에서 음수. **dual-engine
> 전제(uncorrelated 24h) 붕괴 → 한국 momentum 단일 엔진으로 전환.** 상세 §2.4.

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

### 1.4 risk-overlay 실험 결과 — regime filter (lever 1) = 목표 거의 달성 (2026-05-29)

`v3/research/korea_risk_overlay.py` — 검증된 PIT momentum 위에 지수 추세 게이트
(지수 risk-off 시 현금) + KOSPI/KOSDAQ 결합. 기존 PIT 캐시 재사용, 신호/비용/상폐처리
동일 (overlay 효과만 격리). N=20, cost 0.4%, 상폐pen 50%.

| 구성 | annual | Sharpe | MDD | cash% |
|------|-------:|-------:|----:|------:|
| KOSDAQ gate 없음 (baseline) | +18.0% | 0.54 | -51.5% | 0% |
| **KOSDAQ + SMA200** | **+26.6%** | **1.02** | **-24.7%** | 52% |
| KOSDAQ + mom120 | +21.8% | 0.85 | -24.7% | 53% |
| KOSPI + mom60 | +22.1% | 0.74 | -27.9% | 42% |
| 결합(50/50) + mom60 | +23.0% | 0.88 | -27.2% | 35% |

**핵심**: regime gate가 MDD를 반감(-51.5%→-24.7%)하면서 수익은 오히려 상승
(하락장 회피). 1순위 lever 하나로 목표(MDD≤25%, Sharpe≥1.0) 통과 (KOSDAQ+SMA200).

**lever는 robust, 정확한 best는 noise**: 모든 gate·시장에서 MDD 일관 ~25%, Sharpe
~0.8~1.0. 12조합 중 best 선택 + n=62 rebalance(소표본) → "sma200이 정확히 1등"은
과적합 여지. 신뢰 결론 = "regime filter가 MDD 반감 + Sharpe~0.9". win 32%는
현금 52% 탓 (거래 기간만 보면 ~67%).

**다음 검증 과제**: (a) gate 선택 robustness (walk-forward/sub-period로 in-sample
과적합 점검), (b) 잔여 lever(vol-target/trailing stop)로 Sharpe 추가 개선 여지,
(c) Phase B 엔진 구현 (universe collector + 신호 + regime + sizing + KIS 국내 paper).

### 1.5 gate robustness 검증 — 효과 진짜, 정확한 수치 과신 금지 (2026-05-29)

`v3/research/korea_gate_robustness.py` 3각 점검 (param 안정성 / sub-period / walk-forward):

**(A) 파라미터 안정성** — KOSDAQ SMA 전 윈도우(100~300) Sharpe 0.73~1.02 / MDD
−18~30%. baseline −51.5% 대비 **모든 param이 MDD 반감 = plateau** (knife-edge 아님,
robust). 단 **시장별 best param 불일치**: KOSDAQ sma200/mom90, KOSPI sma100/mom60
→ "sma200 최적"은 transfer 안 됨 (best-fit 선택은 과적합).

**(B) sub-period** — 게이트 가치는 2022 하락기에 집중:
| 기간 | KOSDAQ base | KOSDAQ+sma200 |
|------|------------|---------------|
| 2022 하락 | MDD −51.5% / −31.5% | **MDD −13.6% / −4.8%** |
| 2023-24 회복 | −22.6% / +30.4% | −24.7% / +12.7% (게이트 손해) |
| 2025-26 강세 | −12.8% / +122% | −2.4% / +133% |

→ **regime filter = 크래시 보험**. 위기에 크게 방어, 평시엔 수익 프리미엄 지불.
KOSPI 동일 패턴(2022 −41.5%→−13.7%).

**(C) walk-forward** (과거로 gate 선택 → 다음해 적용, in-sample 편향 제거):
- KOSDAQ: WF 1.63/−24.7% ≈ fixed-sma200 1.56 (붕괴 안 함)
- KOSPI: WF 1.21 > no-gate 1.04 > fixed-sma200 0.61
- ⚠️ WF 기간(2023~26)에 크래시 없음 → no-gate가 게이트와 비슷. 게이트 가치는 크래시 시 발현.

**결론**:
- ✅ robust: MDD 반감 효과는 양 시장·전 param·walk-forward 일관. 과적합 아님.
- ⚠️ 과신 금지: 정확한 best(+26.6%/1.02)는 시장별로 안 맞음 → **표준 게이트(200d SMA)
  또는 ensemble을 a-priori 원칙으로 채택, best-fit 금지.** 운영 기대치 = MDD ~20-25%,
  Sharpe ~0.9 (전체 사이클 기준).
- 🚨 **최대 한계: 표본 내 크래시 2022 단 1회.** 크래시 보험을 n=1로 검증한 셈 — 더
  긴 history(2015~, IFRS/금융위기 제외) 또는 다중 하락기 표본으로 보강 필요.

### 1.6 긴 history (2014-2026) 검증 — 2021-2026은 period luck였음 (2026-05-30)

`v3/research/korea_long_history.py` — 2014~ survivorship-free 캐시(KOSDAQ 949 /
KOSPI 758, top600 live + 2015~상폐) 구축, 4개 하락기(2015-16/2018Q4/2020COVID/2022)
다중 검증. **§1.4~1.5의 낙관적 수치(Sharpe~1.0)를 깸.**

**전체 기간 (148 rebalance):**
| | KOSDAQ none | KOSDAQ+sma200 | KOSPI none | KOSPI+mom60 |
|---|---:|---:|---:|---:|
| annual | −3.9% | +11.3% | +1.7% | +8.5% |
| Sharpe | −0.11 | **0.41** | 0.05 | **0.32** |
| MDD | −89.4% | −40.3% | −64.2% | −45.8% |

→ **2021-2026의 +26.6%/1.02/MDD−24.7%는 period luck** (2025-26 +122% 폭발장이 최근
창을 띄움). **full-cycle 현실 = Sharpe 0.3~0.4, MDD ~40% (게이트 적용 후).** 목표
(Sharpe 1.0, MDD 25%) full-cycle 미달. 게이트는 raw momentum의 −89% MDD(momentum
crash)를 −40%로 반감시키는 핵심이긴 함.

**크래시별 [base→sma200] MDD:**
| 크래시 | KOSDAQ | KOSPI | 게이트 |
|--------|-------|-------|--------|
| 2015-16 | −29%→−12% | −33%→−10% | ✅ |
| 2018 Q4 | −25%→0% | −20%→0% | ✅ 완전회피 |
| **2020 COVID** | −17%→−17% (ret −0%→**−13%**) | +19%→**−18%** | ❌ **whipsaw** |
| 2022 | −53%→0% | −32%→0% | ✅ 완전회피 |

**핵심 진단**: regime gate는 **느린 약세장(2015·2018·2022)엔 강력**(MDD 거의 0),
**빠른 V자 크래시(COVID)엔 whipsaw** — 하락 후 뒤늦게 risk-off, 반등 놓침. KOSPI는
COVID에 baseline이 +19%인데 게이트가 −18%로 오히려 손해. 추세 필터의 구조적 약점.

**재조정된 결론**:
- momentum 엣지 + regime gate는 real but **modest: full-cycle Sharpe ~0.4, annual
  ~8-11%, MDD ~40%.** 지수 대비 +α는 분명하나 "Sharpe 1.0 무위험+α" 엔진은 아님.
- 목표 달성하려면 잔여 lever(vol-targeting=momentum crash 직접 대응, trailing stop,
  position stop)가 **선택이 아니라 필수**. + COVID whipsaw 대응(느린 재진입/vol spike
  필터). 그래도 Sharpe 1.0은 불확실.
- 2021-2026 단일창 검증의 위험성을 실증 — 앞으로 모든 V4 수치는 full-cycle 기준.

### 1.7 vol-targeting + trailing stop lever (2026-05-30)

`v3/research/korea_risk_levers.py` — full-cycle(2014-2026) momentum crash 대응.

| KOSDAQ 구성 | annual | Sharpe | MDD | COVID MDD/ret |
|------|-------:|-------:|----:|----:|
| gate only | +11.3% | 0.41 | −40.3% | −17%/−13% |
| **gate + voltarget** | +7.3% | **0.50** | **−21.4%** | −9%/−7% |
| gate + trailing | +4.4% | 0.17 | −47.4% | −15%/−11% |
| gate+voltarget+trailing | +4.1% | 0.29 | −25.6% | −12%/−9% |

- ✅ **vol-targeting = keeper**: MDD −40%→**−21%(목표≤25% 통과)**, COVID whipsaw 부분
  개선(−13%→−7%, 변동성 연속반응이 binary gate보다 우월). Sharpe 0.41→0.50.
  단 cap=1.0(무레버리지)이라 평시 수익 회복 없음 → annual은 하락(+11.3%→+7.3%).
- ❌ **trailing stop 기각**: Sharpe 0.41→0.17. momentum 종목 노이즈 조기청산.
  **V3 기존 교훈("개별 스탑 제거 시 Sharpe 0.65→1.65")과 일치** — 추세전략에 스탑 해롭.
- KOSPI: vol-target 효과 미미(full-cycle 0.17). 약한 시장.

**최선 엔진 (full-cycle 정직)**: KOSDAQ momentum + 200d SMA gate + vol-target =
**+7.3% annual / Sharpe 0.50 / MDD −21%**. vol-target이 MDD는 목표 안에 넣었으나
Sharpe는 0.50에서 막힘 (0.7도 1.0도 아님). 잔여 lever 소진.

**결정 분기**: +7.3%/0.50/MDD−21% 는 재설정 목표("무위험~3% + α", 지수 +4% 대비 우위)에
부합하나 원래 Sharpe 1.0 꿈은 아님. → (a) 이 modest 엔진으로 Phase B 구현+paper,
(b) leverage cap↑/신호 변형으로 return-side 추가 시도(체감 한계), (c) 접근 재검토.

### 1.8 return-side sweep — leverage·신호튜닝 robust 개선 실패 (2026-05-30)

`v3/research/korea_lever_sweep.py`.

**Sweep A (leverage cap, KOSDAQ)**: cap 1.0→3.0 → annual +7.3%→+8.7%, Sharpe
0.50→0.49(평탄), MDD ~21%(평탄). vol-target이 대부분 target vol 근처라 lever-up 드묾
→ **leverage 무용**. KOSPI는 cap↑이 오히려 악화.

**Sweep B (signal lb×N, cap1.5, KOSDAQ)**: lb40_N20 Sharpe 0.92/+14%/MDD−17%(best),
lb60 0.50, lb90 **−0.22(음수!)**, lb120 0.45.
🚨 **과적합 signature**: lb90이 2021-2026 robustness에선 best(1.18)였는데 full-cycle
worst(음수)로 뒤집힘. lookback 응답 jagged(0.92→0.50→−0.11→0.45) = plateau 아님.
→ **lb40=0.92도 신뢰 불가** (lb90 실수 반복 위험).

**확정 결론**: return-side lever robust 개선 실패. 믿을 수 있는 full-cycle 기대치 =
**Sharpe ~0.5 / annual ~7-8% / MDD ~21%** (KOSDAQ momentum + 200d SMA gate +
vol-target cap1.0~1.5, lb60 표준). lookback 파라미터 fragile 확인 → 단일 lb 튜닝
금지, 필요시 multi-lb ensemble(미검증, 예상 ~0.5-0.6)이 유일한 robust 경로.
**메타 교훈**: 2021-2026 lb90 best가 full-cycle 음수 = 단일창·단일파라 튜닝의 위험 실증.

### 1.9 multi-lb ensemble — fragility 제거 + robust 개선 (2026-05-30)

`v3/research/korea_ensemble.py` — 단일 lb 베팅 대신 lookback {40,60,90,120}
cross-sectional 랭크 평균 blend (한 horizon 운/불운 제거). gate+vol-target(cap1.5).

| KOSDAQ | Sharpe | annual | MDD |
|--------|-------:|-------:|----:|
| lb60 단일 (정직 baseline) | 0.52 | +8.2% | −21.4% |
| lb40 단일 (신기루) | 0.93 | +14.2% | −17.5% |
| **ensemble** | **0.66** | **+10.5%** | **−17.2%** |

ensemble이 개별 평균(0.61) 상회 + MDD 최저(분산 효과). lb40의 0.93(신뢰불가)은
아니나 lb60의 0.50을 **robust하게 0.66으로** — 단일 파라 베팅 없이. 크래시 보호 유지
(2018·2022 ~0%, COVID −8%). lb90이 sweep에선 −0.22인데 여기선 +0.53(start date만
달라도 flip) = 단일 lb fragility 재확인, ensemble이 이를 평균으로 제거. KOSPI ensemble
0.15 — 약한 시장 확정 → **엔진 KOSDAQ 단독**.

---

## ✅ 최종 검증 엔진 SPEC (full-cycle 2014-2026, survivorship-free, robust)

```
시장:     KOSDAQ 단독 (KOSPI 기각 — full-cycle Sharpe 0.15)
신호:     multi-lb ensemble momentum (lb 40/60/90/120 cross-sectional 랭크평균
          + TS trend filter: 4개 lb past return 평균 > 0)
universe: 각 rebalance PIT 거래대금(close×vol) top100 (look-ahead 없음)
포지션:   hold 20 거래일, N=20 equal-weight, long-only
regime:   200d SMA gate — 지수 < 200d SMA 시 전량 현금 (느린 약세장 방어)
risk:     vol-targeting — exposure = clip(0.15/trailing_realized_vol, 0, 1.5)
비용가정: 왕복 0.4% + 상폐 정리매매 손실 50%
결과:     Sharpe 0.66 / annual +10.5% / MDD -17% / 크래시(2018·2022) 거의 0
```

**검증 통과 게이트**: survivorship-free(상폐 포함) ✓ / beta-neutral alpha ✓ /
다중 하락기(2015·2018·2020·2022) ✓ / look-ahead 제거(PIT) ✓ / walk-forward ✓ /
파라미터 robust(ensemble) ✓. Sharpe 1.0은 아니나 정직하게 검증된 "무위험+α"
(무위험 ~3% 대비 +7%p, KOSDAQ 지수 +3.8% 대비 +6.7%p, MDD 17%).

**한계/주의**: ① 백테스트 only — 실거래 슬리피지(KOSDAQ 소형 market impact)·세금·
환경변화 미반영. ② COVID류 빠른 V자 크래시엔 부분 whipsaw(−8%). ③ Sharpe 0.66은
modest — 운영 노력 대비 가치는 사용자 판단. ④ KIS 국내 paper로 forward 검증 필요.

**다음 (Phase B 구현 시)**: universe collector(FDR live+상폐 PIT) + ensemble 신호
모듈 + 200d SMA regime + vol-target sizing + KIS 국내 paper 연동. V3 인프라
(regime 프레임워크/sizing/백테스트 엔진/PaperBroker) 재활용, 신호·universe만 교체.

### 1.10 KOSPI 엣지 탐색 — 가격신호 배포불가 확정 (2026-05-30)

"KOSPI도 설계" 요청에 대해 가격 기반 엣지 전수 탐색 (`v3/research/korea_kospi_edge.py`,
KOSPI long캐시 + KS11, full-cycle):

| KOSPI 전략 (raw) | Sharpe |
|---|---:|
| reversion (lb5~20, hold5~20) | −0.35 ~ +0.03 |
| low-vol (hold20) | +0.06 |
| momentum (lb60) | +0.06 |
| best + gate | 0.17 (regime 효과일 뿐, 종목선택 엣지 아님) |
| best + gate + vol-target | 0.05 |

**전부 0 근처/음수 → KOSPI 대형주는 가격신호로 배포불가.** NASDAQ-100과 동일 —
효율적 시장이라 momentum/reversion/low-vol 차익거래로 소멸. 가격신호 KOSPI 엔진 =
Sharpe 0.1 신기루 → 설계 안 함. **유일한 경로 = 펀더멘털(value/저PBR/quality)**, DART
재무데이터 필요한 별도 연구 (보류). **V4 = KOSDAQ momentum 단일 엔진 확정.**

### 1.11 KOSPI 펀더멘털 factor — clean 신호 약함, 0.5는 mirage 가능성 (2026-05-30)

DART API(무료) 확보 → `v3/research/dart_fetch.py`(corp_code 매핑 + 연간재무 IFRS
태그 추출, 620종목 2015-2024) + `kospi_fundamental_factor.py`(PIT lag Y→Y+1 4월).

| factor | 데이터 | raw Sharpe | +gate |
|--------|------|-----------:|------:|
| quality_roe | ✅ clean SF | 0.15 | −0.02 |
| quality_lowdebt | ✅ clean SF | −0.01 | −0.19 |
| value_pbr | ⚠️ survivor-biased | 0.41 | 0.17 |
| composite(value+quality) | ⚠️ survivor-biased | 0.44 | 0.51 |

**완전 survivorship-free(quality)는 약함(0.15).** 0.4~0.5는 전부 value/composite인데
**survivor-biased** — 시총=close×현재주식수라 상폐주(현재 주식수 없음)가 value 랭킹에서
제외됨 = value trap(싸 보였지만 망한 종목) 누락 → value 수익 부풀림. 오늘 내내 본 mirage
패턴. composite+gate 0.51도 gate의 MDD 효과(−50%→−28%)지 factor 자체는 0.44 biased.

**판정**: KOSPI 가격(0.15)에 이어 펀더멘털 clean 신호도 약함. 0.5 근처는 낙관편향 →
survivorship-free 정밀화(상폐주 PIT 발행주식수, DART stockTotqySttus 추가) 시 하락 예상.
**KOSPI는 가격·펀더멘털 둘 다 접근가능 데이터로 deployable 엣지 없음 (효율적 시장).
V4 = KOSDAQ 단일 엔진 유지.** (value SF 정밀검증은 prior 낮아 보류.)

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

### 2.3 미확정 — EODHD survivorship-free 확정 필요 (→ §2.4에서 해소)

survivor-only는 reversion 부풀림 (과매도 중 상폐된 것 누락 → 반등한 것만 보임).
**EODHD $20로 미국 상폐 포함 데이터 확보 후 재검증 필수.**

예상: annual +15% → +8~12%로 하락 (한국 수준), 단 VIX filter MDD 제어는 유지 →
Sharpe 살아남을 것.

### 2.4 재검증 결과 — 기각 (2026-05-29)

EODHD `EOD Historical Data — All World` ($19.99/mo) 구독 후
`v3/research/backtest_nasdaq_pit.py` 로 survivorship-free 재검증. 예상(+8~12% 유지)은
틀렸음 — **reversion edge가 전부 survivorship bias였고, 제거 시 음수로 전환.**

데이터: NASDAQ Common Stock active(4,097) ∪ delisted(10,459) = 14,556 후보 → window
(2021-06~2026-05) 내 데이터 충분 6,753종목. point-in-time 거래대금 universe (top30
mega 제외 → 다음 N), lb5/10 reversion 하위20% long, 상폐 손실 반영.

| 조건 | survivor-only (이전) | survivorship-free (실제) |
|------|---------------------:|-------------------------:|
| lb10 long-only | +30.0% / 0.64 | **-36.1% / -0.86** |
| lb10 + VIX | +14.7% / 0.67 | **-15.8% / -0.47** |
| lb5 long-only | ~+21% | -41.4% / -1.00 |

**상폐 패널티(PEN) 민감도** (lb10, POOL150, minPx$10, VIX ON) — 결과의 핵심 lever:
```
PEN 0.0 (상폐 무손실, 최대 관대):  +3.1% / Sharpe 0.11
PEN 0.3 (현실적):                  -16.4% / -0.50
PEN 0.6 (보수적):                  -38.8% / -0.78
```

robust 음수: price floor $3~$20, POOL 100~300, VIX on/off 전 조합에서 음수.
universe를 아무리 깨끗하게(고가·초유동) 해도 회복 안 됨. 가장 관대한 가정에서도
+3%/0.11 → 배포 기준(Sharpe≥1.0) 및 "무위험+α" 모두 미달.

**원인**: 떨어지는 칼 long의 비대칭. 과매도 종목 중 상당수가 반등 못 하고 계속 하락
/ 상폐 → survivor universe가 이들을 빼버려 "반등한 것만" 보였던 착시.

**결론**: 미국 reversion 엔진 폐기. (참고 산출물: `reports/nasdaq_pit_backtest.json`,
캐시 `reports/nasdaq_pit_cache.parquet` = 재사용 가능한 survivorship-free 패널.)

### 2.5 US 다른 엣지 1회 probe — 전부 기각 (2026-05-29)

reversion 기각 후 같은 survivorship-free 패널로 momentum·low-vol probe
(`v3/research/probe_nasdaq_alphas.py`, long-only, hold 20d):

| 전략 | best (survivorship-free) |
|------|-------------------------:|
| reversion (lb10) | -15.8% / -0.47 |
| TS/CS momentum (lb120+VIX) | -4.1% / -0.16 |
| low-vol | -7.2% / -0.58 |

momentum이 reversion보다 덜 나쁨(승자 매수 = 상폐 함정 회피, 가설 적중)이나 여전히
음수. **모든 단순 long-only US 엣지 음수.** 기간(2021-06~2026-05)이 2022 소형성장주
학살 + de-SPAC 붕괴 포함 → survivorship-free 중소형 long-only는 이 기간 무덤.
한국 momentum은 동일 2022 하락기 포함하고도 4중 게이트 통과 → "시장 선택이 진짜
차이" 결론 강화. **US 엔진 완전 폐기, 한국 단일 엔진 확정.** (패널은 보존 — fundamentals
추가 시 quality-reversion 등 미래 재검증 여지.)

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
- **실거래 미검증** — 백테스트 기반. 주문 경로는 2026-06-12 KR 장중 실주문(005930
  1주 매수→매도)으로 검증됨. 단 전체 바스켓 체결/슬리피지/세금은 6/15 첫 rebalance
  부터 forward 관찰 필요. **6/15 점검: picks=20 전량 체결 확인(`/var/log/quant-v4.log`).**
- **목표 재설정** — "연 20% 신화" 버리고 "무위험+α + 내 시스템". 한국 +20%(raw),
  risk 제어 후 Sharpe 1.0+면 목표 달성.

---

## 6. 검증 결과 아카이브 (reports/)

- `korea_ts_alpha_final.json` — 한국 4중 게이트 alpha
- `kosdaq_pit_backtest.json`, `kospi_pit_backtest.json` — point-in-time net
- `nasdaq_reversion.json`, `nasdaq_engine.json` — 미국 reversion + VIX
- `momentum_by_marketcap.json` — 시총별 비교
- 데이터 캐시 parquet은 .gitignore (재생성 가능)
