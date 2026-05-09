# Quant Trading System

NASDAQ-only Vol Expansion Trader. Conviction-or-cash strategy on monthly cadence.

**Current state**: V3.3 Profitability Engine (코드 완성, paper 활성화 진행 중).
운영 정책 본문은 `CLAUDE.md`. 토론용 reference는 `docs/CORE.md`.

---

## 30초 요약

- **무엇**: NASDAQ 100종목 중 5일 변동성 팽창이 예측되고 알파 방향이 정렬된
  케이스만 진입.
- **수식**: `opportunity = direction × conviction > cost × 1.75` (V3.2.1)
  → V3.3에서 calibration 도출 expected_return으로 확장.
- **사이즈**: Half-Kelly + 변동성 타겟팅 + 상관관계 drag → V3.3에서
  `net_edge / |expected_mae|` 기반 sizing + Pyramid winner add-on.
- **빈도**: 월 5회 이내, 1~3종목 집중. bear regime은 100% 현금.
- **성과 (BT)**: Sharpe 1.65, Return +38%, MDD 4%, Win 64%, PF 3.93.

---

## 디렉토리

```
v3/
├── config/        # v3_config.yaml, schema.py, alpha_weights.json
├── data/          # collector / feature_engineer / macro
├── model/         # VolTransformer (2.26M params)
├── strategy/      # OpportunityScorer, Regime, Sizing
│                  # + V3.3: edge_calibrator, edge_engine, edge_tier,
│                  #         exit_thesis, signal_decay, partial_exit,
│                  #         allocation, pyramid, rotation,
│                  #         book_optimizer, diagnostics
├── rules/         # entry / exit
├── execution/     # broker / paper_broker / executor / position_manager
│                  # + V3.3: execution_quality
├── backtest/      # engine + walk_forward + alpha_weight_trainer
├── pipeline/      # data / train / live
├── research/      # V3.3: build_edge_dataset, calibrate_edge, validate_edge,
│                  #       ablation_specs/runner/report, promotion_decision
├── scripts/       # run_data / run_train / run_backtest / run_live
│                  # + V3.3: run_calibration_pipeline, run_ablation_sweep,
│                  #         run_daily_report
└── tests/         # 476/476 passing
docs/              # CORE, CHANGELOG, FOLLOW_UPS, V3.3_DESIGN/ROADMAP/
                   # INTERFACES/AB_PLAN/CHECKLIST
saved_models/      # vol_transformer_*.pt, normalizer_stats.json,
                   # alpha_weights.json, edge_calibration.json (V3.3)
```

---

## 운영 명령 cheatsheet

```bash
PYTHON=/c/Users/wogus/miniconda3/envs/quant/python.exe

# === V3.2.1 운영 (현재 production) ===
$PYTHON v3/scripts/run_data.py --no-flow            # OHLCV + features
$PYTHON v3/scripts/run_train.py                     # VolTransformer
$PYTHON v3/scripts/run_backtest.py                  # backtest
$PYTHON v3/scripts/run_live.py --mode once          # paper 1회
$PYTHON v3/scripts/run_live.py --mode daemon        # paper daemon

# === V3.3 (코드 완성, paper 활성화는 features OFF default) ===
# Calibration pipeline (server, 매월 자동)
$PYTHON v3/scripts/run_calibration_pipeline.py \
    --ohlcv-path data/research/ohlcv_panel.parquet \
    --macro-path data/research/macro_pctl.parquet \
    --vol-pred-path data/research/vol_predictions.parquet

# Ablation sweep
$PYTHON v3/scripts/run_ablation_sweep.py --synthetic           # sanity
$PYTHON v3/scripts/run_ablation_sweep.py --panel <parquet>     # production

# Daily diagnostic report (server, 매일 16:00 KST 자동)
$PYTHON v3/scripts/run_daily_report.py

# === Server ===
bash deploy_v3.sh 77.42.78.9                        # deploy + systemd 자동 설치
ssh root@77.42.78.9 "systemctl status quant-trading-v3"
ssh root@77.42.78.9 "tail -f /var/log/quant-v3.log"

# V3.3 systemd timers (deploy_v3.sh에서 자동 설치)
ssh root@77.42.78.9 "systemctl list-timers | grep -E 'alpha|calibration|v33'"

# Paper account 확인
$PYTHON -c "from v3.execution.paper_broker import PaperBroker; PaperBroker().print_summary()"
```

---

## V3.3 Feature 활성화

V3.3은 모든 신규 정책이 `features.* = false` default. 활성화는
`v3/config/v3_config.yaml` `features:` 섹션 토글:

```yaml
features:
  # Week 0 (paper 활성화 즉시 가능, read-only)
  no_trade_logger: true
  tc_monitor: true
  execution_quality: true

  # Week 1+ (ablation 통과 후, 1주 간격으로)
  edge_calibrator: true
  edge_engine: true
  edge_tier: true
  conditional_veto: true
  exit_thesis: true
  signal_decay: true
  partial_exit: true
  allocation: true
  pyramid: true
  rotation: true
```

활성화 후 `bash deploy_v3.sh 77.42.78.9` → systemd 재시작 → paper 반영.

자세한 활성화 일정: `CLAUDE.md` "Paper Promotion 일정" 섹션.

---

## 테스트

```bash
PYTHONPATH=. $PYTHON -m pytest v3/tests/             # 476 tests
PYTHONPATH=. $PYTHON -m pytest v3/tests/test_regression.py -v  # 회귀만
```

회귀 hook은 `.claude/hooks/post_edit.py`가 v3/ 파일 편집 시 자동 실행.

---

## 참고 문서

| 문서 | 내용 |
|------|------|
| `CLAUDE.md` | 운영 정책 본문 (V3.2.1 + V3.3 활성화 가이드) |
| `docs/CORE.md` | V3.2.1 thesis (토론용 self-contained) |
| `docs/CHANGELOG.md` | Phase 22~26 (V3.3) 변경 이력 |
| `docs/FOLLOW_UPS.md` | 페르소나 정합성 + 보류 항목 |
| `docs/V3.3_DESIGN.md` | 13 모듈 책임·데이터 흐름 |
| `docs/V3.3_ROADMAP.md` | 5단계 일정·합격 기준 |
| `docs/V3.3_INTERFACES.md` | 함수 시그니처·protocol·CLI |
| `docs/V3.3_AB_PLAN.md` | A1~A9+Full ablation 실험 |
| `docs/V3.3_CHECKLIST.md` | PR별 작업 체크리스트 |
| `memory/v3_architecture.md` | V3 아키텍처 메모 |
| `memory/phase2_plan_detailed.md` | Phase 26 (V3.2) 재설계 상세 |

---

## 페르소나 — "확신 있을 때만, 크게, 빠르게"

| 원칙 | V3.2.1 점수 | V3.3 처방 |
|------|------------|----------|
| 1. 확신 있을 때만 | 5/10 | EdgeCalibrator → expected_return / EdgeTier → S/A/B/C |
| 2. 크게 | 3/10 (가장 심각) | AllocationEngine + Pyramid + Phase 25.2 sizer floor 0.15 |
| 3. 빠르게 | 8/10 | Conditional Veto 정상화 + ExitThesis 4-way decision |

자세한 평가는 `docs/FOLLOW_UPS.md` "페르소나 원칙 점수".
