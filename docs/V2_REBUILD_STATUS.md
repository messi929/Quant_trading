# Quant Trading System v2 — 재구축 현황

## 개요

- **작업일**: 2026-03-30 ~ 2026-03-31
- **목표**: "확신 있을 때만, 크게, 빠르게" 철학 기반 처음부터 재구축
- **상태**: Phase 0~8 코드 완료, 모델 게이트 부분 통과

---

## 신규 생성 파일 (12개)

| 파일 | 역할 | LOC | 상태 |
|------|------|-----|------|
| `config/system_config.yaml` | 통합 설정 (4개 YAML → 1개) | ~200 | ✅ |
| `config/config_loader.py` | 설정 로더 + 캐시 + 유틸 | ~70 | ✅ |
| `data/normalizer.py` | z-score 피처 정규화 (fit/transform/save/load) | ~100 | ✅ |
| `models/alpha_model.py` | AlphaTransformer (mean pooling + confidence head) | ~130 | ✅ |
| `models/alpha_trainer.py` | 학습기 (Huber+Ranking+BCE, 게이트 체크) | ~250 | ✅ |
| `strategy/stock_signal.py` | ConvictionSignalGenerator (1~3종목 집중) | ~220 | ✅ |
| `backtest/intraday_sim.py` | OHLC 기반 장중 청산 시뮬레이션 | ~130 | ✅ |
| `backtest/conviction_engine.py` | Conviction 백테스트 엔진 (profit/stop/time) | ~230 | ✅ |
| `live/executor.py` | 진입/모니터/청산 실행기 | ~250 | ✅ |
| `pipeline/signal_pipeline.py` | 추론 파이프라인 (단일 모델) | ~200 | ✅ |
| `pipeline/train_pipeline_v2.py` | 학습 파이프라인 (게이트 포함) | ~160 | ✅ |
| `scheduler/runner.py` | 2세션 스케줄러 (KR+US) | ~220 | ✅ |

## 수정 파일 (2개)

| 파일 | 변경 내용 |
|------|----------|
| `data/dataset.py` | `create_dataloaders` 반환값에 normalizer 추가, `normalize` 파라미터, `prediction_horizon` 기본값 5→1 |
| `CLAUDE.md` | 투자 철학, 매수/매도 정책, 타이밍, 모델 원칙, 코드 원칙 전체 문서화 |

## 기존 버그 수정 (Phase 21.5, 이번 세션 초반)

| 파일 | 수정 내용 |
|------|----------|
| `live/signal_to_order.py` | 스탑로스 당일 재매수 차단 (`_stoploss_cooldown`), `add_stoploss_cooldown()` 메서드 |
| `tracking/trade_log.py` | `compute_turnover`에 `portfolio_value` 파라미터 추가 (0% 버그 수정) |
| `scheduler/daily_runner.py` | `_make_generator()` 헬퍼 (쿨다운 전파), 7곳 OrderGenerator 통일, turnover 계산 수정 |

---

## 50 Epoch 학습 결과

```
모델: AlphaTransformer (1.4M params, 128 d_model, 4 layers, 8 heads)
데이터: 348 tickers, 5년, 71 features, z-score 정규화
학습: 30 epochs (early stop at 30, patience 15)
시간: ~14분 (RTX 4060 Ti)

Validation:
  Dir Acc:  53.51%  PASS (threshold 52%)
  Rank IC:  0.0923  FAIL (threshold 0.10)

Test (out-of-sample):
  Dir Acc:  52.60%  PASS
  Rank IC:  0.0661  FAIL

v1 비교:
  Dir Acc: 48.9% (v1) → 52.6% (v2) = +3.7%p 개선
  원인: Feature 정규화 + mean pooling
```

### 게이트 결과: Dir Acc PASS, Rank IC FAIL

---

## v1 → v2 아키텍처 비교

| 항목 | v1 | v2 |
|------|----|----|
| **모델** | VAE→Transformer→GAN→RL (4단계) | AlphaTransformer 단일 모델 |
| **Pooling** | Last-token (마지막 1일만) | Mean pooling (전체 60일 평균) |
| **Confidence** | 없음 | Confidence head (방향 정확도 확률) |
| **피처 정규화** | 없음 (원본 스케일) | z-score per column (학습셋 기준) |
| **예측 대상** | 5일 상대수익률 | 1일 상대수익률 |
| **포지션** | 항상 9종목 (3섹터×3종목) | 0~3종목 (conviction 기반) |
| **현금** | 항상 5% 고정 | 0~100% (확신 없으면 전액 현금) |
| **이익 실현** | 없음 | +2.5% 절반, +5% 전량 |
| **손절** | -3% (너무 늦음) | -2% |
| **시간 청산** | 없음 | 2시간 내 +1% 미달 시 |
| **보유 기간** | 무제한 | 최대 3일, 기본 당일 |
| **거래 세션** | 7세션 (30+ 거래/일) | 2세션 (2~6 거래/일) |
| **Signal decay** | 24h half-life | 6h half-life (config) |
| **설정 파일** | 4개 (하드코딩 다수) | 1개 (system_config.yaml) |
| **백테스트** | 라이브와 불일치 | 라이브와 동일 함수 사용 |

---

## 미해결 TODO 목록

### 우선순위 1: 모델 게이트 통과 (Rank IC ≥ 0.10)

- [ ] **옵션 A**: 게이트 기준 완화 검토 (0.10 → 0.07, 현재 val 0.092 통과)
- [ ] **옵션 B**: 하이퍼파라미터 튜닝 (lr, dropout, d_model 256, n_layers 6)
- [ ] **옵션 C**: prediction_horizon 3일로 변경 후 재학습
- [ ] **옵션 D**: 피처 추가 (거래량 프로필, 섹터 상대강도 등)
- [ ] **옵션 E**: ListMLE/LambdaRank loss 도입 (pairwise → listwise)

### 우선순위 2: 통합 테스트

- [ ] `pipeline/train_pipeline_v2.py` end-to-end 실행 테스트
- [ ] `pipeline/signal_pipeline.py` 실제 데이터 추론 테스트
- [ ] `backtest/conviction_engine.py` 히스토리컬 백테스트 실행
- [ ] `live/executor.py` KIS sandbox 주문 테스트
- [ ] `scheduler/runner.py` KR 세션 풀 시뮬레이션

### 우선순위 3: 정리

- [ ] v1 코드 아카이브 (`_archive_v1/` 디렉토리로 이동)
- [ ] `main.py` v2 진입점으로 업데이트
- [ ] 불필요한 v1 설정 파일 정리
- [ ] Git 커밋 (v2 재구축 milestone)

### 우선순위 4: 배포 준비

- [ ] 모델 게이트 통과 후 → 백테스트 게이트 (Sharpe > 1.0)
- [ ] sandbox 5일 연속 운영 테스트
- [ ] 서버 배포 스크립트 업데이트
