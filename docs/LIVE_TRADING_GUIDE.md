# 실전 트레이딩 가이드

## 아키텍처 개요

```
model inference (daily)
    │
    ▼
sector_weights (11개 섹터, 합=1.0)
    │  alpha=0.4 blending (40% 모델 + 60% equal-weight)
    ▼
OrderGenerator → KIS API → KODEX ETF 매수/매도
    │
    ▼
TradeLogger (SQLite) → Streamlit Dashboard
    │
    ▼
weekly retrain (Transformer, ~30분)
monthly retrain (VAE→RL 전체, ~2시간)
```

## 1. 초기 설정

### 1-1. 환경 세팅
```bat
cd C:\src\Qunat_trading
setup_live.bat
```
`setup_live.bat`이 하는 일:
- conda `quant_trading` 환경 확인/생성
- 라이브 트레이딩 패키지 설치
- `.env` 파일 생성 (KIS 자격증명 입력)
- Windows 작업 스케줄러 등록 (선택)

### 1-2. KIS API 발급
1. [KIS Developers](https://apiportal.koreainvestment.com) 접속
2. "나의 API 신청" → OpenAPI 신청
3. AppKey / AppSecret 발급
4. **모의투자 계좌** 먼저 사용 권장

### 1-3. .env 파일 (수동 작성 시)
```
KIS_APP_KEY=your_app_key
KIS_APP_SECRET=your_app_secret
KIS_CANO=12345678          # 계좌번호 8자리
KIS_ACNT_PRDT_CD=01        # 계좌상품코드
KIS_MODE=sandbox            # sandbox | production
```

## 2. 모델 훈련 (최초 1회)

```bash
# Fast 설정으로 전체 훈련 (~2-4시간)
python main.py train --config config/settings_fast.yaml

# 백테스트 확인
python main.py backtest --config config/settings_fast.yaml
```

## 3. 일간 운용

### 수동 실행 (테스트용)
```bash
# 1단계: 최신 데이터 수집
python scheduler/daily_runner.py --step collect

# 2단계: 신호 생성 (섹터 배분 출력)
python scheduler/daily_runner.py --step signal

# 3단계: 리밸런싱 주문 (월요일에만 실행)
python scheduler/daily_runner.py --step order

# 4단계: 종가 기록
python scheduler/daily_runner.py --step eod

# 또는 전체 순서 실행
python scheduler/daily_runner.py
```

### 자동화 (데몬 모드)
```bash
python scheduler/daily_runner.py --daemon
```
설정된 시간(06:00/06:30/08:50/16:10)에 자동 실행됩니다.

### Windows 작업 스케줄러
`setup_live.bat` 실행 시 자동 등록됩니다.
수동 확인: 작업 스케줄러 → "AlphaSignal_*" 태스크

## 4. 재학습

```bash
# 주간 파인튜닝 (Transformer, ~30분)
python scheduler/retrain_runner.py --mode weekly

# 월간 전체 재학습 (VAE→RL, ~2-3시간)
python scheduler/retrain_runner.py --mode monthly

# 재학습 필요 여부 자동 판단
python scheduler/retrain_runner.py --mode check

# 성능 저하 시 롤백
python scheduler/retrain_runner.py --mode rollback
```

**자동 롤백 조건**: 재학습 후 `dir_acc < 48%`이면 이전 모델로 자동 복원

## 5. 대시보드

```bash
streamlit run dashboard/app.py
# → http://localhost:8501
```

대시보드 항목:
- 포트폴리오 가치, 누적 수익률, Sharpe, MDD, 승률
- 수익 곡선 vs 벤치마크
- 섹터 배분 파이 차트
- 방향 예측 정확도 게이지 (기준: 50%)
- 일별 수익률 바 차트
- 최근 거래 내역
- 재학습 이력

## 6. 섹터 → ETF 매핑

| 섹터 | KOSPI ETF | NASDAQ ETF |
|------|-----------|------------|
| Energy | KODEX 에너지화학 (117460) | XLE |
| Materials | KODEX 에너지화학 (117460) | XLB |
| Industrials | KODEX 건설 (214830) | XLI |
| Consumer Disc. | KODEX 소비재 (091220) | XLY |
| Consumer Sta. | KODEX 소비재 (091220) | XLP |
| Healthcare | KODEX 헬스케어 (266410) | XLV |
| Financials | KODEX 금융 (091160) | XLF |
| IT | KODEX IT (261110) | XLK |
| Comm. Svcs | KODEX IT (261110) | XLC |
| Utilities | KODEX 유틸리티 (337140) | XLU |
| Real Estate | KODEX 리츠 (395400) | XLRE |

`config/live_config.yaml`의 `trading.execution_market`으로 KOSPI/NASDAQ 선택

## 7. 리스크 관리

| 조건 | 조치 |
|------|------|
| 일간 손실 > 3% | 당일 주문 중단 |
| 총 손실 > 15% | 거래 중단 |
| MDD > 10% | 거래 중단 |
| RL Env MDD > 10% | 포지션 50% 축소 |
| 재학습 후 dir_acc < 48% | 자동 모델 롤백 |

## 8. 파일 구조

```
config/
  live_config.yaml          # 라이브 트레이딩 설정
  settings_fast.yaml        # 모델 학습 설정
broker/
  kis_api.py                # KIS REST API 래퍼
live/
  sector_instruments.py     # 섹터 → ETF 매핑
  signal_to_order.py        # 신호 → 주문 변환
tracking/
  trade_log.py              # SQLite 거래/성과 기록
  trades.db                 # DB 파일 (자동 생성)
scheduler/
  daily_runner.py           # 일간 자동화
  retrain_runner.py         # 재학습 스케줄러
dashboard/
  app.py                    # Streamlit 대시보드
.env                        # KIS API 자격증명 (gitignore)
setup_live.bat              # 최초 설정 스크립트
run_step.bat                # 스케줄러 헬퍼
run_retrain.bat             # 재학습 헬퍼
```

## 9. 주의사항

1. **반드시 sandbox 모드로 먼저 테스트** — `live_config.yaml`의 `broker.mode: "sandbox"` 유지
2. **실전 전환 체크리스트**:
   - [ ] sandbox로 1주일 이상 신호 검증
   - [ ] dir_acc > 52% 확인
   - [ ] 대시보드에서 섹터 배분 정상 확인
   - [ ] 소액(1백만원)으로 먼저 실전 테스트
3. **API 호출 제한**: KIS는 초당 20회 제한 — 대량 주문 시 자동 대기
4. **KODEX ETF 유동성**: 일부 소형 ETF는 스프레드가 클 수 있음
5. **.env 파일은 절대 git commit 금지** (`.gitignore`에 추가 필요)
