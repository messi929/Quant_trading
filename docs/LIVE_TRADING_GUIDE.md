# 실전 트레이딩 가이드

## 아키텍처 개요

```
model inference (daily)
    │
    ▼
sector_weights (11개 섹터, 합=1.0)
    │  alpha=0.4 blending (40% 모델 + 60% equal-weight)
    ▼
OrderGenerator
    ├── domestic (KOSPI .KS/.KQ) → KIS domestic API → 실 체결
    └── overseas (NASDAQ)        → paper_trading=True → yfinance 가상 체결
    │
    ▼
TradeLogger (SQLite) → Streamlit Dashboard
    │
    ▼
weekly retrain (Transformer, ~30분)
monthly retrain (VAE→RL 전체, ~2시간)
```

**현재 모드** (2026-02-26 기준):
- 국내(KOSPI): KIS sandbox 실주문 (내일 09:10 KR Wave 1부터 첫 체결 예정)
- 해외(NASDAQ): `paper_trading=True` — yfinance 시세 기준 가상 체결, DB에 `mode='paper'` 기록

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
# 국내 계좌 (KIS domestic)
KIS_DOMESTIC_APP_KEY=your_app_key
KIS_DOMESTIC_APP_SECRET=your_app_secret
KIS_DOMESTIC_CANO=12345678
KIS_DOMESTIC_ACNT_PRDT_CD=01

# 해외 계좌 (KIS overseas) — sandbox USD $0이므로 paper_trading=True 사용
KIS_OVERSEAS_APP_KEY=your_app_key
KIS_OVERSEAS_APP_SECRET=your_app_secret
KIS_OVERSEAS_CANO=12345678

KIS_MODE=sandbox            # sandbox | production
```

**주의**: KIS sandbox 해외 모의투자 계좌는 USD 예수금 $0 — `paper_trading: true` 설정으로 yfinance 가상 체결을 사용합니다.

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

**일간 스케줄 (KST)**:

| 시각 | 작업 |
|------|------|
| 06:00 | 데이터 수집 (KOSPI + NASDAQ 증분) |
| 06:10 | [US] NASDAQ 종가 신호 생성 + 하락 매도 |
| 06:30 | [KR] 신호 생성 + 하락 매도 |
| 09:10 | [KR] Wave 1 (40%, domestic) |
| 11:00 | [KR] Wave 2 (35%, domestic) |
| 13:30 | [KR] Wave 3 (25%, domestic) |
| 16:00 | EOD 기록 (국내 잔고 기준) |
| 23:40 | [US] Wave 1 (40%, overseas/paper) |
| 02:00 | [US] Wave 2 (35%, overseas/paper) |
| 04:30 | [US] Wave 3 (25%, overseas/paper) |

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

## 6. 매매 방식

**현재**: `execution_market: "split"` — 개별 종목 직접 매매 (ETF 아님)

- 섹터당 모델 예측 상위 3 종목 선정 (`score > 0`인 종목만)
- KOSPI 종목 (`*.KS`, `*.KQ`) → domestic → KIS sandbox 실주문
- NASDAQ 종목 → overseas → paper_trading (yfinance 가상 체결)
- 섹터 배분 금액을 상위 3 종목에 균등 분배

**참고용 ETF 매핑** (ETF 모드 전환 시):

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

## 9. 서버 관리 명령어

```bash
# 서버 접속
ssh root@77.42.78.9

# 로그 실시간 확인
ssh root@77.42.78.9 "tail -f /opt/quant/logs/quant_$(date +%Y-%m-%d).log"

# 서비스 상태
ssh root@77.42.78.9 "systemctl status quant-trading --no-pager"

# 파일 배포 (변경 후)
scp <파일> root@77.42.78.9:/opt/quant/<경로>
ssh root@77.42.78.9 "systemctl restart quant-trading"

# KOSPI parquet 재구축 (서버에서)
ssh root@77.42.78.9 "cd /opt/quant && python3 <rebuild_script>"
```

## 10. 주의사항

1. **sandbox 모드 유지** — `live_config.yaml`의 `broker.mode: "sandbox"`, `paper_trading: true`
2. **해외 주문은 paper trading** — KIS sandbox USD $0, 가상 체결 (yfinance 시세 기준)
3. **실전 전환 체크리스트**:
   - [ ] sandbox로 1주일 이상 KR 신호 및 주문 검증
   - [ ] dir_acc > 52% 확인
   - [ ] `paper_trading: false`, `mode: "production"` 전환
   - [ ] 소액(1백만원)으로 먼저 실전 테스트
4. **API 호출 제한**: KIS 토큰 분당 1회 → `get_token()` 3회 재시도(5초 간격) 구현됨
5. **KOSPI 섹터 커버리지**: 950종목 중 203종목만 섹터 분류 (21.4%) — unknown 종목은 신호 생성 제외
6. **.env 파일은 절대 git commit 금지**
