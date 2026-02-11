# Next Steps - 다음 작업 로드맵

## 즉시 해야 할 일 (Phase 0: 환경 설정)

### Step 1: Miniconda 설치
```
1. https://docs.anaconda.com/miniconda/ 에서 Windows 64-bit 다운로드
2. 설치 시 "Add to PATH" 체크
3. 터미널 재시작 후 `conda --version` 확인
```

### Step 2: 프로젝트 환경 생성
```bash
cd C:\src\Qunat_trading
setup_env.bat
# 또는 수동:
conda create -n quant python=3.11 -y
conda activate quant
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Step 3: GPU 동작 확인
```python
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# 기대 출력: True, NVIDIA GeForce RTX 4060 Ti
```

---

## 단기 작업 (1-2주)

### Step 4: 데이터 수집 테스트
```bash
# 소규모 테스트 (일부 종목만)
python -c "
from data.collector import MarketDataCollector
c = MarketDataCollector(history_years=5)
kospi = c.get_kospi_tickers()
print(f'KOSPI tickers: {len(kospi)}')
"
```
- [ ] KOSPI 종목 목록 정상 수집 확인
- [ ] NASDAQ 종목 목록 정상 수집 확인
- [ ] OHLCV 데이터 다운로드 테스트 (10종목, 1년)
- [ ] 전체 수집 실행 (예상 소요 시간: 2-4시간)

### Step 5: 데이터 파이프라인 검증
- [ ] `processor.py` - 결측치 처리 동작 확인
- [ ] `sector_classifier.py` - 섹터 분류 정확도 확인 (목표: >80%)
- [ ] `feature_engineer.py` - 40+ 피처 정상 계산 확인
- [ ] `dataset.py` - DataLoader 배치 형태 확인 (batch, seq_len, n_features)

### Step 6: 전체 학습 파이프라인 시작
```bash
python main.py train
```
- [ ] Phase 1 (데이터) 완료 확인
- [ ] Phase 2 (VAE) 학습 곡선 모니터링
- [ ] Phase 3 (Transformer) IC 값 추적
- [ ] Phase 4 (GAN) Wasserstein distance 수렴
- [ ] Phase 5 (RL) 보상 수렴
- [ ] Phase 6 (앙상블 저장) 확인

---

## 중기 작업 (2-4주)

### Step 7: 모델 튜닝 및 최적화
- [ ] VAE latent_dim 실험 (16, 32, 64)
- [ ] Transformer attention 시각화로 중요 시점/지표 파악
- [ ] GAN 합성 데이터 품질 평가 (KS test)
- [ ] RL reward function 실험 (Sharpe vs Sortino)
- [ ] 학습 중 TensorBoard 로그 분석

### Step 8: 백테스트 평가
```bash
python main.py backtest
python main.py backtest --walk-forward
```
- [ ] Out-of-sample 성과 확인
- [ ] Walk-forward 결과 분석
- [ ] 벤치마크(동일 가중) 대비 초과 수익 확인
- [ ] 연간/월별 수익률 분석
- [ ] 최대 낙폭(MDD) 및 회복 기간 분석

### Step 9: 발견된 지표 분석
- [ ] `indicators/evaluator.py`로 지표 품질 평가
- [ ] 높은 novelty 점수를 가진 "진짜 새로운" 지표 식별
- [ ] t-SNE로 잠재 공간 시각화
- [ ] 지표별 예측력(IC) 시계열 안정성 확인

---

## 장기 작업 (1-2개월)

### Step 10: 고도화
- [ ] 실시간 데이터 피드 연결 (WebSocket)
- [ ] 추론 파이프라인 스케줄링 (daily cron)
- [ ] Sector rotation 시그널 강화
- [ ] 다중 타임프레임 (일봉 + 주봉) 지원
- [ ] 앙상블 가중치 동적 조정 (시장 레짐 기반)

### Step 11: 리스크 관리 강화
- [ ] 상관관계 기반 포지션 한도
- [ ] 테일 리스크 헷지 전략
- [ ] 변동성 레짐 감지 → 자동 디레버리지
- [ ] 스트레스 테스트 (2008, 2020, 2022 시나리오)

### Step 12: 인프라
- [ ] MLflow/W&B로 실험 추적
- [ ] 모델 버전 관리 체계화
- [ ] Docker 컨테이너화
- [ ] 알림 시스템 (Telegram/Slack)

---

## 알려진 이슈 및 주의사항

1. **KOSPI 데이터**: pykrx는 KRX 서버 상태에 따라 불안정할 수 있음. 실패 시 재시도 로직 필요할 수 있음
2. **NASDAQ 전종목**: 현재 S&P500 + NASDAQ-100으로 제한됨. 전체 NASDAQ 필요 시 별도 소스 필요
3. **VRAM 8GB 제한**: 배치 크기와 모델 크기에 주의. OOM 발생 시 `physical_batch_size` 줄이기
4. **시간 소요**: 전체 학습 파이프라인은 수일 소요될 수 있음. 각 Phase 별로 체크포인트 저장됨
5. **yfinance 속도 제한**: 대량 다운로드 시 429 에러 가능. batch_size 조절 필요
