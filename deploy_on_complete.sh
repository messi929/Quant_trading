#!/bin/bash
# 학습 완료 감지 → 서버 자동 배포 스크립트

TODAY=$(date '+%Y-%m-%d')
ENSEMBLE=/c/src/Qunat_trading/saved_models/ensemble.pt
SERVER=root@77.42.78.9
SERVER_DIR=/opt/quant
SSH_KEY=~/.ssh/id_ed25519
DEPLOY_LOG=/c/src/Qunat_trading/logs/deploy_${TODAY}.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 배포 대기 시작..." | tee "$DEPLOY_LOG"

# 스크립트 시작 시각 기록 (epoch seconds)
START_TIME=$(date +%s)

# ensemble.pt가 이 스크립트 시작 이후 갱신될 때까지 대기 (최대 5시간)
TIMEOUT=18000
ELAPSED=0
INTERVAL=30

while [ $ELAPSED -lt $TIMEOUT ]; do
    if [ -f "$ENSEMBLE" ]; then
        ENSEMBLE_MTIME=$(stat -c %Y "$ENSEMBLE" 2>/dev/null)
        if [ -n "$ENSEMBLE_MTIME" ] && [ "$ENSEMBLE_MTIME" -ge "$START_TIME" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ensemble.pt 생성 감지 — 30초 후 배포 시작" | tee -a "$DEPLOY_LOG"
            sleep 30  # 파일 쓰기 완전히 끝날 때까지
            break
        fi
    fi
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 타임아웃 — 배포 중단" | tee -a "$DEPLOY_LOG"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 서버 배포 시작 ===" | tee -a "$DEPLOY_LOG"

# 1. 모델 파일 복사
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 모델 파일 복사 중..." | tee -a "$DEPLOY_LOG"
scp -i "$SSH_KEY" -r /c/src/Qunat_trading/saved_models/ "$SERVER:$SERVER_DIR/" >> "$DEPLOY_LOG" 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 모델 복사 실패!" | tee -a "$DEPLOY_LOG"
    exit 1
fi

# 2. 처리된 데이터(processed_data.parquet) 복사 — KOSPI 섹터 포함된 새 버전
echo "[$(date '+%Y-%m-%d %H:%M:%S')] processed_data.parquet 복사 중..." | tee -a "$DEPLOY_LOG"
scp -i "$SSH_KEY" /c/src/Qunat_trading/data/processed/processed_data.parquet \
    "$SERVER:$SERVER_DIR/data/processed/processed_data.parquet" >> "$DEPLOY_LOG" 2>&1

# 3. 코드 변경사항 복사 (pipeline, broker, scheduler)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 코드 파일 복사 중..." | tee -a "$DEPLOY_LOG"
scp -i "$SSH_KEY" /c/src/Qunat_trading/pipeline/inference_pipeline.py \
    "$SERVER:$SERVER_DIR/pipeline/inference_pipeline.py" >> "$DEPLOY_LOG" 2>&1
scp -i "$SSH_KEY" /c/src/Qunat_trading/pipeline/train_pipeline.py \
    "$SERVER:$SERVER_DIR/pipeline/train_pipeline.py" >> "$DEPLOY_LOG" 2>&1

# 4. 서비스 재시작
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 서버 서비스 재시작..." | tee -a "$DEPLOY_LOG"
ssh -i "$SSH_KEY" "$SERVER" "systemctl restart quant-trading && sleep 3 && systemctl status quant-trading --no-pager | tail -5" >> "$DEPLOY_LOG" 2>&1

if [ $? -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 배포 완료 ===" | tee -a "$DEPLOY_LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 서버에서 다음 06:10 신호 생성부터 새 모델이 적용됩니다." | tee -a "$DEPLOY_LOG"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 서비스 재시작 실패!" | tee -a "$DEPLOY_LOG"
    exit 1
fi
