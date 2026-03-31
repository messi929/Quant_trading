#!/bin/bash
# V2 서버 배포 스크립트
# Usage: bash deploy_v2.sh

set -e

SERVER=root@77.42.78.9
SERVER_DIR=/opt/quant
SSH_KEY=~/.ssh/id_ed25519
LOG="logs/deploy_v2_$(date '+%Y-%m-%d').log"

echo "[$(date '+%H:%M:%S')] V2 배포 시작" | tee "$LOG"

# 1. V1 백업
echo "[$(date '+%H:%M:%S')] V1 백업 중..." | tee -a "$LOG"
ssh -i "$SSH_KEY" "$SERVER" "cp -r $SERVER_DIR ${SERVER_DIR}_v1_backup_$(date +%Y%m%d)" 2>&1 | tee -a "$LOG"

# 2. 모델 + normalizer + feature_cols 복사
echo "[$(date '+%H:%M:%S')] 모델 파일 복사..." | tee -a "$LOG"
scp -i "$SSH_KEY" saved_models/alpha_transformer_*.pt "$SERVER:$SERVER_DIR/saved_models/" 2>&1 | tee -a "$LOG"
scp -i "$SSH_KEY" saved_models/normalizer_stats.json "$SERVER:$SERVER_DIR/saved_models/" 2>&1 | tee -a "$LOG"
scp -i "$SSH_KEY" saved_models/feature_cols.json "$SERVER:$SERVER_DIR/saved_models/" 2>&1 | tee -a "$LOG"

# 3. V2 코드 복사
echo "[$(date '+%H:%M:%S')] V2 코드 복사..." | tee -a "$LOG"
scp -i "$SSH_KEY" config/system_config.yaml config/config_loader.py "$SERVER:$SERVER_DIR/config/" 2>&1 | tee -a "$LOG"
scp -i "$SSH_KEY" models/alpha_model.py models/alpha_trainer.py "$SERVER:$SERVER_DIR/models/" 2>&1 | tee -a "$LOG"
scp -i "$SSH_KEY" pipeline/signal_pipeline.py pipeline/train_pipeline_v2.py "$SERVER:$SERVER_DIR/pipeline/" 2>&1 | tee -a "$LOG"
scp -i "$SSH_KEY" strategy/stock_signal.py "$SERVER:$SERVER_DIR/strategy/" 2>&1 | tee -a "$LOG"
scp -i "$SSH_KEY" scheduler/runner.py "$SERVER:$SERVER_DIR/scheduler/" 2>&1 | tee -a "$LOG"
scp -i "$SSH_KEY" live/executor.py "$SERVER:$SERVER_DIR/live/" 2>&1 | tee -a "$LOG"
scp -i "$SSH_KEY" data/normalizer.py data/dataset.py "$SERVER:$SERVER_DIR/data/" 2>&1 | tee -a "$LOG"
scp -i "$SSH_KEY" backtest/conviction_engine.py backtest/intraday_sim.py "$SERVER:$SERVER_DIR/backtest/" 2>&1 | tee -a "$LOG"
scp -i "$SSH_KEY" run_v2.py "$SERVER:$SERVER_DIR/" 2>&1 | tee -a "$LOG"

# 4. 서버 패키지 확인
echo "[$(date '+%H:%M:%S')] 서버 패키지 확인..." | tee -a "$LOG"
ssh -i "$SSH_KEY" "$SERVER" "pip3 install -q schedule loguru 2>&1" | tee -a "$LOG"

# 5. systemd 서비스 업데이트 + 재시작
echo "[$(date '+%H:%M:%S')] 서비스 재시작..." | tee -a "$LOG"
ssh -i "$SSH_KEY" "$SERVER" "
    # ExecStart를 run_v2.py로 변경
    sed -i 's|ExecStart=.*|ExecStart=/usr/bin/python3 /opt/quant/run_v2.py|' /etc/systemd/system/quant-trading.service
    # QUANT_ROOT 환경변수 추가 (없으면)
    grep -q 'QUANT_ROOT' /etc/systemd/system/quant-trading.service || \
        sed -i '/\[Service\]/a Environment=QUANT_ROOT=/opt/quant' /etc/systemd/system/quant-trading.service
    systemctl daemon-reload
    systemctl restart quant-trading
    sleep 3
    systemctl status quant-trading --no-pager | tail -10
" 2>&1 | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] === V2 배포 완료 ===" | tee -a "$LOG"
echo "다음 스케줄: KR 06:10 신호 → 09:10 매수, US 22:00 신호 → 23:40 매수" | tee -a "$LOG"
