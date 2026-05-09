#!/bin/bash
# V3 Vol Expansion Trader — Git-based Server Deployment
# Usage: bash deploy_v3_git.sh [server_ip]
#
# 동작:
#   1. server 기존 /opt/quant → /opt/quant_v33_backup_<ts>
#   2. /tmp에 fresh git clone
#   3. runtime artifacts 보존 (.env, models, saved_models, raw data)
#   4. /tmp/quant_new → /opt/quant 교체
#   5. systemd unit 4개 (service + 3 timer) 자동 설치
#   6. service restart + status 확인
#
# Windows local 환경에서 rsync 없이 git clone 기반으로 deploy.
# Idempotent — 여러 번 실행 안전.

set -euo pipefail

SERVER=${1:-"77.42.78.9"}
USER="root"
REMOTE_DIR="/opt/quant"
REPO_URL="https://github.com/messi929/Quant_trading.git"
SERVICE_NAME="quant-trading-v3"
TS=$(date +%Y%m%d_%H%M%S)

echo "=== V3.3 Git-based Deployment to $SERVER ==="

ssh $USER@$SERVER "
set -euo pipefail

echo '[1/7] Backing up current /opt/quant...'
if [ -d $REMOTE_DIR ]; then
    cp -a $REMOTE_DIR ${REMOTE_DIR}_v33_backup_${TS}
    echo \"  Backup: ${REMOTE_DIR}_v33_backup_${TS}\"
fi

echo '[2/7] Cloning fresh from GitHub...'
rm -rf /tmp/quant_new
git clone --depth 1 $REPO_URL /tmp/quant_new
echo '  Cloned successfully'

echo '[3/7] Preserving runtime artifacts...'
mkdir -p /tmp/quant_new/v3/saved_models
mkdir -p /tmp/quant_new/v3/data/raw

# .env (KIS API keys)
[ -f $REMOTE_DIR/.env ] && cp $REMOTE_DIR/.env /tmp/quant_new/.env && echo '  .env preserved'

# Model checkpoints (large, .gitignore'd)
if [ -d $REMOTE_DIR/v3/saved_models ]; then
    cp $REMOTE_DIR/v3/saved_models/*.pt /tmp/quant_new/v3/saved_models/ 2>/dev/null && echo '  *.pt preserved' || true
    cp $REMOTE_DIR/v3/saved_models/*.json /tmp/quant_new/v3/saved_models/ 2>/dev/null && echo '  *.json preserved' || true
    cp $REMOTE_DIR/v3/saved_models/*.jsonl /tmp/quant_new/v3/saved_models/ 2>/dev/null && echo '  *.jsonl preserved' || true
    [ -d $REMOTE_DIR/v3/saved_models/no_trade_logs ] && cp -r $REMOTE_DIR/v3/saved_models/no_trade_logs /tmp/quant_new/v3/saved_models/ && echo '  no_trade_logs preserved' || true
fi

# Raw OHLCV / macro data
if [ -d $REMOTE_DIR/v3/data/raw ]; then
    cp -r $REMOTE_DIR/v3/data/raw/* /tmp/quant_new/v3/data/raw/ 2>/dev/null && echo '  raw data preserved' || true
fi

# Existing alpha-retrain config (Phase 25.2 적용된 거 보존)
[ -d $REMOTE_DIR/v3/config/alpha_weights_history ] && cp -r $REMOTE_DIR/v3/config/alpha_weights_history /tmp/quant_new/v3/config/ && echo '  alpha_weights_history preserved' || true

echo '[4/7] Replacing /opt/quant...'
mv $REMOTE_DIR ${REMOTE_DIR}_old_${TS}
mv /tmp/quant_new $REMOTE_DIR
echo \"  Old: ${REMOTE_DIR}_old_${TS} (cleanup later)\"

echo '[5/7] Setting up Python venv...'
cd $REMOTE_DIR
if [ ! -d venv ]; then
    # venv가 없으면 backup에서 복사 (더 빠름)
    if [ -d ${REMOTE_DIR}_old_${TS}/venv ]; then
        cp -r ${REMOTE_DIR}_old_${TS}/venv $REMOTE_DIR/venv
        echo '  venv copied from backup'
    else
        python3 -m venv venv
        source venv/bin/activate
        pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
        pip install -q pandas numpy scipy loguru pyyaml pydantic yfinance requests beautifulsoup4 tqdm pyarrow
        echo '  venv created from scratch'
    fi
fi

echo '[6/7] Installing systemd units...'

# Main service
cat > /etc/systemd/system/$SERVICE_NAME.service << 'SVCEOF'
[Unit]
Description=V3 Vol Expansion Trader
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/quant
Environment=PYTHONPATH=/opt/quant
EnvironmentFile=/opt/quant/.env
ExecStart=/opt/quant/venv/bin/python -u v3/scripts/run_live.py --mode daemon
Restart=always
RestartSec=30
StandardOutput=append:/var/log/quant-v3.log
StandardError=append:/var/log/quant-v3-error.log

[Install]
WantedBy=multi-user.target
SVCEOF

# V3.3 Timer 1: calibration auto-retrain (월 1회)
cat > /etc/systemd/system/calibration-retrain.service << 'CSVEOF'
[Unit]
Description=V3.3 monthly calibration pipeline
After=network.target alpha-retrain.service
Wants=alpha-retrain.service

[Service]
Type=oneshot
User=root
WorkingDirectory=/opt/quant
Environment=PYTHONPATH=/opt/quant
EnvironmentFile=/opt/quant/.env
ExecStart=/opt/quant/venv/bin/python -u v3/scripts/run_calibration_pipeline.py \\
    --ohlcv-path data/research/ohlcv_panel.parquet \\
    --macro-path data/research/macro_pctl.parquet \\
    --vol-pred-path data/research/vol_predictions.parquet \\
    --years-back 5 --oos-days 180 \\
    --output-dir data/research --report-dir research/reports \\
    --latest-calibration-path v3/config/edge_calibration.json
StandardOutput=append:/var/log/calibration-retrain.log
StandardError=append:/var/log/calibration-retrain-error.log
CSVEOF

cat > /etc/systemd/system/calibration-retrain.timer << 'CTMEOF'
[Unit]
Description=Monthly V3.3 calibration retrain (1st @ 07:00 KST)

[Timer]
OnCalendar=*-*-01 07:00:00
Persistent=true

[Install]
WantedBy=timers.target
CTMEOF

# V3.3 Timer 2: daily diagnostic report
cat > /etc/systemd/system/v33-daily-report.service << 'DRSEOF'
[Unit]
Description=V3.3 daily diagnostic report
After=network.target

[Service]
Type=oneshot
User=root
WorkingDirectory=/opt/quant
Environment=PYTHONPATH=/opt/quant
ExecStart=/opt/quant/venv/bin/python -u v3/scripts/run_daily_report.py \\
    --log-dir v3/saved_models \\
    --output-dir research/reports/daily
StandardOutput=append:/var/log/v33-daily-report.log
StandardError=append:/var/log/v33-daily-report-error.log
DRSEOF

cat > /etc/systemd/system/v33-daily-report.timer << 'DRTEOF'
[Unit]
Description=V3.3 daily report (16:00 KST)

[Timer]
OnCalendar=*-*-* 16:00:00
Persistent=true

[Install]
WantedBy=timers.target
DRTEOF

# V3.3 Timer 3: rollback check
cat > /etc/systemd/system/v33-rollback-check.service << 'RBSEOF'
[Unit]
Description=V3.3 daily rollback check (1w PnL -2% → feature OFF)
After=network.target v33-daily-report.service
Wants=v33-daily-report.service

[Service]
Type=oneshot
User=root
WorkingDirectory=/opt/quant
Environment=PYTHONPATH=/opt/quant
ExecStart=/opt/quant/venv/bin/python -u v3/scripts/run_rollback_check.py \\
    --window-days 7 --threshold -0.02
StandardOutput=append:/var/log/v33-rollback.log
StandardError=append:/var/log/v33-rollback-error.log
RBSEOF

cat > /etc/systemd/system/v33-rollback-check.timer << 'RBTEOF'
[Unit]
Description=V3.3 rollback check (16:30 KST)

[Timer]
OnCalendar=*-*-* 16:30:00
Persistent=true

[Install]
WantedBy=timers.target
RBTEOF

systemctl daemon-reload

# Enable + start
systemctl enable $SERVICE_NAME 2>/dev/null
systemctl enable calibration-retrain.timer 2>/dev/null
systemctl enable v33-daily-report.timer 2>/dev/null
systemctl enable v33-rollback-check.timer 2>/dev/null
systemctl start calibration-retrain.timer 2>/dev/null
systemctl start v33-daily-report.timer 2>/dev/null
systemctl start v33-rollback-check.timer 2>/dev/null

echo '  systemd units installed + enabled'

echo '[7/7] Restarting quant-trading-v3...'
systemctl restart $SERVICE_NAME
sleep 3
systemctl status $SERVICE_NAME --no-pager | head -10
echo ''
echo '=== Deployment Complete ==='
echo \"  Backup:  ${REMOTE_DIR}_v33_backup_${TS}\"
echo \"  Old:     ${REMOTE_DIR}_old_${TS} (manual cleanup recommended)\"
echo '  Logs:    tail -f /var/log/quant-v3.log'
echo '  Timers:  systemctl list-timers | grep -E alpha\\|calibration\\|v33'
"

echo ""
echo "=== Local Deploy Script Done ==="
echo "Run verification:"
echo "  ssh $USER@$SERVER 'systemctl status $SERVICE_NAME --no-pager'"
echo "  ssh $USER@$SERVER 'systemctl list-timers | grep -E alpha\\|calibration\\|v33'"
