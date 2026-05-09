#!/bin/bash
# V3 Vol Expansion Trader — Server Deployment Script
# Usage: bash deploy_v3.sh [server_ip]

set -euo pipefail

SERVER=${1:-"77.42.78.9"}
USER="root"
REMOTE_DIR="/opt/quant"
VENV_PYTHON="/opt/quant/venv/bin/python"
SERVICE_NAME="quant-trading-v3"

echo "=== V3 Deployment to $SERVER ==="

# 1. Backup existing V2 (if not already backed up)
echo "[1/6] Backing up V2..."
ssh $USER@$SERVER "
    if [ ! -d /opt/quant_v2_backup ]; then
        cp -r $REMOTE_DIR /opt/quant_v2_backup 2>/dev/null || true
        echo '  V2 backed up to /opt/quant_v2_backup'
    else
        echo '  V2 backup already exists'
    fi
"

# 2. Upload V3 code
echo "[2/6] Uploading V3 code..."
rsync -avz --delete \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='data/raw/*.parquet' \
    --exclude='data/processed/*.parquet' \
    --exclude='logs/' \
    --exclude='results/' \
    v3/ $USER@$SERVER:$REMOTE_DIR/v3/

# Upload shared files
scp broker/kis_api.py $USER@$SERVER:$REMOTE_DIR/broker/kis_api.py
scp .env $USER@$SERVER:$REMOTE_DIR/.env 2>/dev/null || echo "  .env not found (manual setup needed)"

# 3. Upload model + normalizer
echo "[3/6] Uploading model checkpoint..."
scp v3/saved_models/vol_transformer_epoch*.pt $USER@$SERVER:$REMOTE_DIR/v3/saved_models/
scp v3/saved_models/normalizer_stats.json $USER@$SERVER:$REMOTE_DIR/v3/saved_models/
scp v3/saved_models/feature_config.json $USER@$SERVER:$REMOTE_DIR/v3/saved_models/

# 4. Setup venv on server
echo "[4/6] Setting up Python environment..."
ssh $USER@$SERVER "
    cd $REMOTE_DIR
    if [ ! -d venv ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -q pandas numpy scipy loguru pyyaml pydantic yfinance requests beautifulsoup4 tqdm
    echo '  Dependencies installed'
"

# 5. Create systemd service
echo "[5/6] Creating systemd service..."
ssh $USER@$SERVER "cat > /etc/systemd/system/$SERVICE_NAME.service << 'SVCEOF'
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

systemctl daemon-reload
echo '  Service created'
"

# 6. Start service
echo "[6/6] Starting service..."
ssh $USER@$SERVER "
    systemctl enable $SERVICE_NAME
    systemctl restart $SERVICE_NAME
    sleep 2
    systemctl status $SERVICE_NAME --no-pager || true
    echo ''
    echo 'Recent logs:'
    tail -5 /var/log/quant-v3.log 2>/dev/null || echo '  (no logs yet)'
"

# 7. (V3.3 P5) Install calibration auto-retrain timer
echo "[7/7] Installing V3.3 calibration auto-retrain timer..."
ssh $USER@$SERVER "
    cat > /etc/systemd/system/calibration-retrain.service << 'CSVEOF'
[Unit]
Description=V3.3 monthly calibration pipeline (build->calibrate->validate)
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
    --years-back 5 \\
    --oos-days 180 \\
    --output-dir data/research \\
    --report-dir research/reports \\
    --latest-calibration-path v3/config/edge_calibration.json
StandardOutput=append:/var/log/calibration-retrain.log
StandardError=append:/var/log/calibration-retrain-error.log
CSVEOF

    cat > /etc/systemd/system/calibration-retrain.timer << 'CTMEOF'
[Unit]
Description=Monthly V3.3 calibration auto-retrain (1st @ 07:00 KST)

[Timer]
OnCalendar=*-*-01 07:00:00
Persistent=true

[Install]
WantedBy=timers.target
CTMEOF

    systemctl daemon-reload
    systemctl enable calibration-retrain.timer 2>/dev/null || true
    systemctl start calibration-retrain.timer 2>/dev/null || true
    echo '  Calibration timer installed; first run on next 1st @ 07:00 KST'
    echo '  NOTE: requires data/research/ohlcv_panel.parquet etc. (manual prep)'
"

echo ""
echo "=== V3 Deployment Complete ==="
echo "  Server:  $SERVER"
echo "  Service: $SERVICE_NAME"
echo "  Logs:    ssh $USER@$SERVER tail -f /var/log/quant-v3.log"
echo "  Status:  ssh $USER@$SERVER systemctl status $SERVICE_NAME"
echo "  Stop:    ssh $USER@$SERVER systemctl stop $SERVICE_NAME"
echo "  V2 restore: ssh $USER@$SERVER 'rm -rf /opt/quant && mv /opt/quant_v2_backup /opt/quant'"
echo ""
echo "V3.3 timers (manual check):"
echo "  alpha-retrain:        ssh $USER@$SERVER systemctl status alpha-retrain.timer"
echo "  calibration-retrain:  ssh $USER@$SERVER systemctl status calibration-retrain.timer"
