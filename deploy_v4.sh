#!/bin/bash
# V4 KOSDAQ Momentum Trader — systemd 배포 (daily paper session)
# Usage: bash deploy_v4.sh [server_ip]
#
# 전제: /opt/quant 에 최신 코드 존재 (git pull 또는 deploy_v3_git.sh 후).
# 동작:
#   1. **별도 venv_v4** 생성 + V4 의존 설치 (V3 venv 의 numpy1.26/torch 격리 — 라이브 무영향)
#   2. systemd 유닛 설치 — quant-v4-korea.service(oneshot) + .timer(평일 09:05 KST)
#   3. daemon-reload (※ enable/start 는 수동 — 계좌리셋+smoke 후)
#
# ⚠️ V4는 FinanceDataReader(numpy 2.x) 필요 → V3 venv(numpy 1.26+torch) 와 충돌.
#    별도 venv_v4 로 격리하여 라이브 V3 에 영향 0.
#
# ⚠️ 가동 전 필수 (KR 장중, 월요일):
#   1) 계좌 리셋:  python v4/scripts/reset_sandbox_account.py --execute
#   2) plan smoke: python v4/scripts/run_v4_live.py --mode once --no-execute
#   3) 실주문 smoke: python v4/scripts/run_v4_live.py --mode once   (소량 검증)
#   4) 위 OK 후 timer enable (스크립트 마지막 출력 명령)
#
# V4 = KOSDAQ ensemble momentum + 200d SMA gate + vol-target. sandbox 국내 paper.

set -euo pipefail

SERVER=${1:-"77.42.78.9"}
USER="root"
REMOTE_DIR="/opt/quant"
SVC="quant-v4-korea"

echo "=== V4 KOSDAQ systemd 배포 → $SERVER ==="

ssh $USER@$SERVER "
set -euo pipefail
cd $REMOTE_DIR

echo '[1/3] 별도 venv_v4 생성 + V4 의존 설치 (V3 venv 격리)...'
if [ ! -d venv_v4 ]; then
    python3 -m venv venv_v4
    echo '  venv_v4 생성'
fi
./venv_v4/bin/pip install -q --upgrade pip 2>&1 | tail -1 || true
./venv_v4/bin/pip install -q finance-datareader pandas numpy scipy loguru pyyaml pydantic pyarrow requests python-dotenv beautifulsoup4 lxml pytest 2>&1 | tail -2 || true
PYTHONPATH=$REMOTE_DIR ./venv_v4/bin/python -c 'import FinanceDataReader, numpy, pandas; print(\"  venv_v4 ok: FDR\", FinanceDataReader.__version__, \"numpy\", numpy.__version__)' || \
    { echo '  ⚠️ venv_v4 import 실패'; exit 1; }
# V3 venv 무손상 확인
./venv/bin/python -c 'import torch, numpy; print(\"  V3 venv 무손상: torch\", torch.__version__, \"numpy\", numpy.__version__)' 2>/dev/null || echo '  (V3 venv 점검 생략)'

echo '[2/3] systemd 유닛 설치...'
cat > /etc/systemd/system/$SVC.service << 'SVCEOF'
[Unit]
Description=V4 KOSDAQ Momentum Trader (daily paper session)
After=network.target

[Service]
Type=oneshot
User=root
WorkingDirectory=/opt/quant
Environment=PYTHONPATH=/opt/quant
EnvironmentFile=/opt/quant/.env
ExecStart=/opt/quant/venv_v4/bin/python -u v4/scripts/run_v4_live.py --mode once
StandardOutput=append:/var/log/quant-v4.log
StandardError=append:/var/log/quant-v4-error.log
SVCEOF

# 평일 09:05 KST — 직전 완성 종가로 신호 → 개장 직후 체결(장중, sandbox 수락).
# (장외 주문 거부 제약 때문에 종가후가 아닌 익일 개장 직후 실행)
cat > /etc/systemd/system/$SVC.timer << 'TMREOF'
[Unit]
Description=V4 KOSDAQ daily session (Mon-Fri 09:05 KST)

[Timer]
OnCalendar=Mon-Fri *-*-* 09:05:00
Persistent=true

[Install]
WantedBy=timers.target
TMREOF

systemctl daemon-reload
echo '  유닛 설치 + daemon-reload 완료 (enable 은 수동)'

echo '[3/3] 상태...'
echo '  service:' \$(systemctl is-enabled $SVC.service 2>/dev/null || echo disabled)
echo '  timer:  ' \$(systemctl is-enabled $SVC.timer 2>/dev/null || echo disabled)
"

echo ""
echo "=== 배포 완료 (timer 미활성) ==="
echo ""
echo "⚠️  가동 전 필수 (KR 장중, 다음 거래일):"
echo "  ssh $USER@$SERVER 'cd $REMOTE_DIR && PYTHONPATH=. venv_v4/bin/python v4/scripts/reset_sandbox_account.py --execute'"
echo "  ssh $USER@$SERVER 'cd $REMOTE_DIR && PYTHONPATH=. venv_v4/bin/python v4/scripts/run_v4_live.py --mode once --no-execute'   # plan smoke"
echo "  ssh $USER@$SERVER 'cd $REMOTE_DIR && PYTHONPATH=. venv_v4/bin/python v4/scripts/run_v4_live.py --mode once'               # 실주문 smoke"
echo ""
echo "✅ smoke OK 후 timer 활성화:"
echo "  ssh $USER@$SERVER 'systemctl enable --now $SVC.timer && systemctl list-timers | grep v4'"
echo ""
echo "관찰:"
echo "  ssh $USER@$SERVER 'tail -f /var/log/quant-v4.log'"
