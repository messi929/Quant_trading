#!/bin/bash
# V4 KOSDAQ Momentum Trader — systemd 배포 (daily paper session)
# Usage: bash deploy_v4.sh [server_ip]
#
# 전제: /opt/quant 에 최신 코드 존재 (deploy_v3_git.sh 또는 git 으로 v4/ 포함 배포 후).
# 동작:
#   1. venv 에 finance-datareader (V4 데이터 의존) 설치
#   2. systemd 유닛 설치 — quant-v4-korea.service(oneshot) + .timer(평일 09:05 KST)
#   3. daemon-reload (※ enable/start 는 수동 — 계좌리셋+smoke 후)
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

echo '[1/3] V4 데이터 의존 설치 (finance-datareader)...'
if [ -d venv ]; then
    ./venv/bin/pip install -q finance-datareader 2>&1 | tail -2 || true
    echo '  finance-datareader 설치/확인 완료'
    # numpy 회귀 sanity (로컬에서 v3 598 + v4 38 통과 검증됨)
    PYTHONPATH=$REMOTE_DIR ./venv/bin/python -c 'import FinanceDataReader, numpy; print(\"  FDR ok, numpy\", numpy.__version__)' || \
        echo '  ⚠️ import 실패 — venv 점검 필요'
else
    echo '  ⚠️ venv 없음 — deploy_v3_git.sh 먼저 실행 필요'; exit 1
fi

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
ExecStart=/opt/quant/venv/bin/python -u v4/scripts/run_v4_live.py --mode once
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
echo "  ssh $USER@$SERVER 'cd $REMOTE_DIR && PYTHONPATH=. venv/bin/python v4/scripts/reset_sandbox_account.py --execute'"
echo "  ssh $USER@$SERVER 'cd $REMOTE_DIR && PYTHONPATH=. venv/bin/python v4/scripts/run_v4_live.py --mode once --no-execute'   # plan smoke"
echo "  ssh $USER@$SERVER 'cd $REMOTE_DIR && PYTHONPATH=. venv/bin/python v4/scripts/run_v4_live.py --mode once'               # 실주문 smoke"
echo ""
echo "✅ smoke OK 후 timer 활성화:"
echo "  ssh $USER@$SERVER 'systemctl enable --now $SVC.timer && systemctl list-timers | grep v4'"
echo ""
echo "관찰:"
echo "  ssh $USER@$SERVER 'tail -f /var/log/quant-v4.log'"
