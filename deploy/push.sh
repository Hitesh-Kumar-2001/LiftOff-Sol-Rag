#!/usr/bin/env bash
#
# Push this checkout's secrets to an instance and bring the stack up. Run from
# WSL, from the repo root:
#
#   ./deploy/push.sh 12.34.56.78
#
# The code itself does not travel this way -- deploy/setup.sh clones it from
# GitHub on the far side. What moves here is only the two gitignored files that
# a clone cannot carry: .env and the Firestore service account key.
set -euo pipefail

HOST="${1:-}"
KEY="${KEY:-$HOME/Test2.pem}"
USER_AT="${SSH_USER:-ubuntu}"
APP_DIR="${APP_DIR:-/opt/rag/app}"

if [ -z "$HOST" ]; then
  echo "usage: $0 <public-ip>   (KEY=/path/to.pem to override the key)" >&2
  exit 1
fi

# A key on /mnt/... is a DrvFs path, which reports 0777 no matter what chmod is
# asked for, and ssh refuses a world-readable private key outright. Copying it
# into the WSL filesystem is the only fix that sticks, so do it here rather than
# failing with an error that reads like the key itself is wrong.
case "$KEY" in
  /mnt/*)
    echo "==> $KEY is on a Windows mount; copying to ~/$(basename "$KEY")"
    cp "$KEY" "$HOME/$(basename "$KEY")"
    KEY="$HOME/$(basename "$KEY")"
    ;;
esac
chmod 600 "$KEY" 2>/dev/null || true

SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$USER_AT@$HOST")
SCP=(scp -i "$KEY" -o StrictHostKeyChecking=accept-new)

# Checked here rather than on the instance. Without them the far side builds for
# several minutes and only then does the api exit on checkConfiguration, while
# the worker sits looking healthy -- a slow, quiet failure worth pre-empting.
test -f .env || { echo "no .env in $(pwd)" >&2; exit 1; }
compgen -G "keys/*.json" >/dev/null || { echo "no service account key in $(pwd)/keys/" >&2; exit 1; }

echo "==> bootstrapping host"
"${SCP[@]}" deploy/setup.sh "$USER_AT@$HOST:~/setup.sh"
"${SSH[@]}" 'sudo bash ~/setup.sh'

echo "==> uploading secrets"
"${SCP[@]}" .env "$USER_AT@$HOST:$APP_DIR/.env"
"${SCP[@]}" -r keys "$USER_AT@$HOST:$APP_DIR/"

# sudo because the docker group membership setup.sh grants is only read at
# login, so the very first deploy runs in a shell that does not have it yet.
echo "==> building and starting (first build takes 5-15 min)"
"${SSH[@]}" "cd $APP_DIR && sudo docker compose up -d --build && sudo docker compose ps"

echo
echo "health: curl http://$HOST:8000/health"
echo "logs:   ssh -i $KEY $USER_AT@$HOST 'cd $APP_DIR && sudo docker compose logs -f'"
