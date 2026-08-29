#!/usr/bin/env bash
#
# One-shot host setup for a single-node deployment, run ON the Ubuntu instance:
#
#   scp -i Test.pem deploy/setup.sh ubuntu@<IP>:~
#   ssh -i Test.pem ubuntu@<IP> 'sudo bash ~/setup.sh'
#
# Safe to run more than once -- every step checks for its own result first, so a
# re-run after a failure resumes rather than duplicating (a second `fallocate`
# on a live swapfile, in particular, is not survivable).
#
# WHAT THIS DOES NOT INSTALL, AND WHY
#
# Redis and unrar are NOT installed on this host. Both live inside containers,
# and putting a second copy on the host would be worse than useless:
#
#   redis   docker-compose.yml runs redis:7-alpine with --appendonly and a
#           512mb ceiling, on the compose network only, deliberately NOT
#           published to the host. An apt redis-server would bind 0.0.0.0 with
#           no password and protected-mode off -- on a public instance that is
#           an open Redis anyone can write to, and `CONFIG SET dir` from there
#           writes files as the redis user. It would also sit on 6379 and make
#           the container's port bind fail.
#
#   unrar   the Dockerfile installs Debian's non-free `unrar` into the image
#           and then verifies it against a real RAR5 archive at build time
#           (docker/checkRarBackend.py). That check exists because the free
#           alternatives fail by SILENT TRUNCATION rather than by erroring --
#           unar returns zero bytes for some members, bsdtar 51 bytes for a
#           39KB one -- and app/ingestion/documents.py skips a member it cannot
#           read, so either would build a RAG database quietly missing files.
#           The binary has to be where the ingesting process is, which is the
#           worker container, not here.
#
# The host needs exactly three things: docker, the compose plugin, and git.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Hitesh-Kumar-2001/LiftOff-Sol-Rag.git}"
REPO_BRANCH="${REPO_BRANCH:-project-ids-remove-auth}"
APP_DIR="${APP_DIR:-/opt/rag/app}"

# Who owns the checkout and gets added to the docker group. Under `sudo bash
# setup.sh` the login user is in SUDO_USER; run as root directly, fall back to
# the AMI's default account rather than leaving root-owned files that the user
# then cannot scp into.
RUN_USER="${SUDO_USER:-ubuntu}"
id "$RUN_USER" >/dev/null 2>&1 || RUN_USER=root

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root: sudo bash $0" >&2
  exit 1
fi

echo "==> installing into $APP_DIR for user $RUN_USER"

# ---------------------------------------------------------------- swap -------
# The heaviest thing that will ever happen on this box is the `uv sync` inside
# the image build: deepagents pulls langgraph and langchain-core, and four
# provider SDKs sit beside pdfplumber, tiktoken, pinecone and firebase-admin.
# On 1-2 GiB that resolution is OOM-killed, and the build then fails with an
# exit code that says nothing whatsoever about memory -- an easy hour to lose.
# Swap makes it slow instead of fatal. It does not make a t3.micro correct:
# at run time api and worker each hold that whole stack resident.
TOTAL_MB=$(free -m | awk '/^Mem:/ {print $2}')
if [ "$TOTAL_MB" -lt 4000 ] && [ ! -f /swapfile ]; then
  echo "==> ${TOTAL_MB}MB RAM: adding 2G swap"
  fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# -------------------------------------------------------------- packages -----
# From Ubuntu's own archive, not download.docker.com. This AMI is 26.04
# ("resolute") and is new enough that Docker's apt repository may not publish a
# suite for it yet -- and a missing suite is not a soft failure, it makes the
# whole `apt-get update` exit non-zero and takes the rest of this script with it
# under `set -e`.
export DEBIAN_FRONTEND=noninteractive
echo "==> installing docker, compose, git"
apt-get update -y
apt-get install -y --no-install-recommends \
  docker.io docker-compose-v2 git ca-certificates curl

systemctl enable --now docker

# So the login user can run docker without sudo. This does NOT affect the shell
# already open -- group membership is read at login, so they have to reconnect.
usermod -aG docker "$RUN_USER" || true

# docker-compose-v2 is in universe, which is enabled on the stock server AMI but
# not on every image. If the plugin is missing, fetch it straight from the
# release page rather than failing here: everything below is `docker compose`.
if ! docker compose version >/dev/null 2>&1; then
  echo "==> compose plugin missing, installing from GitHub"
  case "$(uname -m)" in
    x86_64)  COMPOSE_ARCH=x86_64  ;;
    aarch64) COMPOSE_ARCH=aarch64 ;;
    *) echo "unsupported arch $(uname -m)" >&2; exit 1 ;;
  esac
  install -d /usr/local/lib/docker/cli-plugins
  curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-compose \
    "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-${COMPOSE_ARCH}"
  chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
  docker compose version
fi

# ------------------------------------------------------------------ code -----
# The branch is named explicitly. main is well behind and still carries the
# Celery job manager and the in-process stores; a default-branch clone would
# build an image whose app/ has no app/jobs/worker.py in it at all, and compose
# would fail on a command that does not exist.
install -d -o "$RUN_USER" -g "$RUN_USER" "$(dirname "$APP_DIR")"
if [ -d "$APP_DIR/.git" ]; then
  echo "==> updating existing checkout"
  sudo -u "$RUN_USER" git -C "$APP_DIR" fetch origin "$REPO_BRANCH"
  sudo -u "$RUN_USER" git -C "$APP_DIR" checkout "$REPO_BRANCH"
  sudo -u "$RUN_USER" git -C "$APP_DIR" pull --ff-only origin "$REPO_BRANCH"
else
  echo "==> cloning $REPO_BRANCH"
  sudo -u "$RUN_USER" git clone --branch "$REPO_BRANCH" "$REPO_URL" "$APP_DIR"
fi

# keys/ is bind-mounted read-only into api and worker, and is gitignored, so it
# does not arrive with the clone. Create it owned by the login user NOW: if
# compose runs first, Docker creates the missing source path itself as a
# root-owned directory, and the later scp then fails on permissions for a
# reason that looks nothing like the cause.
install -d -o "$RUN_USER" -g "$RUN_USER" "$APP_DIR/keys"

# ----------------------------------------------------------------- start -----
# Checked before calling compose because the failure without them is slow and
# quiet: the image builds for several minutes and only then does the api exit
# on checkConfiguration, while the worker sits there looking healthy.
#
# Nothing here edits .env. The Windows RAG_UNRAR_TOOL path in a dev .env is
# already handled -- docker-compose.yml blanks that variable for both services,
# and app/ingestion/documents.py ignores an override that is not present, so
# the container falls through to the unrar installed in the image.
cd "$APP_DIR"
if [ -f .env ] && compgen -G "keys/*.json" >/dev/null; then
  echo "==> .env and key present, building (first build takes 5-15 min)"
  docker compose up -d --build
  docker compose ps
  echo
  echo "health:  curl http://localhost:8000/health"
  echo "logs:    docker compose -f $APP_DIR/docker-compose.yml logs -f"
else
  cat <<EOF

==> host ready. Two files are still missing; upload them from your machine:

    scp -i Test.pem .env      ubuntu@<PUBLIC_IP>:$APP_DIR/.env
    scp -i Test.pem -r keys   ubuntu@<PUBLIC_IP>:$APP_DIR/

  then start it:

    cd $APP_DIR && docker compose up -d --build

  (log out and back in first, so your docker group membership applies and you
  can drop the sudo.)
EOF
fi
