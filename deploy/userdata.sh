#!/bin/bash
#
# EC2 user data for a single-node test deployment of this service.
#
# Pasted into the "User data" field of a launch template, unchecked for "already
# base64 encoded". Runs once, as root, on first boot.
#
# What it deliberately does NOT do is start the stack. .env and the Firestore
# service account key are gitignored and are not carried here, because user data
# is readable from inside the instance through the metadata endpoint for the
# life of the instance -- and this service has no authentication, so anything
# that reaches code execution on the box would also read the key out of it.
# They are uploaded separately and /opt/rag/start.sh finishes the job.
#
# Sizing: this does not fit a t3.micro. The dependency set in pyproject.toml
# (deepagents, which pulls langgraph and langchain-core, plus four provider
# SDKs, pdfplumber, tiktoken, pinecone and firebase-admin) does not resolve in
# 1 GiB, and at run time the api and worker containers each load that whole
# stack alongside a Redis holding up to 512 MiB. t3.small is the floor and
# t3.medium is comfortable. The swap below makes a small instance survivable,
# not correct.
set -euxo pipefail
exec > >(tee /var/log/ragBootstrap.log | logger -t ragBootstrap -s 2>/dev/console) 2>&1

# Swap, so that a dependency resolution which briefly exceeds physical memory
# gets slow rather than getting OOM-killed. A killed `uv sync` fails the docker
# build with an exit code that says nothing about memory, which is a bad hour to
# spend on a test instance.
if [ ! -f /swapfile ]; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# Docker from Ubuntu's own archive rather than download.docker.com. The AMI is
# 26.04 ("resolute"), new enough that Docker's apt repository may not publish a
# suite for it yet -- and a missing suite is not a soft failure, it makes the
# whole `apt-get update` non-zero and takes the rest of this script with it
# under `set -e`. docker.io and docker-compose-v2 are in universe and are
# current enough for `docker compose`.
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  docker.io docker-compose-v2 git ca-certificates

systemctl enable --now docker
usermod -aG docker ubuntu

# The branch is named explicitly. main is several commits behind and still
# carries the Celery job manager and the in-process stores, which no longer
# match this compose file -- a default-branch clone would build an image whose
# app/ does not have app/jobs/worker.py in it at all.
install -d -o ubuntu -g ubuntu /opt/rag
sudo -u ubuntu git clone \
  --branch project-ids-remove-auth \
  https://github.com/Hitesh-Kumar-2001/LiftOff-Sol-Rag.git /opt/rag/app

# keys/ is bind-mounted read-only into both containers by docker-compose.yml,
# and is gitignored, so it does not arrive with the clone. Create it now, owned
# by ubuntu: if compose runs first Docker creates the missing source path itself
# as a root-owned directory, and the later scp then fails on permissions for a
# reason that looks nothing like the cause.
install -d -o ubuntu -g ubuntu /opt/rag/app/keys

# The finishing step, run by hand once the two secrets are uploaded. It checks
# for both before calling compose, because the failure mode without them is
# silent-ish and slow: the api container builds for several minutes and only
# then exits on checkConfiguration, and the worker sits idle looking healthy.
cat > /opt/rag/start.sh <<'EOF'
#!/bin/bash
set -euo pipefail
cd /opt/rag/app
test -f .env || { echo "missing /opt/rag/app/.env"; exit 1; }
compgen -G "keys/*.json" >/dev/null || { echo "missing service account key in /opt/rag/app/keys/"; exit 1; }
docker compose up -d --build
docker compose ps
EOF
chmod +x /opt/rag/start.sh
chown ubuntu:ubuntu /opt/rag/start.sh

echo "bootstrap done -- upload .env and keys/, then run /opt/rag/start.sh"
