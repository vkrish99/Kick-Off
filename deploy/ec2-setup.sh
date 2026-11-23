#!/usr/bin/env bash
# Sets up Kick-Off (PitchMaster AI) on a fresh Amazon Linux 2023 EC2 instance and starts it on port 80.
# Safe to re-run: it pulls the latest code, rebuilds the image and restarts the stack.
#
#   sudo dnf install -y git && git clone https://github.com/vkrish99/Kick-Off.git && bash Kick-Off/deploy/ec2-setup.sh
set -euo pipefail

COMPOSE_VERSION="v2.29.7"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

echo "==> 1/5 Swap (a 1 GB server needs it to build the image)"
if ! sudo swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "==> 2/5 Docker and the Compose plugin"
sudo dnf install -y -q docker git
sudo systemctl enable --now docker
if ! sudo docker compose version >/dev/null 2>&1; then
  sudo mkdir -p /usr/local/lib/docker/cli-plugins
  sudo curl -fsSL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-$(uname -m)" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
  sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

echo "==> 3/5 Latest code"
git pull --ff-only || echo "   (git pull skipped: local changes or detached HEAD)"

echo "==> 4/5 Secrets (.env)"
if [ ! -f .env ]; then
  echo "   Paste your keys. They are saved only in $APP_DIR/.env on this server and are not shown."
  read -rsp "   GROQ_API_KEY: " groq_key </dev/tty; echo
  read -rsp "   PINECONE_API_KEY: " pinecone_key </dev/tty; echo
  read -rp  "   PINECONE_INDEX_NAME [pitchmaster-index]: " index_name </dev/tty
  api_key="$(openssl rand -hex 24)"
  ( umask 077
    cat > .env <<EOF
GROQ_API_KEY=${groq_key}
PINECONE_API_KEY=${pinecone_key}
PINECONE_INDEX_NAME=${index_name:-pitchmaster-index}
API_KEY=${api_key}
CORS_ORIGINS=*
EOF
  )
  echo
  echo "   Generated API_KEY for the /ingest/* endpoints (save it somewhere safe):"
  echo "   ${api_key}"
  echo
else
  echo "   .env already exists; keeping it."
fi

echo "==> 5/5 Build and start"
sudo docker build -q -t kick-off:latest .
sudo docker compose -f docker-compose.prod.yml up -d --remove-orphans
sudo docker image prune -f >/dev/null

echo "   Waiting for the app to report healthy..."
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1/health >/dev/null 2>&1; then
    token="$(curl -fsS -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || true)"
    public_ip="$(curl -fsS -H "X-aws-ec2-metadata-token: ${token}" http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo '<this server public IP>')"
    echo
    echo "Kick-Off is running:  http://${public_ip}/"
    echo "API docs:             http://${public_ip}/docs"
    exit 0
  fi
  sleep 3
done
echo "The app didn't become healthy. Logs:"
sudo docker compose -f docker-compose.prod.yml logs --tail 50 app
exit 1
