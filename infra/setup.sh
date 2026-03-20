#!/usr/bin/env bash
# GDF Hub — standalone bootstrap script
# Run on a fresh Amazon Linux 2023 EC2 instance:
#   curl -sL https://raw.githubusercontent.com/gdf-ai/gdf/main/infra/setup.sh | bash -s -- <TOKEN>
#
# Or manually: bash setup.sh <TOKEN>

set -euo pipefail

TOKEN="${1:?Usage: setup.sh <GDF_TOKEN>}"

echo "==> Installing system packages..."
sudo dnf update -y
sudo dnf install -y python3.11 python3.11-pip

echo "==> Installing GDF..."
python3.11 -m pip install gdf

echo "==> Creating gdf user..."
sudo useradd -r -s /sbin/nologin gdf 2>/dev/null || true
sudo mkdir -p /opt/gdf
sudo chown gdf:gdf /opt/gdf

echo "==> Initializing model..."
cd /opt/gdf
sudo -u gdf python3.11 -m gdf init --name hub_model.pt

echo "==> Downloading seeds..."
sudo -u gdf curl -sL "https://raw.githubusercontent.com/gdf-ai/gdf/main/seeds.txt" \
    -o /opt/gdf/seeds.txt

echo "==> Writing environment file..."
sudo tee /opt/gdf/.env > /dev/null <<EOF
GDF_TOKEN=${TOKEN}
GDF_HUB_URL=https://hub.gdf.ai
GDF_HUB_TOKEN=${TOKEN}
EOF
sudo chmod 600 /opt/gdf/.env
sudo chown gdf:gdf /opt/gdf/.env

echo "==> Installing Caddy..."
sudo dnf install -y dnf-plugins-core
sudo dnf copr enable -y @caddy/caddy epel-9-x86_64 || true
sudo dnf install -y caddy || {
    echo "    Falling back to binary install..."
    curl -sL "https://github.com/caddyserver/caddy/releases/latest/download/caddy_2.9.1_linux_amd64.tar.gz" \
        | sudo tar xz -C /usr/local/bin caddy
    sudo chmod +x /usr/local/bin/caddy
}

echo "==> Configuring Caddy..."
sudo mkdir -p /etc/caddy /var/log/caddy
sudo tee /etc/caddy/Caddyfile > /dev/null <<'EOF'
hub.gdf.ai {
    reverse_proxy localhost:7677
    log {
        output file /var/log/caddy/access.log
    }
}
EOF

echo "==> Installing systemd service..."
sudo cp "$(dirname "$0")/gdf-hub.service" /etc/systemd/system/gdf-hub.service 2>/dev/null || \
sudo tee /etc/systemd/system/gdf-hub.service > /dev/null <<'EOF'
[Unit]
Description=GDF Hub Server
After=network.target

[Service]
Type=simple
User=gdf
WorkingDirectory=/opt/gdf
EnvironmentFile=/opt/gdf/.env
ExecStart=/usr/local/bin/gdf hub --model /opt/gdf/hub_model.pt --token ${GDF_TOKEN} --seeds /opt/gdf/seeds.txt
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "==> Starting services..."
sudo systemctl daemon-reload
sudo systemctl enable --now caddy
sudo systemctl enable --now gdf-hub

echo ""
echo "==> GDF Hub is running!"
echo "    URL:   https://hub.gdf.ai"
echo "    Token: ${TOKEN}"
echo "    Check: curl https://hub.gdf.ai/status"
echo "    Logs:  journalctl -u gdf-hub -f"
