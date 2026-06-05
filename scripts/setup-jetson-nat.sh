#!/bin/bash
# Comparte internet desde una Mac hacia la Jetson por Ethernet (NAT con pf).
#
# Uso:
#   sudo WIFI_IF=en0 ETH_IF=en5 JETSON_NET=10.10.10.0/24 ./setup-jetson-nat.sh
#
# Defaults para el setup más común (WiFi en en0, Jetson conectada por USB-Ethernet en5).
# Ajusta WIFI_IF, ETH_IF y JETSON_NET para tu red.
set -e

WIFI_IF="${WIFI_IF:-en0}"
ETH_IF="${ETH_IF:-en5}"
JETSON_NET="${JETSON_NET:-10.10.10.0/24}"

ANCHOR_FILE="/etc/pf.anchors/jetson-nat"
PF_CONF="/etc/pf.conf"
PLIST="/Library/LaunchDaemons/com.oxmtech.jetson-nat.plist"

echo "==> Creando anchor pf NAT ($JETSON_NET → $WIFI_IF)..."
tee "$ANCHOR_FILE" << EOF
nat on $WIFI_IF from $JETSON_NET to any -> ($WIFI_IF)
EOF

echo "==> Habilitando IP forwarding..."
sysctl -w net.inet.ip.forwarding=1

echo "==> Añadiendo anchor a pf.conf (si no está ya)..."
if ! grep -q "jetson-nat" "$PF_CONF"; then
  # Añade nat-anchor después de la línea de com.apple
  sed -i '' '/nat-anchor "com.apple\/\*"/a\
nat-anchor "jetson-nat"
' "$PF_CONF"
  # Añade anchor y load al final
  cat >> "$PF_CONF" << 'PFEOF'

# Jetson internet sharing
anchor "jetson-nat"
load anchor "jetson-nat" from "/etc/pf.anchors/jetson-nat"
PFEOF
fi

echo "==> Activando pfctl..."
pfctl -e -f "$PF_CONF" 2>/dev/null || true

echo "==> Verificando reglas NAT activas..."
pfctl -s nat

echo "==> Creando LaunchDaemon para persistencia en reinicios..."
tee "$PLIST" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.oxmtech.jetson-nat</string>
  <key>ProgramArguments</key>
  <array>
    <string>/sbin/pfctl</string>
    <string>-e</string>
    <string>-f</string>
    <string>/etc/pf.conf</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
</dict>
</plist>
EOF

launchctl load -w "$PLIST" 2>/dev/null || true

echo ""
echo "✓ Listo. La Jetson ($JETSON_NET) ya tiene internet via $WIFI_IF."
echo "  Prueba desde la Jetson: ping -c 3 8.8.8.8"
