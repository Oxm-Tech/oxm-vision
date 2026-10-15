#!/bin/bash
# Wrapper compatible con el flujo viejo. El stack real lo maneja docker-compose.yml.
# Uso:  ./run-container.sh           -> levanta el stack (idempotente)
#       ./run-container.sh logs      -> sigue logs en vivo
#       ./run-container.sh restart   -> down + up limpio (reinicia nvargus-daemon)
#       ./run-container.sh down      -> baja el stack

set -e
cd "$(dirname "$0")"

case "${1:-up}" in
  up)
    if sudo docker ps --format '{{.Names}}' | grep -q '^oxm-vision-oxm-vision-1$'; then
      echo "[run-container] ya está corriendo. Usa 'logs' o 'restart'."
      sudo docker ps --filter name=oxm-vision --format 'table {{.Names}}\t{{.Status}}'
      exit 0
    fi
    sudo systemctl restart nvargus-daemon && sleep 3
    sudo docker compose up -d
    sleep 4
    sudo docker compose ps
    ;;
  restart)
    sudo docker compose down
    sudo systemctl restart nvargus-daemon && sleep 3
    sudo docker compose up -d
    sleep 4
    sudo docker compose ps
    ;;
  down)
    sudo docker compose down
    ;;
  logs)
    sudo docker compose logs -f --tail=100
    ;;
  *)
    echo "Uso: $0 [up|restart|down|logs]"
    exit 1
    ;;
esac
