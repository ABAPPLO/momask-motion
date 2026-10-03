#!/bin/bash
# 人体动作生成对比台 · 启动/停止脚本
#
# 用法: ./start.sh [模式|命令]
#   full     完整模式：5 模型全部启用，翻译在 GPU（默认；Kimodo 编码在 CPU，占内存 ~17GB）
#   lite     低CPU模式：跳过 Kimodo（省 17GB 内存与满核 CPU），翻译切 CPU（省 2.6GB 显存）
#   minimal  精简模式：仅 MoMask 单人/拼合 + 翻译(CPU)，跳过 InterGen/in2IN/Kimodo
#   auto     按当前资源自动选 full / lite / minimal
#   stop     停止全部服务
#   status   查看各服务状态
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEBAPP="$ROOT/webapp"
MOMASK_PY=/home/applo/anaconda3/envs/momask/bin/python
INTERACT_PY=/home/applo/anaconda3/envs/interact/bin/python
KIMODO_PY=/home/applo/anaconda3/envs/kimodo/bin/python

PORTS=(7862 7863 7864 7865)
NAMES=(主服务 翻译 MCP Kimodo)

pids_on_port() { ss -tlnp 2>/dev/null | grep ":$1 " | grep -oP 'pid=\K[0-9]+' | sort -u; }

stop_all() {
  echo "停止全部服务..."
  for p in "${PORTS[@]}"; do
    for pid in $(pids_on_port "$p"); do kill "$pid" 2>/dev/null && echo "  port $p (pid $pid) 已停止"; done
  done
  sleep 2
}

status_all() {
  local labels=(主服务Web 翻译sidecar MCPServer Kimodo-sidecar)
  for i in "${!PORTS[@]}"; do
    if ss -tlnp 2>/dev/null | grep -q ":${PORTS[$i]} "; then
      echo "✓ ${labels[$i]}  port ${PORTS[$i]}"
    else
      echo "✗ ${labels[$i]}  port ${PORTS[$i]} 未启动"
    fi
  done
  curl -s --max-time 3 http://127.0.0.1:7862/api/health 2>/dev/null | head -c 300; echo
}

detect_auto() {
  local cores ram_free vram_free vram_total
  cores=$(nproc)
  ram_free=$(free -m | awk '/^Mem:/{print $7}')
  vram_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  vram_free=$(( vram_total - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1) ))
  echo "资源检测: ${cores}核 / 可用内存 $((ram_free/1024))GB / 空闲显存 ${vram_free:-无GPU}MB"
  if [ -z "${vram_total:-}" ]; then
    echo "→ 未检测到 GPU，选择 minimal（MoMask 也可 CPU 运行但很慢）"; MODE=minimal
  elif [ "${vram_free:-0}" -lt 4000 ]; then
    echo "→ 空闲显存 <4GB，选择 minimal"; MODE=minimal
  elif [ "${ram_free:-0}" -lt 26000 ] || [ "${cores}" -lt 4 ]; then
    echo "→ 可用内存 <26GB 或核数 <4（Kimodo 编码器需要 ~17GB 内存+多核），选择 lite"; MODE=lite
  else
    echo "→ 资源充足，选择 full"; MODE=full
  fi
}

wait_health() {  # url, timeout_s
  local url=$1 t=${2:-60} elapsed=0
  while [ $elapsed -lt $t ]; do
    curl -s --max-time 2 "$url" 2>/dev/null | grep -q '"ok"\|"status".*"ok"' && return 0
    sleep 3; elapsed=$((elapsed+3))
  done
  return 1
}

CMD="${1:-auto}"
case "$CMD" in
  stop) stop_all; exit 0 ;;
  status) status_all; exit 0 ;;
  full|lite|minimal) MODE="$CMD" ;;
  auto) detect_auto ;;
  *) echo "用法: $0 [full|lite|minimal|auto|stop|status]"; exit 1 ;;
esac

echo "=== 以 [$MODE] 模式部署 ==="
stop_all

# 翻译设备：lite/minimal 切 CPU 省 2.6GB 显存
TRANSLATOR_DEVICE=auto
[ "$MODE" != "full" ] && TRANSLATOR_DEVICE=cpu

export MOMASK_MODE=$MODE
export TRANSLATOR_DEVICE

# 1. 主服务（MoMask + 所有 API）
(cd "$ROOT" && setsid nohup "$MOMASK_PY" webapp/app.py --port 7862 \
   </dev/null >/tmp/momask_web.log 2>&1 &)
echo "主服务启动中 (port 7862)..."

# 2. 翻译 sidecar（健康检查会由主服务自动拉起，这里显式启动以便立即可用）
(cd "$WEBAPP" && setsid nohup "$INTERACT_PY" translator_server.py \
   </dev/null >/tmp/translator.log 2>&1 &)
echo "翻译服务启动中 (port 7863, device=$TRANSLATOR_DEVICE)..."

# 3. MCP Server
(cd "$WEBAPP" && setsid nohup "$INTERACT_PY" mcp_server.py \
   </dev/null >/tmp/mcp_server.log 2>&1 &)
echo "MCP Server 启动中 (port 7864)..."

# 4. Kimodo sidecar（仅 full 模式；lite/minimal 下主服务也会拒绝 kimodo 请求）
if [ "$MODE" = "full" ]; then
  (cd "$WEBAPP" && setsid nohup "$KIMODO_PY" kimodo_server.py \
     </dev/null >/tmp/kimodo_server.log 2>&1 &)
  echo "Kimodo sidecar 启动中 (port 7865，加载约 2 分钟)..."
fi

# 等待主服务就绪
if wait_health http://127.0.0.1:7862/api/health 90; then
  echo "✓ 主服务就绪"
else
  echo "✗ 主服务未在 90s 内就绪，查看 /tmp/momask_web.log"; exit 1
fi

wait_health http://127.0.0.1:7863/health 120 && echo "✓ 翻译就绪" || echo "△ 翻译仍在加载（首次用时自动等待）"
wait_health http://127.0.0.1:7864/health 60 2>/dev/null || true
curl -s --max-time 3 -o /dev/null http://127.0.0.1:7864/mcp && echo "✓ MCP 就绪"

if [ "$MODE" = "full" ]; then
  if wait_health http://127.0.0.1:7865/health 300; then echo "✓ Kimodo 就绪"; else echo "△ Kimodo 仍在加载（最长约 5 分钟，期间该模型暂不可用）"; fi
fi

echo
echo "=== 部署完成 [$MODE] ==="
echo "Web:    http://127.0.0.1:7862"
echo "MCP:    http://127.0.0.1:7864/mcp"
case "$MODE" in
  full)     echo "可用模型: MoMask单人/拼合 + InterGen + in2IN + Kimodo (5个)" ;;
  lite)     echo "可用模型: MoMask单人/拼合 + InterGen + in2IN (4个，Kimodo 已跳过)" ;;
  minimal)  echo "可用模型: MoMask单人/拼合 (2个，InterGen/in2IN/Kimodo 已跳过)" ;;
esac
