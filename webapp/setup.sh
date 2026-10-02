#!/bin/bash
# 首次部署准备脚本：检查环境、准备模型权重
#
# 用法:
#   bash webapp/setup.sh                       # 完整检查，缺 MoMask 权重时从 HF 镜像下载
#   bash webapp/setup.sh --link-checkpoints DIR # 从已有目录软链 MoMask 权重（同机复用，秒完成）
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MIRROR="https://hf-mirror.com"
ok=0; warn=0
good() { echo "✓ $1"; ok=$((ok+1)); }
bad()  { echo "✗ $1"; warn=$((warn+1)); }
info() { echo "… $1"; }

echo "=== 1. Python 环境 ==="
for env in momask interact kimodo; do
  py="/home/applo/anaconda3/envs/$env/bin/python"
  [ -x "$py" ] && good "conda env $env" || bad "conda env $env 缺失（人工创建：见 webapp/README.md 环境与权重一节）"
done

echo "=== 2. 硬件与系统 ==="
nvidia-smi >/dev/null 2>&1 && good "GPU 可用: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)" || bad "GPU 不可用（minimal 模式可跑 MoMask，但很慢）"
which ffmpeg >/dev/null && good "ffmpeg" || bad "ffmpeg 缺失（视频渲染需要）"

echo "=== 3. MoMask 权重 (checkpoints/t2m) ==="
LINK_FROM=""
if [ "${1:-}" = "--link-checkpoints" ] && [ -n "${2:-}" ]; then LINK_FROM="$2"; fi

if [ -f "checkpoints/t2m/t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns/model/latest.tar" ]; then
  good "MoMask 权重已存在"
elif [ -n "$LINK_FROM" ] && [ -d "$LINK_FROM/t2m" ]; then
  mkdir -p checkpoints
  ln -sfn "$(cd "$LINK_FROM" && pwd)/t2m" checkpoints/t2m
  good "已软链 MoMask 权重 -> $LINK_FROM/t2m"
else
  info "下载 MoMask 权重（~188MB，来自 hf-mirror camenduru/MoMask）..."
  mkdir -p checkpoints/t2m && cd checkpoints/t2m
  if curl -sL --retry 5 --retry-all-errors -C - --max-time 3600 -o humanml3d_models.zip \
      "$MIRROR/camenduru/MoMask/resolve/main/humanml3d_models.zip" && file -b humanml3d_models.zip | grep -q gzip; then
    unzip -q humanml3d_models.zip && rm humanml3d_models.zip
    good "MoMask 权重下载完成"
  else
    bad "MoMask 权重下载失败（手动放置见 README）"
  fi
  cd "$ROOT"
fi

echo "=== 4. 可选组件（缺失时对应模式/模型不可用，不影响 MoMask） ==="
[ -f /home/applo/project/InterGen/worker_intergen.py ] && good "InterGen worker" || bad "InterGen 未部署（/home/applo/project/InterGen）"
[ -f /home/applo/project/in2IN/worker_in2in.py ] && good "in2IN worker" || bad "in2IN 未部署（/home/applo/project/in2IN）"
[ -d /home/applo/project/models/nllb-200-distilled-600M ] && good "翻译模型 NLLB" || bad "翻译模型缺失（中文翻译不可用，英文不受影响）"
[ -d /home/applo/project/models/llm2vec-mntp ] && good "Kimodo 文本编码器适配器" || bad "Kimodo 组件缺失（lite 模式可忽略）"
[ -d /home/applo/project/kimodo ] && good "Kimodo 仓库" || bad "Kimodo 仓库缺失（lite 模式可忽略）"

echo
echo "=== 检查完成：$ok 项就绪, $warn 项缺失 ==="
echo "启动: bash webapp/start.sh full|lite|minimal|auto"
[ $warn -gt 0 ] && exit 1 || exit 0
