# 人体动作生成对比台（MoMask / InterGen / in2IN）

基于 Flask + three.js 的本地测试界面，支持四种子模式，可在同一 3D 视口中对比：

| 模式 | 模型 | 说明 | 耗时（2080 Ti） |
|------|------|------|------|
| MoMask 单人 | [momask-codes](https://github.com/EricGuo5513/momask-codes)（CVPR 2024） | 单人文本生成动作 | ~3s |
| MoMask 拼合 | momask ×2 | 两个单人动作独立生成后并排摆放（**无真实交互**，作对比基线） | ~5s |
| InterGen | [InterGen](https://github.com/tr3e/InterGen)（IJCV 2024） | 双人交互扩散模型，一条描述生成两人互动动作 | ~20-40s |
| in2IN | [in2IN](https://github.com/pabloruizponce/in2IN)（CVPRW 2024） | 双人扩散模型，支持交互描述 + 每人独立描述 | ~15s |

**中文输入支持**：输入包含中文的提示词时自动本地翻译成英文再送给模型（默认开启，面板可关）。
翻译服务为独立 sidecar 进程（`webapp/translator_server.py`，端口 7863，由主服务自动拉起），
默认使用 `facebook/nllb-200-distilled-600M`（本地权重 `/home/applo/project/models/nllb-200-distilled-600M`），
备选 `Helsinki-NLP/opus-mt-zh-en`（`--model opus`），并内置运动领域术语表（跑步机→treadmill、
开合跳→jumping jack、击剑→fencing 等）预替换以纠正小模型的领域误译。

## 启动

```bash
cd <momask-codes 仓库根目录>
conda activate momask
python webapp/app.py --port 7862
```

浏览器打开 http://127.0.0.1:7862（局域网用本机 IP）。

## 环境与权重

主服务运行在 `momask` 环境（Python 3.7 + PyTorch 1.7.1）；
InterGen / in2IN 通过子进程运行在 `interact` 环境（Python 3.10 + torch 2.x）：

```bash
conda create -n interact python=3.10
pip install numpy==1.26.4 scipy yacs clip-anytorch==2.5.2 matplotlib pillow tqdm
# torch 走机器上已有的 2.x CUDA 版本即可
```

| 项 | 位置 |
|----|------|
| MoMask 权重 | `momask-codes/checkpoints/t2m/` |
| InterGen 权重 | `/home/applo/project/InterGen/checkpoints/intergen.ckpt`（来自 HF `ZeyuLing/hftrainer-intergen-interhuman`） |
| in2IN 权重 | `/home/applo/project/in2IN/checkpoints/pytorch_model.bin`（来自 HF `pabloruizponce/in2IN`，键名有 `model.` 前缀已在 worker 中处理） |
| CLIP 缓存 | `~/.cache/clip/`（ViT-B/32、ViT-L/14@336px） |
| InterGen/in2IN 推理脚本 | 各仓库根目录下 `worker_*.py`，输出 JSON（两人 × 210 帧 × 22 关节世界坐标） |

## MCP Server（供其他 Agent 调用）

`webapp/mcp_server.py` 提供 MCP（Model Context Protocol）服务，任何支持 MCP 的 Agent
（Claude Desktop / Cursor / ZCode 等）都可以直接调用动作生成与测试工具。

**启动**（streamable-http 模式，局域网可用，需要先启动主服务）：

```bash
/home/applo/anaconda3/envs/interact/bin/python webapp/mcp_server.py --port 7864
```

**接入配置**（二选一）：

局域网 Agent 直接连 URL（推荐，Claude Code / Cursor / ZCode 通用）：
```json
{ "mcpServers": { "momask-motion": { "type": "streamable-http", "url": "http://10.168.1.112:7864/mcp" } } }
```

本机 Claude Desktop 用 stdio 模式：
```json
{ "mcpServers": { "momask-motion": {
    "command": "/home/applo/anaconda3/envs/interact/bin/python",
    "args": ["/home/applo/project/momask-codes/webapp/mcp_server.py", "--stdio"] } } }
```

**工具列表**：

| 工具 | 功能 |
|------|------|
| `list_models` | 列出 4 种模型与适用场景 |
| `generate_motion` | MoMask 单人生成（中文自动翻译，时长/种子/IK 可选） |
| `generate_interaction` | InterGen / in2IN 双人生成（in2IN 支持个体描述） |
| `generate_momask_dual` | 双人拼合基线 |
| `analyze_motion` | 量化分析：步数、根速度、关节能量、左右腕高度、动作顺序、两人距离 |
| `render_video` | 渲染骨骼动画 mp4，返回本地路径 + 下载 URL |

生成工具返回 result_id 与文件路径，不含关节数组（避免撑爆 Agent 上下文）；
后续用 `analyze_motion` 取指标、`render_video` 出视频。

## API

- `POST /api/generate` — MoMask 单人：`{text, length秒, use_ik, seed, auto_translate}` → `{joints, translations, ...}`
- `POST /api/generate_momask_dual` — `{text_a, text_b, length, offset_x, use_ik, seed, auto_translate}` → `{persons}`
- `POST /api/generate_interaction` — `{model: 'intergen'|'in2in', interaction, ind1, ind2, seed, auto_translate}` → `{persons}`
- `GET /results/<id>/<file>` — 下载生成的 npy / bvh
- 翻译：`auto_translate: true`（默认）时，含中文的字段会先本地译成英文，响应中 `translations` 字段记录 `{字段: {src, en}}`

> 注意：InterDiff（ICCV 2023）调研后确认是**人-物体**交互生成（操纵箱子等），并非双人交互，故未纳入对比。
