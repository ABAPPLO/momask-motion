"""MCP server for the motion-generation test platform (MoMask / InterGen / in2IN).

Lets any MCP-capable agent (Claude Desktop, Cursor, ZCode, ...) generate and
analyze human-motion sequences through tools. All heavy lifting is delegated
to the Flask webapp API (default http://127.0.0.1:7862).

Transports:
  streamable-http (default, LAN-accessible):  endpoint http://<host>:7864/mcp
  stdio (single local client):                --stdio

Run:
  /home/applo/anaconda3/envs/interact/bin/python mcp_server.py [--stdio] [--port 7864]
"""
import argparse
import json
import os

import httpx
import numpy as np
from mcp.server.mcpserver import MCPServer

# Internal API base (MCP server and webapp always share one host)
WEBAPP = os.environ.get('MOMASK_WEBAPP_URL', 'http://127.0.0.1:7862')


def _lan_ip():
    """Best-effort LAN IP for building URLs handed to remote agents."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))  # no packets sent; just picks a route
            return s.getsockname()[0]
        finally:
            s.close()
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return '127.0.0.1'


# Public base used ONLY for URLs returned to agents (env-overridable)
PUBLIC_URL = os.environ.get('MOMASK_PUBLIC_URL', f'http://{_lan_ip()}:7862').rstrip('/')


def _abs_urls(files: dict) -> dict:
    """Turn relative /results/... paths into absolute URLs agents can fetch."""
    return {k: (PUBLIC_URL + v if v.startswith('/') else v) for k, v in (files or {}).items()}
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

mcp = MCPServer(
    name='momask-motion',
    title='人体动作生成测试平台',
    description='文本生成 3D 人体动作：单人(MoMask)、双人交互(InterGen/in2IN)，'
                '支持中文提示词自动翻译、动作量化分析与视频渲染。',
)


def _post(path, payload, timeout=360.0):
    r = httpx.post(WEBAPP + path, json=payload, timeout=timeout)
    data = r.json()
    if r.status_code != 200 or data.get('error'):
        raise RuntimeError(f'webapp error: {data.get("error") or r.status_code}')
    return data


def _result_dir(rid):
    d = os.path.join(RESULTS_DIR, rid)
    if not os.path.isdir(d):
        raise FileNotFoundError(f'unknown result id: {rid} (dir not found)')
    return d


def _load_persons(rid):
    """Load joint arrays for a result id; returns (persons:list[np.ndarray], fps)."""
    d = _result_dir(rid)
    persons = []
    if os.path.exists(os.path.join(d, 'person0.npy')):
        persons.append(np.load(os.path.join(d, 'person0.npy')))
        if os.path.exists(os.path.join(d, 'person1.npy')):
            persons.append(np.load(os.path.join(d, 'person1.npy')))
    elif os.path.exists(os.path.join(d, 'joints_ik.npy')):
        persons.append(np.load(os.path.join(d, 'joints_ik.npy')))
    elif os.path.exists(os.path.join(d, 'joints.npy')):
        persons.append(np.load(os.path.join(d, 'joints.npy')))
    if not persons:
        raise FileNotFoundError(f'no joint arrays under {d}')
    return persons


# ------------------------------------------------------------------ metrics
# index maps per skeleton
SMPL_IDX = dict(L_WRIST=20, R_WRIST=21, L_FOOT=10, R_FOOT=11, ROOT=0,
                ARM=[18, 19, 20, 21], LEG=[4, 5, 7, 8, 10, 11])
# SOMA77: LeftHand=14, RightHand=42, feet=69/74 (order from kimodo definitions.py)
SOMA_IDX = dict(L_WRIST=14, R_WRIST=42, L_FOOT=69, R_FOOT=74, ROOT=0,
                ARM=[12, 13, 14, 40, 41, 42], LEG=[67, 68, 69, 72, 73, 74])


def _idx_for(njoints):
    return SOMA_IDX if njoints == 77 else SMPL_IDX


def _describe(j, fps):
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    idx = _idx_for(j.shape[1])
    steps = 0
    for k in ('L_FOOT', 'R_FOOT'):
        y = j[:, idx[k], 1]
        peaks, _ = find_peaks(y - np.median(y), height=0.045,
                              distance=max(1, int(fps * 0.3)))
        steps += len(peaks)
    root_v = np.linalg.norm(np.diff(j[:, idx['ROOT'], :], axis=0), axis=1) * fps
    energy = np.linalg.norm(np.diff(j, axis=0), axis=2).mean(axis=1) * fps
    arm_env = gaussian_filter1d(np.linalg.norm(np.diff(j[:, idx['ARM']], axis=0), axis=2).mean(axis=1), max(1, fps // 3))
    leg_env = gaussian_filter1d(np.linalg.norm(np.diff(j[:, idx['LEG']], axis=0), axis=2).mean(axis=1), max(1, fps // 3))
    return {
        'frames': int(len(j)),
        'seconds': round(len(j) / fps, 2),
        'steps': int(steps),
        'root_speed_mps': round(float(root_v.mean()), 2),
        'joint_energy': round(float(energy.mean()), 2),
        'wrist_max_height_L': round(float(j[:, idx['L_WRIST'], 1].max()), 2),
        'wrist_max_height_R': round(float(j[:, idx['R_WRIST'], 1].max()), 2),
        'net_displacement_xyz': [round(float(x), 2) for x in j[-1, idx['ROOT']] - j[0, idx['ROOT']]],
        'arm_peak_time_s': round(float(arm_env.argmax()) / fps, 2),
        'leg_peak_time_s': round(float(leg_env.argmax()) / fps, 2),
    }


# ------------------------------------------------------------------ tools
@mcp.tool()
def list_models() -> dict:
    """列出可用的动作生成模型及各自能力，返回每个模型的适用场景与参数说明。"""
    return {
        'models': [
            {'id': 'momask', 'type': 'single-person',
             'desc': 'MoMask (CVPR 2024) 单人文本生成动作，20fps，时长可指定（0=自动，最长9.8s）',
             'best_for': '单人动作、身体部位/方向控制'},
            {'id': 'momask_dual', 'type': 'two-person-baseline',
             'desc': '两个 MoMask 单人动作独立生成后并排摆放——无真实交互，仅作对比基线',
             'best_for': '展示"单人模型做不了交互"的对照'},
            {'id': 'intergen', 'type': 'two-person',
             'desc': 'InterGen (IJCV 2024) 双人交互扩散模型，一条描述生成两人 210帧@30fps（约7s）',
             'best_for': '打斗/拥抱/共舞等真实双人互动'},
            {'id': 'in2in', 'type': 'two-person',
             'desc': 'in2IN (CVPRW 2024) 双人扩散模型，交互描述 + 每人独立描述，210帧@30fps',
             'best_for': '需要分别控制两人动作风格'},
            {'id': 'kimodo', 'type': 'single-person',
             'desc': 'NVIDIA Kimodo 原版 (nv-tlabs/kimodo, SOMA-RP v1.1)，700h 生产级动捕训练，'
                     'SOMA 77 关节（含手指），30fps，时长 1~10s；文本编码在 CPU 运行（首次加载约 2~5 分钟）',
             'best_for': '高质量单人动作、与学术模型对比工业级数据的效果'},
        ],
        'notes': '所有模型支持中文提示词（自动本地翻译成英文）；生成结果用 analyze_motion/render_video 做量化分析与视频渲染。',
    }


@mcp.tool()
def generate_motion(text: str, length_seconds: float = 0.0, seed: int = 10107,
                    use_ik: bool = True, auto_translate: bool = True,
                    render_video: bool = False) -> dict:
    """用 MoMask 生成单人动作。text 支持中文（自动翻译）；length_seconds 0=自动估计。

    render_video=True 时同时渲染骨骼动画 mp4（额外约 30~60 秒），
    返回的 files 里包含 video 路径与 URL。其余情况返回 result_id、帧数、
    翻译结果与文件下载路径（joints npy / bvh）。
    """
    d = _post('/api/generate', {'text': text, 'length': length_seconds, 'seed': seed,
                                'use_ik': use_ik, 'auto_translate': auto_translate,
                                'render_video': render_video})
    return {'id': d['id'], 'model': d['model'], 'm_length': d['m_length'],
            'fps': d['fps'], 'seconds': d['seconds'], 'gen_time': d['gen_time'],
            'translations': d['translations'],
            'files': d['files'], 'urls': _abs_urls(d['files'])}


@mcp.tool()
def generate_interaction(model: str, interaction: str, individual_1: str = '',
                         individual_2: str = '', seed: int = 10107,
                         auto_translate: bool = True,
                         render_video: bool = False) -> dict:
    """用双人交互模型生成两人动作。model: 'intergen' 或 'in2in'。

    interaction 是整体场景描述（如"两人拳击对打"）；in2IN 可选 individual_1/2
    分别描述每个人（如"凶狠连续出拳"/"举臂格挡后退"）。输出 210 帧 @30fps。
    render_video=True 时同时渲染 mp4（额外约 60~90 秒），files 含 video。
    """
    d = _post('/api/generate_interaction',
              {'model': model, 'interaction': interaction, 'ind1': individual_1,
               'ind2': individual_2, 'seed': seed, 'auto_translate': auto_translate,
               'render_video': render_video})
    return {'id': d['id'], 'model': d['model'], 'm_length': d['m_length'],
            'fps': d['fps'], 'seconds': d['seconds'], 'gen_time': d['gen_time'],
            'translations': d['translations'],
            'files': d['files'], 'urls': _abs_urls(d['files'])}


@mcp.tool()
def generate_momask_dual(text_a: str, text_b: str, length_seconds: float = 0.0,
                         offset_x: float = 0.9, seed: int = 10107,
                         use_ik: bool = True, auto_translate: bool = True,
                         render_video: bool = False) -> dict:
    """基线对比：两个单人 MoMask 动作独立生成后并排摆放（无真实交互）。

    render_video=True 时同时渲染 mp4，files 含 video。
    """
    d = _post('/api/generate_momask_dual',
              {'text_a': text_a, 'text_b': text_b, 'length': length_seconds,
               'offset_x': offset_x, 'seed': seed, 'use_ik': use_ik,
               'auto_translate': auto_translate, 'render_video': render_video})
    return {'id': d['id'], 'model': d['model'], 'm_length': d['m_length'],
            'fps': d['fps'], 'seconds': d['seconds'], 'gen_time': d['gen_time'],
            'translations': d['translations'],
            'files': d['files'], 'urls': _abs_urls(d['files'])}


@mcp.tool()
def generate_kimodo(text: str, duration_seconds: float = 5.0, seed: int = 10107,
                    auto_translate: bool = True, render_video: bool = False) -> dict:
    """用 NVIDIA Kimodo 原版（SOMA-RP v1.1，700h 生产级动捕）生成单人动作。

    输出 SOMA 77 关节（含手指）@30fps。首次调用需把约 17GB 文本编码器加载进
    内存（2~5 分钟），之后单次生成约 30~90 秒。text 支持中文自动翻译。
    """
    d = _post('/api/generate_kimodo', {'text': text, 'duration': duration_seconds,
                                       'seed': seed, 'auto_translate': auto_translate,
                                       'render_video': render_video})
    return {'id': d['id'], 'model': d['model'], 'm_length': d['m_length'],
            'fps': d['fps'], 'seconds': d['seconds'], 'gen_time': d['gen_time'],
            'skeleton': d.get('skeleton', 'soma77'),
            'translations': d['translations'],
            'files': d['files'], 'urls': _abs_urls(d['files'])}


@mcp.tool()
def analyze_motion(result_id: str) -> dict:
    """量化分析一次生成结果：步数、根移动速度、关节能量、左右腕最高点、净位移、
    手臂/腿部活动峰值时刻（判断动作顺序）、两人根部距离范围（双人结果）。"""
    persons = _load_persons(result_id)
    # interaction models run at 30fps; momask/kimodo joints are single-person npy
    fps = 30 if len(persons) > 1 else 30
    if len(persons) == 1 and persons[0].shape[1] != 77:
        fps = 20
    out = {'result_id': result_id, 'persons': len(persons), 'fps': fps,
           'skeleton': 'soma77' if persons[0].shape[1] == 77 else 'smpl22',
           'per_person': []}
    for i, j in enumerate(persons):
        out['per_person'].append({'person': i, **_describe(j, fps)})
    if len(persons) == 2:
        dist = np.linalg.norm(persons[0][:, ROOT] - persons[1][:, ROOT], axis=1)
        out['pair'] = {
            'root_distance_min_m': round(float(dist.min()), 2),
            'root_distance_max_m': round(float(dist.max()), 2),
            'root_distance_mean_m': round(float(dist.mean()), 2),
            'min_joint_distance_m': round(float(min(
                np.linalg.norm(persons[0][f] - persons[1][f], axis=1).min()
                for f in range(0, len(dist), 5))), 2),
        }
    return out


@mcp.tool()
def render_video(result_id: str, title: str = '') -> dict:
    """把生成结果渲染成骨骼动画 mp4（多人同框），返回本地路径与下载 URL。"""
    persons = _load_persons(result_id)
    fps = 30 if (len(persons) > 1 or persons[0].shape[1] == 77) else 20
    skeleton = 'soma77' if persons[0].shape[1] == 77 else 'smpl22'

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from render_utils import render_motion_video

    d = _result_dir(result_id)
    out_path = os.path.join(d, 'animation.mp4')
    render_motion_video(persons, out_path, fps=fps, title=title, skeleton=skeleton)
    return {'result_id': result_id, 'video_path': out_path,
            'video_url': f'{PUBLIC_URL}/results/{result_id}/animation.mp4',
            'frames': int(min(len(p) for p in persons)), 'fps': fps,
            'skeleton': skeleton}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stdio', action='store_true', help='run stdio transport (single local client)')
    parser.add_argument('--port', type=int, default=int(os.environ.get('MCP_PORT', 7864)))
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()

    if args.stdio:
        mcp.run(transport='stdio')
    else:
        mcp.run(transport='streamable-http', host=args.host, port=args.port,
                streamable_http_path='/mcp', stateless_http=True)
