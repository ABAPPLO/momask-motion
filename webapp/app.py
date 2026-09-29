"""MoMask Web demo — Flask backend serving a text-to-motion test page.

Usage (from the repository root):
    conda activate momask
    python webapp/app.py [--port 8080]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from torch.distributions.categorical import Categorical

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator
from utils.fixseed import fixseed
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from visualization.joints2bvh import Joint2BVHConvertor

clip_version = 'ViT-B/32'

RESULTS_DIR = pjoin(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------- model loading
def load_vq_model(vq_opt):
    vq_model = RVQVAE(vq_opt, vq_opt.dim_pose, vq_opt.nb_code, vq_opt.code_dim,
                      vq_opt.output_emb_width, vq_opt.down_t, vq_opt.stride_t,
                      vq_opt.width, vq_opt.depth, vq_opt.dilation_growth_rate,
                      vq_opt.vq_act, vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                      map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt


def load_trans_model(model_opt, device):
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', 'latest.tar'),
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {model_opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer


def load_res_model(res_opt, vq_opt, device):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                          cond_mode='text',
                                          latent_dim=res_opt.latent_dim,
                                          ff_size=res_opt.ff_size,
                                          num_layers=res_opt.n_layers,
                                          num_heads=res_opt.n_heads,
                                          dropout=res_opt.dropout,
                                          clip_dim=512,
                                          shared_codebook=vq_opt.shared_codebook,
                                          cond_drop_prob=res_opt.cond_drop_prob,
                                          share_weight=res_opt.share_weight,
                                          clip_version=clip_version,
                                          opt=res_opt)
    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                      map_location=device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer


def load_len_estimator(opt):
    model = LengthEstimator(512, 50)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location=opt.device)
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from epoch {ckpt["epoch"]}!')
    return model


class ModelBundle:
    def __init__(self, device):
        self.device = device
        checkpoints_dir = './checkpoints'
        dataset_name = 't2m'
        root_dir = pjoin(checkpoints_dir, dataset_name, 't2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns')

        model_opt = get_opt(pjoin(root_dir, 'opt.txt'), device=device)

        vq_opt = get_opt(pjoin(checkpoints_dir, dataset_name, model_opt.vq_name, 'opt.txt'), device=device)
        vq_opt.dim_pose = 263
        self.vq_model, vq_opt = load_vq_model(vq_opt)

        model_opt.num_tokens = vq_opt.nb_code
        model_opt.num_quantizers = vq_opt.num_quantizers
        model_opt.code_dim = vq_opt.code_dim

        res_opt = get_opt(pjoin(checkpoints_dir, dataset_name, 'tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw', 'opt.txt'),
                          device=device)
        self.res_model = load_res_model(res_opt, vq_opt, device)
        assert res_opt.vq_name == model_opt.vq_name

        self.t2m_transformer = load_trans_model(model_opt, device)
        self.length_estimator = load_len_estimator(model_opt)

        self.t2m_transformer.eval()
        self.vq_model.eval()
        self.res_model.eval()
        self.length_estimator.eval()
        self.res_model.to(device)
        self.t2m_transformer.to(device)
        self.vq_model.to(device)
        self.length_estimator.to(device)

        self.mean = np.load(pjoin(checkpoints_dir, dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
        self.std = np.load(pjoin(checkpoints_dir, dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
        self.converter = Joint2BVHConvertor()

    def inv_transform(self, data):
        return data * self.std + self.mean

    @torch.no_grad()
    def generate(self, text, motion_length=0, use_ik=True, seed=10107,
                 cond_scale=4, temperature=1.0, topkr=0.9, time_steps=18):
        """Returns (raw_joints, ik_joints, m_length) — joints arrays are (nframes, 22, 3)."""
        fixseed(seed)
        prompt_list = [text]
        if motion_length == 0:
            text_embedding = self.t2m_transformer.encode_text(prompt_list)
            pred_dis = self.length_estimator(text_embedding)
            probs = F.softmax(pred_dis, dim=-1)
            token_lens = Categorical(probs).sample()
        else:
            token_lens = torch.LongTensor([motion_length]) // 4
            token_lens = token_lens.to(self.device).long()

        m_length = (token_lens * 4).item()
        captions = prompt_list

        mids = self.t2m_transformer.generate(captions, token_lens,
                                             timesteps=time_steps,
                                             cond_scale=cond_scale,
                                             temperature=temperature,
                                             topk_filter_thres=topkr,
                                             gsample=False)
        mids = self.res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
        pred_motions = self.vq_model.forward_decoder(mids)
        pred_motions = pred_motions.detach().cpu().numpy()
        data = self.inv_transform(pred_motions)

        joint_data = data[0][:m_length]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

        ik_joint = None
        if use_ik:
            _, ik_joint = self.converter.convert(joint, filename=None, iterations=100)
        return joint, ik_joint, m_length


# ---------------------------------------------------------------- flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')
MODEL = None
GEN_LOCK = threading.Lock()

INTERACT_ENV_PYTHON = '/home/applo/anaconda3/envs/interact/bin/python'
INTERGEN_WORKER = '/home/applo/project/InterGen/worker_intergen.py'
IN2IN_WORKER = '/home/applo/project/in2IN/worker_in2in.py'
WORKER_LOCKS = {'intergen': threading.Lock(), 'in2in': threading.Lock()}
WORKERS = {
    'intergen': (os.path.exists(INTERGEN_WORKER), INTERGEN_WORKER),
    'in2in': (os.path.exists(IN2IN_WORKER), IN2IN_WORKER),
}

# ------------------------------------------------------- zh->en translation
TRANSLATOR_URL = 'http://127.0.0.1:7863'
TRANSLATOR_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'translator_server.py')
_translator_proc = None
TRANSLATOR_LOCK = threading.Lock()


def has_cjk(s):
    return any('\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf' for ch in (s or ''))


def translator_alive():
    try:
        with urllib.request.urlopen(TRANSLATOR_URL + '/health', timeout=3) as r:
            return json.load(r).get('status') == 'ok'
    except Exception:
        return False


def ensure_translator(timeout=150):
    """Start the translation sidecar if it is not up; returns True when ready."""
    global _translator_proc
    if translator_alive():
        return True
    with TRANSLATOR_LOCK:
        if translator_alive():
            return True
        if not os.path.exists(TRANSLATOR_SERVER):
            return False
        log = open('/tmp/translator.log', 'a')
        _translator_proc = subprocess.Popen(
            [INTERACT_ENV_PYTHON, TRANSLATOR_SERVER],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if translator_alive():
                return True
            if _translator_proc.poll() is not None:
                return False
            time.sleep(2)
        return False


def translate_texts(texts):
    """Translate a list of prompts; returns (translated_texts, ok_flags).

    English-only or empty strings pass through untouched. On translator
    failure the original text is used and the flag is False.
    """
    results = list(texts)
    ok = [False] * len(texts)
    todo = [i for i, t in enumerate(texts) if t and has_cjk(t)]
    if not todo:
        return results, ok
    if not ensure_translator():
        return results, ok
    body = json.dumps({'texts': [texts[i] for i in todo]}).encode()
    try:
        req = urllib.request.Request(TRANSLATOR_URL + '/translate', data=body,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.load(r)
        for j, i in enumerate(todo):
            en = (out.get('texts_en') or [None] * len(todo))[j]
            if en:
                results[i] = en
                ok[i] = True
    except Exception as e:
        print(f'translation failed, using original text: {e}')
    return results, ok


def translate_payload_texts(payload, fields):
    """Translate the given payload text fields in place; returns {field: en}."""
    if not payload.get('auto_translate', True):
        return {}
    texts = [payload.get(f) or '' for f in fields]
    translated, _ = translate_texts(texts)
    out = {}
    for f, src, en in zip(fields, texts, translated):
        if src and src != en:
            payload[f] = en
            out[f] = {'src': src, 'en': en}
    return out


def run_interaction_worker(model_key, args, timeout=300):
    """Run an isolated inference subprocess; returns parsed JSON dict."""
    ok, worker = WORKERS[model_key]
    if not ok:
        raise RuntimeError(f'worker for {model_key} is not installed')
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
        out_path = tf.name
    cmd = [INTERACT_ENV_PYTHON, worker, '--out', out_path] + args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=os.path.dirname(worker))
        if proc.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            tail = (proc.stderr or proc.stdout or '')[-1500:]
            raise RuntimeError(f'{model_key} worker failed:\n{tail}')
        with open(out_path) as f:
            return json.load(f)
    finally:
        os.unlink(out_path)


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': MODEL is not None,
                    'device': str(MODEL.device) if MODEL else None})


@app.route('/api/generate', methods=['POST'])
def api_generate():
    payload = request.get_json(force=True, silent=True) or {}
    translations = translate_payload_texts(payload, ['text'])
    text = (payload.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'text prompt is required'}), 400

    seconds = payload.get('length', 0)  # 0 => auto estimate
    try:
        seconds = max(0, min(float(seconds), 9.8))
    except (TypeError, ValueError):
        seconds = 0
    motion_length = int(seconds * 20)

    use_ik = bool(payload.get('use_ik', True))
    try:
        seed = int(payload.get('seed', 10107))
    except (TypeError, ValueError):
        seed = 10107

    with GEN_LOCK:
        t0 = time.time()
        try:
            joint, ik_joint, m_length = MODEL.generate(text, motion_length, use_ik, seed)
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'generation failed: {e}'}), 500
        gen_time = time.time() - t0

    rid = uuid.uuid4().hex[:12]
    out_dir = pjoin(RESULTS_DIR, rid)
    os.makedirs(out_dir, exist_ok=True)

    np.save(pjoin(out_dir, 'joints.npy'), joint)
    files = {'npy': f'/results/{rid}/joints.npy'}

    display = ik_joint if ik_joint is not None else joint
    if ik_joint is not None:
        np.save(pjoin(out_dir, 'joints_ik.npy'), ik_joint)
        files['npy_ik'] = f'/results/{rid}/joints_ik.npy'

    try:
        bvh_path = pjoin(out_dir, 'motion.bvh')
        _, _ = MODEL.converter.convert(display, filename=bvh_path, iterations=100)
        files['bvh'] = f'/results/{rid}/motion.bvh'
    except Exception as e:  # BVH export is a nice-to-have; don't fail the request
        print(f'BVH export failed: {e}')

    return jsonify({
        'id': rid,
        'model': 'momask',
        'text': text,
        'translations': translations,
        'm_length': int(m_length),
        'fps': 20,
        'seconds': round(m_length / 20, 2),
        'gen_time': round(gen_time, 2),
        'used_ik': use_ik,
        'joints': np.round(display, 4).tolist(),
        'files': files,
    })


@app.route('/results/<rid>/<path:fname>')
def result_file(rid, fname):
    return send_from_directory(pjoin(RESULTS_DIR, rid), fname, as_attachment=True)


# --------------------------------------------------------- two-person models
@app.route('/api/generate_interaction', methods=['POST'])
def api_generate_interaction():
    payload = request.get_json(force=True, silent=True) or {}
    translations = translate_payload_texts(payload, ['interaction', 'ind1', 'ind2'])
    model_key = payload.get('model', '')
    if model_key not in WORKERS:
        return jsonify({'error': f'unknown model: {model_key}'}), 400
    interaction = (payload.get('interaction') or '').strip()
    if not interaction:
        return jsonify({'error': 'interaction prompt is required'}), 400
    ind1 = (payload.get('ind1') or '').strip()
    ind2 = (payload.get('ind2') or '').strip()
    try:
        seed = int(payload.get('seed', 10107))
    except (TypeError, ValueError):
        seed = 10107

    args = ['--seed', str(seed)]
    if model_key == 'intergen':
        args += ['--prompt', interaction]
    else:
        args += ['--interaction', interaction, '--ind1', ind1, '--ind2', ind2]

    with WORKER_LOCKS[model_key]:
        t0 = time.time()
        try:
            result = run_interaction_worker(model_key, args)
        except subprocess.TimeoutExpired:
            return jsonify({'error': f'{model_key} generation timed out'}), 504
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        gen_time = time.time() - t0

    # persist raw joints for download
    rid = uuid.uuid4().hex[:12]
    out_dir = pjoin(RESULTS_DIR, rid)
    os.makedirs(out_dir, exist_ok=True)
    np.save(pjoin(out_dir, 'person0.npy'), np.array(result['persons'][0]))
    np.save(pjoin(out_dir, 'person1.npy'), np.array(result['persons'][1]))
    files = {
        'npy0': f'/results/{rid}/person0.npy',
        'npy1': f'/results/{rid}/person1.npy',
    }

    return jsonify({
        'id': rid,
        'model': model_key,
        'text': interaction,
        'translations': translations,
        'm_length': result['nframes'],
        'fps': result['fps'],
        'seconds': round(result['nframes'] / result['fps'], 1),
        'gen_time': round(gen_time, 1),
        'persons': result['persons'],
        'files': files,
    })


@app.route('/api/generate_momask_dual', methods=['POST'])
def api_generate_momask_dual():
    """Baseline: two independent single-person MoMask generations, person B offset on X."""
    payload = request.get_json(force=True, silent=True) or {}
    translations = translate_payload_texts(payload, ['text_a', 'text_b'])
    text_a = (payload.get('text_a') or '').strip()
    text_b = (payload.get('text_b') or '').strip()
    if not text_a or not text_b:
        return jsonify({'error': 'both prompts (text_a, text_b) are required'}), 400
    seconds = max(0, min(float(payload.get('length', 0) or 0), 9.8))
    motion_length = int(seconds * 20)
    try:
        seed = int(payload.get('seed', 10107))
    except (TypeError, ValueError):
        seed = 10107
    use_ik = bool(payload.get('use_ik', True))

    offset_x = float(payload.get('offset_x', 0.9))

    persons = []
    m_length = 0
    with GEN_LOCK:
        t0 = time.time()
        try:
            for i, text in enumerate([text_a, text_b]):
                joint, ik_joint, m_length = MODEL.generate(text, motion_length, use_ik, seed + i * 17)
                j = ik_joint if ik_joint is not None else joint
                if i == 1:
                    j = j.copy()
                    j[:, :, 0] += offset_x
                persons.append(np.round(j, 4).tolist())
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'generation failed: {e}'}), 500
        gen_time = time.time() - t0

    rid = uuid.uuid4().hex[:12]
    out_dir = pjoin(RESULTS_DIR, rid)
    os.makedirs(out_dir, exist_ok=True)
    np.save(pjoin(out_dir, 'person0.npy'), np.array(persons[0]))
    np.save(pjoin(out_dir, 'person1.npy'), np.array(persons[1]))
    files = {'npy0': f'/results/{rid}/person0.npy', 'npy1': f'/results/{rid}/person1.npy'}

    # pad shorter sequence so both persons share one timeline
    n = max(len(p) for p in persons)
    for p in persons:
        while len(p) < n:
            p.append(p[-1])

    return jsonify({
        'id': rid,
        'model': 'momask_dual',
        'text': text_a + ' / ' + text_b,
        'translations': translations,
        'm_length': n,
        'fps': 20,
        'seconds': round(n / 20, 1),
        'gen_time': round(gen_time, 1),
        'persons': persons,
        'files': files,
        'note': '两个单人动作独立生成，无真实交互',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=int(os.environ.get('MOMASK_WEB_PORT', 8080)))
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Loading MoMask models on {device} ...')
    t0 = time.time()
    MODEL = ModelBundle(device)
    print(f'Models loaded in {time.time() - t0:.1f}s')

    app.run(host=args.host, port=args.port, threaded=True)
