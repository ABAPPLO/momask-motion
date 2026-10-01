"""Kimodo (NVIDIA original) inference sidecar for the comparison webapp.

Loads nv-tlabs/kimodo (SOMA-RP v1.1) once and serves generation over HTTP:
    POST /generate {"text": "...", "duration_seconds": 5.0, "seed": 10107}
    GET  /health

VRAM strategy: TEXT_ENCODER_DEVICE=cpu keeps the LLM2Vec/Llama-3-8B encoder in
system RAM (~17GB); only the diffusion model sits on GPU (<3GB).

The gated meta-llama base is redirected to the NousResearch weight-identical
mirror via a locally patched adapter config (see setup_kimodo_adapters()).

Run (managed automatically by webapp/app.py):
    /home/applo/anaconda3/envs/kimodo/bin/python kimodo_server.py
"""
import argparse
import json
import os
import sys

# env must be set before importing kimodo/huggingface_hub
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('TEXT_ENCODER_MODE', 'local')
os.environ.setdefault('TEXT_ENCODER_DEVICE', 'cpu')

KIMODO_ROOT = '/home/applo/project/kimodo'
ADAPTER_BASE = '/home/applo/project/models/llm2vec-mntp'
ADAPTER_SUP = '/home/applo/project/models/llm2vec-supervised'

sys.path.insert(0, KIMODO_ROOT)
os.chdir(KIMODO_ROOT)


def patch_text_encoder_presets():
    """Point the llm2vec preset at local adapter dirs (base redirected to mirror)."""
    import kimodo.model.load_model  # noqa: F401 — ensure the module is loaded
    lm = sys.modules['kimodo.model.load_model']  # package __init__ re-exports a same-named function
    preset = lm.TEXT_ENCODER_PRESETS['llm2vec']
    preset['kwargs']['base_model_name_or_path'] = ADAPTER_BASE
    preset['kwargs']['peft_model_name_or_path'] = ADAPTER_SUP


_model = None


def get_model():
    global _model
    if _model is None:
        patch_text_encoder_presets()
        from kimodo import load_model
        print('loading Kimodo-SOMA-RP (motion model on cuda, text encoder on cpu)...', flush=True)
        _model = load_model('kimodo-soma-rp')
        print('kimodo model ready', flush=True)
    return _model


def build_app():
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.route('/health')
    def health():
        return jsonify({'status': 'ok', 'model': 'kimodo-soma-rp-v1.1',
                        'loaded': _model is not None})

    @app.route('/generate', methods=['POST'])
    def generate():
        payload = request.get_json(force=True, silent=True) or {}
        text = (payload.get('text') or '').strip()
        if not text:
            return jsonify({'error': 'text is required'}), 400
        try:
            duration = max(1.0, min(float(payload.get('duration_seconds', 5.0)), 20.0))
        except (TypeError, ValueError):
            duration = 5.0
        seed = int(payload.get('seed', 10107) or 10107)

        import numpy as np
        import torch
        from kimodo.tools import seed_everything

        try:
            model = get_model()
            fps = model.fps
            num_frames = int(round(duration * fps))
            seed_everything(seed)
            with torch.no_grad():
                output = model([text], [num_frames], constraint_lst=[],
                               num_denoising_steps=100, num_samples=1,
                               multi_prompt=True, num_transition_frames=5,
                               post_processing=True, return_numpy=True)
            joints = output['posed_joints'][0]  # (T, 77, 3)
            joints = np.round(np.asarray(joints, dtype=float), 4).tolist()
            return jsonify({'joints': joints, 'fps': fps, 'nframes': len(joints),
                            'skeleton': 'soma77', 'text': text, 'seed': seed})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'kimodo generation failed: {e}'}), 500

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=int(os.environ.get('KIMODO_PORT', 7865)))
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()
    app = build_app()
    # preload at startup so the first request is fast
    get_model()
    app.run(host=args.host, port=args.port, threaded=False)
