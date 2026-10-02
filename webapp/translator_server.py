"""Local zh->en translation sidecar for the motion-generation webapp.

Loads Helsinki-NLP/opus-mt-zh-en once and serves fast translations over HTTP:
    POST /translate  {"text": "中文"}        -> {"text_en": "..."}
    POST /translate  {"texts": ["中", "文"]} -> {"texts_en": ["...", "..."]}
    GET  /health                             -> {"status": "ok"}

Run (managed automatically by webapp/app.py):
    /home/applo/anaconda3/envs/interact/bin/python translator_server.py
"""
import argparse
import os

MODEL_DIR = '/home/applo/project/models/opus-mt-zh-en'

# Domain glossary: pre-substitute motion terms the small MT model often
# mistranslates (e.g. 跑步机 -> "jockey"). Longer keys first.
GLOSSARY_ZH_EN = [
    ('跑步机', ' treadmill '), ('跳绳', ' jump rope '), ('哑铃', ' dumbbell '), ('杠铃', ' barbell '),
    ('俯卧撑', ' push-up '), ('仰卧起坐', ' sit-up '), ('开合跳', ' jumping jack '), ('波比跳', ' burpee '),
    ('深蹲', ' squat '), ('后空翻', ' backflip '), ('侧手翻', ' cartwheel '), ('空翻', ' flip '),
    ('倒立', ' handstand '), ('劈叉', ' split '), ('箭步蹲', ' lunge '),
    ('广场舞', ' square dance '), ('街舞', ' breakdance '), ('芭蕾', ' ballet '), ('华尔兹', ' waltz '),
    ('探戈', ' tango '), ('桑巴', ' samba '), ('萨尔萨', ' salsa '), ('拉丁舞', ' latin dance '),
    ('踢踏舞', ' tap dance '), ('民族舞', ' folk dance '), ('现代舞', ' contemporary dance '),
    ('太极拳', ' tai chi '), ('太极', ' tai chi '), ('咏春', ' wing chun '), ('功夫', ' kung fu '),
    ('空手道', ' karate '), ('跆拳道', ' taekwondo '), ('柔道', ' judo '),
    ('拳击', ' boxing '), ('击剑', ' fencing '), ('摔跤', ' wrestling '), ('武术', ' martial arts '),
    ('出拳', ' punching '), ('挥拳', ' punching '), ('踢腿', ' kicking '), ('蹬腿', ' kicking '),
    ('挥手', ' waving '), ('招手', ' waving '), ('鞠躬', ' bowing '), ('拥抱', ' hugging '),
    ('握手', ' shaking hands '), ('点头', ' nodding '), ('摇头', ' shaking head '),
    ('左臂', ' left arm '), ('右臂', ' right arm '), ('左手', ' left hand '), ('右手', ' right hand '),
    ('左腿', ' left leg '), ('右腿', ' right leg '), ('左脚', ' left foot '), ('右脚', ' right foot '),
    ('向后', ' backward '), ('向前', ' forward '), ('原地', ' in place '), ('转圈', ' in a circle '),
    ('慢速', ' slowly '), ('快速', ' quickly '), ('慢慢地', ' slowly '), ('快速地', ' quickly '),
]


def apply_glossary(text):
    for zh, en in GLOSSARY_ZH_EN:
        if zh in text:
            text = text.replace(zh, en)
    return ' '.join(text.split())

app = None  # flask app, created in main


def build_app(translator, model_name):
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.route('/health')
    def health():
        return jsonify({'status': 'ok', 'model': model_name})

    @app.route('/translate', methods=['POST'])
    def translate():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            if 'texts' in payload:
                texts = [apply_glossary(str(t)[:512]) for t in payload['texts']]
                out = translator(texts)
                return jsonify({'texts_en': out})
            text = apply_glossary(str(payload.get('text', ''))[:512])
            return jsonify({'text_en': translator([text])[0]})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app


def main():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='nllb',
                        choices=['nllb', 'opus'],
                        help='nllb = facebook/nllb-200-distilled-600M (better), '
                             'opus = Helsinki-NLP/opus-mt-zh-en (smaller)')
    parser.add_argument('--port', type=int, default=int(os.environ.get('TRANSLATOR_PORT', 7863)))
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()

    NLLB_DIR = '/home/applo/project/models/nllb-200-distilled-600M'
    OPUS_DIR = '/home/applo/project/models/opus-mt-zh-en'
    model_dir = NLLB_DIR if args.model == 'nllb' else OPUS_DIR
    if not os.path.exists(os.path.join(model_dir, 'pytorch_model.bin')):
        print(f'{args.model} weights missing at {model_dir}, falling back to the other model')
        model_dir = OPUS_DIR if args.model == 'nllb' else NLLB_DIR
        args.model = 'opus' if model_dir == OPUS_DIR else 'nllb'

    device_env = os.environ.get('TRANSLATOR_DEVICE', 'auto').lower()
    if device_env in ('cpu', 'cuda', 'cuda:0'):
        device = torch.device(device_env)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'loading {args.model} translator on {device} ...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    print('translator ready', flush=True)

    if args.model == 'nllb':
        @torch.no_grad()
        def translate(texts):
            tokenizer.src_lang = 'zho_Hans'
            enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                            max_length=256).to(device)
            bos_id = tokenizer.convert_tokens_to_ids('eng_Latn')
            gen = model.generate(**enc, forced_bos_token_id=bos_id,
                                 max_new_tokens=256, num_beams=4)
            return tokenizer.batch_decode(gen, skip_special_tokens=True)
    else:
        @torch.no_grad()
        def translate(texts):
            enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                            max_length=256).to(device)
            gen = model.generate(**enc, max_new_tokens=256, num_beams=4)
            return tokenizer.batch_decode(gen, skip_special_tokens=True)

    app = build_app(translate, args.model)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
