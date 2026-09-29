"""Text-motion alignment & tempo control test harness.

Runs prompt suites through the deployed web API (MoMask / InterGen / in2IN),
computes quantitative metrics on the generated joint sequences, renders review
videos, and writes a markdown report.

Usage:  /home/applo/anaconda3/envs/momask/bin/python run_tests.py [--quick]
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BASE = 'http://127.0.0.1:7862'
OUT = os.path.dirname(os.path.abspath(__file__))
VIDEOS = os.path.join(OUT, 'videos')
os.makedirs(VIDEOS, exist_ok=True)

# SMPL/HumanML3D joint indices
L_WRIST, R_WRIST = 20, 21
L_FOOT, R_FOOT = 10, 11
ROOT = 0
ARM_IDX = [18, 19, 20, 21]
LEG_IDX = [4, 5, 7, 8, 10, 11]
PERSON_COLORS = ['#5b8cff', '#ff9e5b']


# ---------------------------------------------------------------- api helpers
def post(path, body, timeout=360):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def gen_momask(text, length=6.0, seed=10107):
    d = post('/api/generate', {'text': text, 'length': length, 'use_ik': True, 'seed': seed})
    return np.array(d['joints']), d


def gen_interaction(model, interaction, ind1='', ind2='', seed=10107):
    d = post('/api/generate_interaction', {'model': model, 'interaction': interaction,
                                           'ind1': ind1, 'ind2': ind2, 'seed': seed})
    return [np.array(p) for p in d['persons']], d


# ---------------------------------------------------------------- metrics
def wrist_gain(j, side):
    """Height gain of a wrist vs its initial posture."""
    idx = L_WRIST if side == 'l' else R_WRIST
    y = j[:, idx, 1]
    return float(y.max() - np.median(y[:10]))


def foot_lifts(j, fps, thresh=0.045):
    """Number of foot-lift events (both feet)."""
    total = 0
    for idx in (L_FOOT, R_FOOT):
        y = j[:, idx, 1]
        peaks, _ = find_peaks(y - np.median(y), height=thresh,
                              distance=max(1, int(fps * 0.3)))
        total += len(peaks)
    return int(total)


def root_speed(j, fps):
    v = np.linalg.norm(np.diff(j[:, ROOT, :], axis=0), axis=1) * fps
    return float(v.mean())


def joint_energy(j, fps):
    """Mean per-joint speed (m/s), a proxy for overall motion energy."""
    v = np.linalg.norm(np.diff(j, axis=0), axis=2).mean(axis=1) * fps
    return float(v.mean())


def net_displacement(j):
    return j[-1, ROOT, :] - j[0, ROOT, :]


def arm_leg_peak_times(j, fps):
    """Peak-activity time (s) of arms vs legs envelopes."""
    d = np.diff(j, axis=0)
    arm = np.linalg.norm(d[:, ARM_IDX], axis=2).mean(axis=1)
    leg = np.linalg.norm(d[:, LEG_IDX], axis=2).mean(axis=1)
    sigma = max(1, fps // 3)
    arm = gaussian_filter1d(arm, sigma)
    leg = gaussian_filter1d(leg, sigma)
    return float(arm.argmax()) / fps, float(leg.argmax()) / fps


def min_wrist_y(j, side):
    idx = L_WRIST if side == 'l' else R_WRIST
    return float(j[:, idx, 1].min())


def pair_distance(persons):
    return np.linalg.norm(persons[0][:, ROOT, :] - persons[1][:, ROOT, :], axis=1)


# ---------------------------------------------------------------- video render
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21], [9, 13, 16, 18, 20],
]


def render_video(persons, fname, fps, title=''):
    """Multi-person stick-figure video, all persons in one frame."""
    colors = ['royalblue', 'darkorange']
    n = min(len(p) for p in persons)
    allp = np.concatenate(persons, axis=1)  # for bounds
    xmin, xmax = allp[:, :, 0].min(), allp[:, :, 0].max()
    zmin, zmax = allp[:, :, 2].min(), allp[:, :, 2].max()
    pad = 0.5
    fig = plt.figure(figsize=(6, 6), dpi=72)
    ax = fig.add_subplot(111, projection='3d')
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)

    lines_by_p, pts_by_p = [], []
    with writer.saving(fig, os.path.join(VIDEOS, fname), dpi=72):
        for f in range(n):
            ax.cla()
            frames = [p[min(f, len(p) - 1)] for p in persons]
            allc = np.concatenate(frames, axis=0)
            ax.set_xlim(allc[:, 0].min() - pad, allc[:, 0].max() + pad)
            ax.set_ylim(allc[:, 1].min() - pad, allc[:, 1].max() + pad)
            ax.set_zlim(allc[:, 2].min() - pad, allc[:, 2].max() + pad)
            for ci, fr in enumerate(frames):
                col = colors[ci % len(colors)]
                for chain in KINEMATIC_CHAIN:
                    pts = fr[chain]
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c=col, linewidth=2.5)
                ax.scatter(fr[:, 0], fr[:, 1], fr[:, 2], c=col, s=14)
            # ground trail of roots
            for ci, p in enumerate(persons):
                tr = p[:f + 1, ROOT, :]
                ax.plot(tr[:, 0], np.full(len(tr), 0), tr[:, 2],
                        c=colors[ci % len(colors)], alpha=0.35, linewidth=1)
            ax.set_title(title, fontsize=9)
            ax.view_init(elev=12, azim=-90)
            writer.grab_frame()
    plt.close(fig)


# ---------------------------------------------------------------- test defs
def t1_single():
    """Alignment tests on single-person prompts (MoMask)."""
    cases = [
        ('armL', 'A person raises his left arm straight up and holds it.',
         lambda j, fps: {'gain_L': round(wrist_gain(j, 'l'), 2), 'gain_R': round(wrist_gain(j, 'r'), 2)}),
        ('armR', 'A person raises his right arm straight up and holds it.',
         lambda j, fps: {'gain_L': round(wrist_gain(j, 'l'), 2), 'gain_R': round(wrist_gain(j, 'r'), 2)}),
        ('walkF', 'A person walks forward.',
         lambda j, fps: {'disp': [round(float(x), 2) for x in net_displacement(j)]}),
        ('walkB', 'A person walks backward.',
         lambda j, fps: {'disp': [round(float(x), 2) for x in net_displacement(j)]}),
        ('kickThenPunch', 'A person first kicks with the right leg, then throws punches.',
         lambda j, fps: dict(zip(('t_arm_peak_s', 't_leg_peak_s'),
                                 [round(x, 1) for x in arm_leg_peak_times(j, fps)]))),
        ('punchThenKick', 'A person first throws punches, then kicks with the right leg.',
         lambda j, fps: dict(zip(('t_arm_peak_s', 't_leg_peak_s'),
                                 [round(x, 1) for x in arm_leg_peak_times(j, fps)]))),
        ('steps3', 'A person takes three steps forward.',
         lambda j, fps: {'steps': foot_lifts(j, fps)}),
        ('steps8', 'A person takes eight steps forward.',
         lambda j, fps: {'steps': foot_lifts(j, fps)}),
        ('pickupR', 'A person bends down and picks up an object from the ground with his right hand.',
         lambda j, fps: {'min_R_wrist_y': round(min_wrist_y(j, 'r'), 2),
                         'min_L_wrist_y': round(min_wrist_y(j, 'l'), 2)}),
    ]
    results = []
    for name, prompt, metric in cases:
        for seed in (10107, 20260926):
            j, d = gen_momask(prompt, 6.0, seed)
            row = {'case': name, 'model': 'momask', 'seed': seed, 'fps': d['fps'],
                   'prompt': prompt, **metric(j, d['fps'])}
            results.append(row)
            print(f"[momask/{name}/s{seed}] {row}", flush=True)
    return results


def t1_interaction():
    results = []
    # InterGen: interaction-level details (order, spatial relation)
    ig_cases = [
        ('ig_hugOrder', 'Two people first shake hands and then hug each other.'),
        ('ig_walkAway', 'Two people stand face to face, then turn around and walk away from each other.'),
        ('ig_danceSlow', 'Two people dance together slowly and calmly.'),
        ('ig_danceFast', 'Two people dance together quickly and energetically.'),
    ]
    for name, prompt in ig_cases:
        persons, d = gen_interaction('intergen', prompt, seed=10107)
        dist = pair_distance(persons)
        row = {'case': name, 'model': 'intergen', 'seed': 10107, 'fps': d['fps'],
               'prompt': prompt, 'gen_time': d['gen_time'],
               'dist_min': round(float(dist.min()), 2),
               'dist_max': round(float(dist.max()), 2),
               'dist_first1s': round(float(dist[:d['fps']].mean()), 2),
               'dist_last1s': round(float(dist[-d['fps']:].mean()), 2)}
        if name.startswith('ig_dance'):
            row['energy_p0'] = round(joint_energy(persons[0], d['fps']), 2)
            row['energy_p1'] = round(joint_energy(persons[1], d['fps']), 2)
            row['steps_p0'] = foot_lifts(persons[0], d['fps'])
        results.append(row)
        print(f"[{name}] {row}", flush=True)

    # in2IN: per-person detail control (handedness)
    persons, d = gen_interaction(
        'in2in', 'Two people raise their hands to greet each other.',
        ind1='A person raises his left arm high while keeping the right arm down.',
        ind2='A person raises his right arm high while keeping the left arm down.',
        seed=10107)
    p0, p1 = persons
    row = {'case': 'i2_handedness', 'model': 'in2in', 'seed': 10107, 'fps': d['fps'],
           'prompt': 'greet: A=LEFT arm up, B=RIGHT arm up',
           'A_gain_L': round(wrist_gain(p0, 'l'), 2), 'A_gain_R': round(wrist_gain(p0, 'r'), 2),
           'B_gain_L': round(wrist_gain(p1, 'l'), 2), 'B_gain_R': round(wrist_gain(p1, 'r'), 2),
           'gen_time': d['gen_time']}
    results.append(row)
    print(f"[i2_handedness] {row}", flush=True)

    # in2IN speed wording
    for name, word in [('i2_danceSlow', 'slowly and calmly'), ('i2_danceFast', 'quickly and energetically')]:
        persons, d = gen_interaction(
            'in2in', f'Two people dance together {word}.',
            ind1=f'A person dances {word}.',
            ind2=f'A person dances {word}.',
            seed=10107)
        p0, p1 = persons
        dist = pair_distance(persons)
        row = {'case': name, 'model': 'in2in', 'seed': 10107, 'fps': d['fps'],
               'prompt': f'dance {word}', 'gen_time': d['gen_time'],
               'energy_p0': round(joint_energy(p0, d['fps']), 2),
               'energy_p1': round(joint_energy(p1, d['fps']), 2),
               'steps_p0': foot_lifts(p0, d['fps']),
               'dist_min': round(float(dist.min()), 2), 'dist_max': round(float(dist.max()), 2)}
        results.append(row)
        print(f"[{name}] {row}", flush=True)
    return results


def t2_single():
    """Tempo tests on MoMask: fixed 6s duration, only speed wording changes."""
    cases = [
        ('walk_verySlow', 'A person walks very slowly.'),
        ('walk_normal', 'A person walks.'),
        ('walk_veryFast', 'A person walks very fast.'),
        ('run_fast', 'A person is running as fast as possible.'),
        ('swim_slow', 'A person swims slowly.'),
        ('swim_fast', 'A person swims fast.'),
    ]
    results = []
    for name, prompt in cases:
        for seed in (10107, 20260926):
            j, d = gen_momask(prompt, 6.0, seed)
            row = {'case': name, 'model': 'momask', 'seed': seed, 'fps': d['fps'],
                   'prompt': prompt,
                   'steps': foot_lifts(j, d['fps']),
                   'steps_per_s': round(foot_lifts(j, d['fps']) / (len(j) / d['fps']), 2),
                   'root_speed': round(root_speed(j, d['fps']), 2),
                   'energy': round(joint_energy(j, d['fps']), 2)}
            results.append(row)
            print(f"[{name}/s{seed}] {row}", flush=True)
    return results


# ---------------------------------------------------------------- main
def main():
    quick = '--quick' in sys.argv
    t0 = time.time()
    all_results = {}

    print('=== T1a: MoMask alignment ===', flush=True)
    all_results['t1_single'] = t1_single()

    print('=== T2a: MoMask tempo ===', flush=True)
    all_results['t2_single'] = t2_single()

    print('=== T1b/T2b: interaction models ===', flush=True)
    all_results['t1_interaction'] = t1_interaction()

    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=1)
    print(f'ALL DONE in {time.time()-t0:.0f}s -> results.json', flush=True)


if __name__ == '__main__':
    main()
