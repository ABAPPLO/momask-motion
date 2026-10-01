"""Multi-person skeleton video renderer (mp4) for the webapp.

Used by the generate endpoints when the caller asks for `render_video: true`.
Runs in the momask env (matplotlib + ffmpeg). Supports smpl22 (22 joints) and
soma77 (Kimodo's 77-joint skeleton) hierarchies.
"""
import os

# SMPL/HumanML3D 22-joint kinematic chains
SMPL22_CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
                [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

# SOMA77 joint order (mirrors nv-tlabs/kimodo skeleton/definitions.py)
_FINGER_JOINT_NUMS = {'Thumb': (1, 2, 3), 'Index': (1, 2, 3, 4), 'Middle': (1, 2, 3, 4),
                      'Ring': (1, 2, 3, 4), 'Pinky': (1, 2, 3, 4)}  # thumb has no 4


def _hand_joint_names(side):
    out = []
    for finger, nums in _FINGER_JOINT_NUMS.items():
        out += [f'{side}Hand{finger}{i}' for i in nums] + [f'{side}Hand{finger}End']
    return out


_SOMA_NAMES = (['Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head', 'HeadEnd',
                'Jaw', 'LeftEye', 'RightEye']
               + ['LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'] + _hand_joint_names('Left')
               + ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand'] + _hand_joint_names('Right')
               + ['LeftLeg', 'LeftShin', 'LeftFoot', 'LeftToeBase', 'LeftToeEnd',
                  'RightLeg', 'RightShin', 'RightFoot', 'RightToeBase', 'RightToeEnd'])
assert len(_SOMA_NAMES) == 77, len(_SOMA_NAMES)

_SOMA_PARENT_NAMES = {
    'Hips': None, 'Spine1': 'Hips', 'Spine2': 'Spine1', 'Chest': 'Spine2',
    'Neck1': 'Chest', 'Neck2': 'Neck1', 'Head': 'Neck2', 'HeadEnd': 'Head',
    'Jaw': 'Head', 'LeftEye': 'Head', 'RightEye': 'Head',
    'LeftShoulder': 'Chest', 'LeftArm': 'LeftShoulder', 'LeftForeArm': 'LeftArm', 'LeftHand': 'LeftForeArm',
    'RightShoulder': 'Chest', 'RightArm': 'RightShoulder', 'RightForeArm': 'RightArm', 'RightHand': 'RightForeArm',
    'LeftLeg': 'Hips', 'LeftShin': 'LeftLeg', 'LeftFoot': 'LeftShin', 'LeftToeBase': 'LeftFoot', 'LeftToeEnd': 'LeftToeBase',
    'RightLeg': 'Hips', 'RightShin': 'RightLeg', 'RightFoot': 'RightShin', 'RightToeBase': 'RightFoot', 'RightToeEnd': 'RightToeBase',
}
for _side in ('Left', 'Right'):
    for _finger, _nums in _FINGER_JOINT_NUMS.items():
        for _i in _nums:
            _SOMA_PARENT_NAMES[f'{_side}Hand{_finger}{_i}'] = (
                f'{_side}Hand{_finger}{_i - 1}' if _i > 1 else f'{_side}Hand')
        _SOMA_PARENT_NAMES[f'{_side}Hand{_finger}End'] = f'{_side}Hand{_finger}{_nums[-1]}'

# facial joints skipped in stick rendering
_SOMA_SKIP = {'Jaw', 'LeftEye', 'RightEye'}
SOMA77_EDGES = []
for _i, _name in enumerate(_SOMA_NAMES):
    _p = _SOMA_PARENT_NAMES.get(_name)
    if _p is None or _name in _SOMA_SKIP or _p in _SOMA_SKIP:
        continue
    SOMA77_EDGES.append((_SOMA_NAMES.index(_p), _i))

_SOMA_THIN = set()
for _i, _n in enumerate(_SOMA_NAMES):
    if 'Hand' in _n or 'Toe' in _n:
        _SOMA_THIN.add(_i)
_SOMA_SPINE = {0, 1, 2, 3, 4, 5, 6}


def _edge_specs(skeleton):
    """Return list of (a, b, radius) for the requested skeleton."""
    if skeleton == 'soma77':
        return [(a, b, 0.008 if (a in _SOMA_THIN or b in _SOMA_THIN)
                 else (0.024 if a in _SOMA_SPINE else 0.013))
                for a, b in SOMA77_EDGES]
    seen, out = set(), []
    for chain in SMPL22_CHAIN:
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            spine = key in {(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)}
            out.append((a, b, 0.021 if spine else 0.013))
    return out


def render_motion_video(persons, out_path, fps, title='', skeleton='smpl22'):
    """persons: list of (nframes, J, 3) joint arrays (world coords, y-up)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the '3d' projection
    from matplotlib import animation
    import numpy as np

    edges = _edge_specs(skeleton)
    head = 7 if skeleton == 'soma77' else 15
    colors = ['royalblue', 'darkorange']
    n = min(len(p) for p in persons)
    root = 0

    fig = plt.figure(figsize=(6, 6), dpi=72)
    ax = fig.add_subplot(111, projection='3d')
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)

    with writer.saving(fig, out_path, dpi=72):
        for f in range(n):
            ax.cla()
            frames = [p[min(f, len(p) - 1)] for p in persons]
            allc = np.concatenate(frames, axis=0)
            pad = 0.5
            ax.set_xlim(allc[:, 0].min() - pad, allc[:, 0].max() + pad)
            ax.set_ylim(allc[:, 1].min() - pad, allc[:, 1].max() + pad)
            ax.set_zlim(allc[:, 2].min() - pad, allc[:, 2].max() + pad)
            for ci, fr in enumerate(frames):
                col = colors[ci % len(colors)]
                for a, b, r in edges:
                    ax.plot(fr[[a, b], 0], fr[[a, b], 1], fr[[a, b], 2],
                            c=col, linewidth=2.5 if r > 0.02 else 1.6)
                main = [i for i in range(len(fr)) if i != head]
                ax.scatter(fr[main, 0], fr[main, 1], fr[main, 2], c=col, s=10)
                ax.scatter([fr[head, 0]], [fr[head, 1]], [fr[head, 2]], c=col, s=55)
                tr = persons[ci][:f + 1, root, :]
                ax.plot(tr[:, 0], np.full(len(tr), 0), tr[:, 2],
                        c=col, alpha=0.35, linewidth=1)
            if title:
                ax.set_title(title[:80], fontsize=9)
            ax.view_init(elev=12, azim=-90)
            writer.grab_frame()
    plt.close(fig)
    return out_path
