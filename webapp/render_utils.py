"""Multi-person skeleton video renderer (mp4) for the webapp.

Used by the generate endpoints when the caller asks for `render_video: true`.
Runs in the momask env (matplotlib + ffmpeg).
"""
import os


def render_motion_video(persons, out_path, fps, title=''):
    """persons: list of (nframes, 22, 3) joint arrays (world coords, y-up)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import numpy as np

    chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
             [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
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
                for ch in chain:
                    pts = fr[ch]
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c=col, linewidth=2.5)
                ax.scatter(fr[:, 0], fr[:, 1], fr[:, 2], c=col, s=14)
                tr = persons[ci][:f + 1, root, :]
                ax.plot(tr[:, 0], np.full(len(tr), 0), tr[:, 2],
                        c=col, alpha=0.35, linewidth=1)
            if title:
                ax.set_title(title[:80], fontsize=9)
            ax.view_init(elev=12, azim=-90)
            writer.grab_frame()
    plt.close(fig)
    return out_path
