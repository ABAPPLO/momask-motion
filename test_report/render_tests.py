"""Render review videos for key test cases (same seeds -> same motions)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_tests import (gen_momask, gen_interaction, render_video, VIDEOS)


def main():
    # --- T2 tempo: MoMask walk slow->run (fixed 6s) ---
    for name, prompt in [
            ('t2_walk_verySlow', 'A person walks very slowly.'),
            ('t2_walk_normal', 'A person walks.'),
            ('t2_walk_veryFast', 'A person walks very fast.'),
            ('t2_run_fast', 'A person is running as fast as possible.')]:
        j, d = gen_momask(prompt, 6.0, 10107)
        render_video([j], name + '.mp4', d['fps'], title=prompt)
        print('rendered', name, flush=True)

    # --- T1 alignment: arm L vs R ---
    for name, prompt in [
            ('t1_armL', 'A person raises his left arm straight up and holds it.'),
            ('t1_armR', 'A person raises his right arm straight up and holds it.')]:
        j, d = gen_momask(prompt, 6.0, 10107)
        render_video([j], name + '.mp4', d['fps'], title=prompt)
        print('rendered', name, flush=True)

    # --- InterGen: order + tempo ---
    for name, prompt in [
            ('t1_ig_hugOrder', 'Two people first shake hands and then hug each other.'),
            ('t2_ig_danceSlow', 'Two people dance together slowly and calmly.'),
            ('t2_ig_danceFast', 'Two people dance together quickly and energetically.')]:
        persons, d = gen_interaction('intergen', prompt, seed=10107)
        render_video(persons, name + '.mp4', d['fps'], title=prompt)
        print('rendered', name, flush=True)

    # --- in2IN: per-person handedness ---
    persons, d = gen_interaction(
        'in2in', 'Two people raise their hands to greet each other.',
        ind1='A person raises his left arm high while keeping the right arm down.',
        ind2='A person raises his right arm high while keeping the left arm down.',
        seed=10107)
    render_video(persons, 't1_i2_handedness.mp4', d['fps'],
                 title='in2IN greet: A=LEFT up (blue), B=RIGHT up (orange)')
    print('rendered t1_i2_handedness', flush=True)


if __name__ == '__main__':
    main()
