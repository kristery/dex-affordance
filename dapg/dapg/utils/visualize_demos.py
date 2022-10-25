import mj_envs
import click
import os
import gym
import numpy as np
import pickle

DESC = '''
Helper script to visualize demonstrations.\n
USAGE:\n
    Visualizes demonstrations on the env\n
    $ python utils/visualize_demos --env_name relocate-v0\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required=True)
@click.option('--demos_file', type=str, help='path to demos file', required=False)
def main(env_name, demos_file):
    if env_name is "":
        print("Unknown env.")
        return
    if demos_file is None:
        demos_file = './demonstrations/' + env_name + '_demos.pickle'
    print('Loading demos from: {}'.format(demos_file))
    demos = pickle.load(open(demos_file, 'rb'))
    # render demonstrations
    demo_playback(env_name, demos)

def demo_playback(env_name, demo_paths):
    e = gym.make(env_name)
    e.reset()
    for path in demo_paths:
        e.env.ss(path['init_state_dict'])
        actions = path['actions']
        for t in range(actions.shape[0]):
            e.step(actions[t])
            e.env.mj_render()

if __name__ == '__main__':
    main()
