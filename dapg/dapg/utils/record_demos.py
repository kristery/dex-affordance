import mj_envs
import click
import os
import gym
import numpy as np
import pickle

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.random_policy import RandomPolicy

DESC = '''
Helper script to record demonstrations.\n
USAGE:\n
    Records demonstrations on the env\n
    $ python utils/record_demos --env_name relocate-v0 --policy policies/relocate-v0.pickle\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--policy', type=str, help='absolute path of the policy file', required=True)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('--out_dir', type=str, help='output directory', default='/tmp')
def main(env_name, policy, mode, out_dir):
    # Record demos
    paths = record_demos(env_name, policy, mode)
    # Save demons
    out_f = os.path.join(out_dir, '{}_demos.pickle'.format(env_name))
    with open(out_f, 'wb') as f:
        pickle.dump(paths, f)
    print('Wrote demos to: {}'.format(out_f))


def record_demos(env_name, policy, mode):
    # Construct the env
    e = GymEnv(env_name)
    # Load the policy
    with open(policy, 'rb') as f:
        pi = pickle.load(f)
    # Record demos
    paths = e.record_demos(pi, num_episodes=5, horizon=e.horizon, mode=mode)
    return paths


if __name__ == '__main__':
    main()
