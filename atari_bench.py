from adept.env.env_registry import ATARI_ENVS
from subprocess import call
import os

for env in ATARI_ENVS:
    call([
        'python',
        '-m',
        'adept.app',
        'distrib',
        f'--env {env}',
        '--nb-env 128',
        '--nb-step 50e6',
        '--eval',
        '-y',
        '--logdir ~/Documents/atari_bench'
    ], env=os.environ)
