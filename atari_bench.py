from adept.environments.env_registry import ATARI_ENVS
from subprocess import call
import os

for env in ATARI_ENVS:
    exit(
        call([
            'python',
            '-m',
            'adept.scripts.distrib',
            f'--env {env}'
            '--nb-env 128',
            '--nb-step 50e6',
            '--eval',
            '--use-defaults',
            '--logdir ~/Documents/atari_bench'
        ], env=os.environ)
    )
