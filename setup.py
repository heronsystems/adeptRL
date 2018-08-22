from setuptools import setup, find_packages
from adept.globals import VERSION

# https://github.com/kennethreitz/setup.py/blob/master/setup.py

setup(
    name='adept',

    version=VERSION,
    description='Reinforcement Learning Framework',
    url='https://github.com/heronsystems/adeptRL',
    author='heron',
    license='GNU',
    python_requires='>=3.5.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.14',
        'gym[atari]>=0.10',
        'absl-py>=0.2',
        'tensorboardX>=1.2',
        'cloudpickle>=0.5',
        'opencv-python>=3.4'
    ],
    extras_require={
        'mpi': ['mpi4py>=3.0'],
        'sc2': ['pysc2>=2.0'],
        'profiler': ['pyinstrument>=2.0']
    },
    include_package_data=True
)
