from setuptools import setup, find_packages
from adept.globals import VERSION

# https://github.com/kennethreitz/setup.py/blob/master/setup.py


with open("README.md", "r") as fh:
    long_description = fh.read()

extras = {
    'mpi': ['mpi4py>=3.0'],
    'sc2': ['pysc2>=2.0'],
    'profiler': ['pyinstrument>=2.0']
}
test_deps = ['pytest']

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
all_deps = all_deps + test_deps
extras['all'] = all_deps


setup(
    name='adeptRL',

    version=VERSION,
    author='heron',
    author_email='adept@heronsystems.com',
    description='Reinforcement Learning Framework',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/heronsystems/adeptRL',
    license='GNU',
    python_requires='>=3.5.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.14',
        'gym[atari]>=0.10',
        'absl-py>=0.2',
        'tensorboardX>=1.2',
        'cloudpickle>=0.5',
        'opencv-python-headless>=3.4',
        'pyzmq>=17.1.2',
        'docopt>=0.6'
    ],
    test_requires=test_deps,
    extras_require=extras,
    include_package_data=True
)
