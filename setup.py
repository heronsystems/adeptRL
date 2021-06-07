from setuptools import setup, find_packages
from adept.globals import VERSION

# https://github.com/kennethreitz/setup.py/blob/master/setup.py


with open("README.md", "r") as fh:
    long_description = fh.read()

extras = {
    "profiler": ["pyinstrument>=2.0"],
    "atari": [
        "gym[atari]>=0.10",
        "opencv-python-headless<4,>=3.4",
    ]
}
test_deps = ["pytest"]

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
all_deps = all_deps + test_deps
extras["all"] = all_deps


setup(
    name="adeptRL",
    version=VERSION,
    author="heron",
    author_email="adept@heronsystems.com",
    description="Reinforcement Learning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/heronsystems/adeptRL",
    license="GNU",
    python_requires=">=3.5.0",
    packages=find_packages(),
    install_requires=[
        "protobuf>=3.15.3",
        "numpy>=1.14",
        "tensorflow<3,>=2.4.0",
        "cloudpickle>=0.5",
        "pyzmq>=17.1.2",
        "docopt>=0.6",
        "torch>=1.3.1",
        "torchvision>=0.4.2",
        "ray>=1.3.0",
        "pandas>=1.0.5",
        "msgpack<2,>=1.0.2",
        "msgpack-numpy<1,>=0.4.7",
    ],
    test_requires=test_deps,
    extras_require=extras,
    include_package_data=True,
)
