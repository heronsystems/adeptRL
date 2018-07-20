from setuptools import setup
from adept.globals import VERSION

setup(
    name='adeptRL',

    version=VERSION,
    description='adeptRL',
    url='https://github.com/heronsystems/adeptRL',
    author='heron',
    license='GNU',
    packages=[
        'adept',
        'scripts',
    ],

    entry_points={
        'console_scripts': []
    },
    install_requires=[]
)
