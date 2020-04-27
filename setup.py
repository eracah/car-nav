from setuptools import find_packages, setup

setup(
    name='carnav',
    packages=find_packages(),
    version='0.0.1',
    install_requires=['gym', 'numpy', 'pillow', 'scipy', 'matplotlib']
)