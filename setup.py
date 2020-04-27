from setuptools import find_packages, setup

setup(
    name='carnav',
    packages=['carnav'],
    version='0.0.1',
    include_package_data=True,
    install_requires=['gym', 'numpy', 'pillow', 'scipy', 'matplotlib']
)