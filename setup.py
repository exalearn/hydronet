from setuptools import setup, find_packages

setup(
    name='hydronet',
    packages=find_packages(),
    version='0.0.1',
    install_requires=['networkx','ase','pandas']
)
