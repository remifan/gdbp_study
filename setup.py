# coding=utf-8
"""Install GDBP."""

from setuptools import setup, find_packages

setup(name='gdbp',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['tqdm', 'wget']
)
