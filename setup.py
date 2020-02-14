import os
import re
import setuptools
from pathlib import Path

p = Path(__file__)

setup_requires = [
    'numpy',
    'pytest-runner'
]

install_requires = [
    'tensorboardx'
]
test_require = [
    'pytest-cov',
    'pytest-html',
    'pytest'
]

setuptools.setup(
    name="tfcg",
    version='0.1.0',
    python_requires='>3.5',
    author="Koji Ono",
    author_email="kbu94982@gmail.com",
    description="Pytorch Extension Module",
    url='https://github.com/0h-n0/tfcg',
    long_description=(p.parent / 'README.md').open(encoding='utf-8').read(),
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=test_require,
    extras_require={
        'docs': [
            'sphinx >= 1.4',
            'sphinx_rtd_theme']},
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)
