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
]

test_require = [
    'pytest-cov',
    'pytest-html',
    'pytest'
]

setuptools.setup(
    name="kgcn_torch",
    version="0.1.0",
    python_requires='>3.5',
    author="Ryosuke Kojima",
    author_email="kojima.ryosuke.8e@kyoto-u.ac.jp",
    description="graph convolutional network library",
    long_description_content_type="text/markdown",
    long_description=(p.parent / 'README.md').open(encoding='utf-8').read(),
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    setup_requires=setup_requires,
    url="https://github.com/clinfo/kGCN_torch",
    tests_require=test_require,
    extras_require={
        'docs': [
            'sphinx >= 1.4',
            'sphinx_rtd_theme']},
    entry_points={
        'console_scripts': [
            'kgcnt = kgcn_torch.backward_compatibility.console:gcn',
            'kgcnt-chem = kgcn_torch.backward_compatibility.console:chem',
            'kgcnt-cv-splitter = kgcn_torch.backward_compatibility.console:cv_splitter',
            'kgcnt-opt = kgcn_torch.backward_compatibility.console:opt',
            'kgcnt-gen = kgcn_torch.backward_compatibility.console:gen',
            'kgcnt-sparse = kgcn_torch.backward_compatibility.console:task_sparse_gcn',
        ],
    },
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
