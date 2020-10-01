from pathlib import Path
import setuptools

path = Path(__file__)

setup_requires = ["numpy", "pytest-runner", "pytest-pylint"]

install_requires = [
    "click",
]

test_require = ["pytest-cov", "pytest-html", "pytest", "click", "pylint"]

setuptools.setup(
    name="kgcn_torch",
    version="0.1.0",
    python_requires=">3.5",
    author=["Ryosuke Kojima","Koji Ono"],
    author_email=["kojima.ryosuke.8e@kyoto-u.ac.jp", "kbu94982@gmail.com"],
    description="graph convolutional network library",
    long_description_content_type="text/markdown",
    long_description=(path.parent / "README.md").open(encoding="utf-8").read(),
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    setup_requires=setup_requires,
    url="https://github.com/clinfo/kGCN_torch",
    tests_require=test_require,
    extras_require={"docs": ["sphinx >= 1.4", "sphinx_rtd_theme"]},
    entry_points={
        "console_scripts": [
            "kgcnt = kgcn_torch.backward_compatibility.gcn:main",
            "kgcnt-chem = kgcn_torch.backward_compatibility.chem:main",
            "kgcnt-cv-splitter = kgcn_torch.backward_compatibility.cv_splitter:main",
            "kgcnt-opt = kgcn_torch.backward_compatibility.opt:main",
            "kgcnt-gen = kgcn_torch.backward_compatibility.gen:main",
            "kgcnt-sparse = kgcn_torch.backward_compatibility.task_sparse_gcn:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
