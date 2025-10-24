#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="playground",
    version="0.1.0",
    description="Generative models (AR, MD4, SEDD, RADD, etc.) for the GMM binary dataset",
    author="",
    author_email="",
    url="https://github.com/user/playground",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.2",
        "hydra-colorlog",
        "wandb",
        "tqdm>=4.66",
        "numpy>=1.24",
        "matplotlib>=3.7",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "playground-train = playground.train:main",
            "playground-eval = playground.eval:main",
        ]
    },
)
