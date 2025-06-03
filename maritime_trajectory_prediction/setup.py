#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="maritime_trajectory_prediction",
    version="0.1.0",
    description="Maritime vessel trajectory prediction using AIS data",
    author="Maritime Research Team",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "pytorch-lightning",
        "torch-geometric",
        "matplotlib",
        "seaborn",
        "hydra-core",
        "wandb",
        "xgboost",
        "pyproj",
        "scikit-learn",
        "folium",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
    },
)