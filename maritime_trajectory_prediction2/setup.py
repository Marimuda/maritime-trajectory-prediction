"""
Setup script for maritime trajectory prediction package.
"""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="maritime-trajectory-prediction",
    version="0.1.0",
    author="Jákup Svøðstein",
    author_email="jakupsv@setur.fo",
    description="A package for predicting maritime vessel trajectories using transformer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jakupsv/maritime-trajectory-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "ais-predict=maritime_trajectory_prediction.scripts.predict_trajectory:main",
            "ais-process=maritime_trajectory_prediction.scripts.process_ais_catcher:main",
        ],
    },
)
