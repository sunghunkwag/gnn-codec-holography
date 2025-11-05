from setuptools import setup, find_packages
import os

# Safely read README.md with fallback
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    print("Warning: README.md not found, using default description")
    long_description = "GNN-based neural network weight compression using holographic representations"

# Safely read requirements.txt with fallback
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    print("Warning: requirements.txt not found, using minimal requirements")
    requirements = [
        "torch>=2.0.0",
        "torch-geometric>=2.4.0", 
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.60.0"
    ]

print(f"Installing with requirements: {requirements}")

setup(
    name="gnn-codec-holography",
    version="0.1.0",
    author="Sung hun kwag",
    author_email="sunghunkwag@gmail.com",
    description="GNN-based neural network weight compression using holographic representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sunghunkwag/gnn-codec-holography",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
)