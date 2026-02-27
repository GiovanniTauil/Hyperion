from setuptools import setup, find_packages
import os

# To support setup.py living INSIDE the hyperion package directory:
# We map the "hyperion" module to the current directory ".", and 
# dynamically discover its submodules to include them in the installation.
sub_packages = find_packages(where='.')
packages = ['hyperion'] + [f'hyperion.{p}' for p in sub_packages if not p.startswith('data_test')]

setup(
    name="hyperion",
    version="1.0.0",
    description="Python-based open-source toolkit for GNSS satellite orbit processing and I/O parsing.",
    author="Antigravity",
    packages=packages,
    package_dir={'hyperion': '.'},
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "tables",
        "hatanaka"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
