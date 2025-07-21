from setuptools import setup, find_packages

setup(
    name="hyperion",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pandas", "numpy"],
    description="GNSS orbit processing toolkit for SP3, YUMA, and RINEX navigation files",
    python_requires=">=3.7",
)