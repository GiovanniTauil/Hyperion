from setuptools import setup, find_packages

setup(
    name="hyperion",
    version="1.0.0-rc1",
    packages=find_packages(),
    install_requires=["pandas", "numpy"],
    description="GNSS orbit processing toolkit for SP3, YUMA, RINEX navigation/observation, and IONEX files",
    python_requires=">=3.7",
)