from setuptools import setup, find_packages

setup(
    name="hyperion",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pandas"],
    description="GNSS SP3 utilities",
    python_requires=">=3.7",
)