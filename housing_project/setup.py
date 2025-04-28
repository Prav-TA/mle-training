from setuptools import find_packages, setup

setup(
    name="src",
    version="1.0.0",
    description="Housing project",
    author="Prav-TA",
    packages=find_packages(include=["src", "src.*"]),
)
