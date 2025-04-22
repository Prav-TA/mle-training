from setuptools import find_packages, setup

setup(
    name="test_archive",
    version="0.1",
    packages=find_packages(include=["test_archive", "test_archive.*"]),
)
