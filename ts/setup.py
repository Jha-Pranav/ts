from setuptools import find_packages, setup

setup(
    name="ts",
    version="0.0.1",
    author="Pranav Jha",
    author_email="pranavjha1993@gmail.com",
    description="A short description of the project.",
    url="https://github.com/Jha-Pranav/ts.git",
    packages=find_packages(),
    test_suite="src.tests.test_all.suite",
)
