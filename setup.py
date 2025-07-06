from setuptools import setup, find_packages

packages = find_packages()

setup(
    name="FinanceQA",
    version="0.1.0",
    description="A project for generating and processing financial QA data.",
    packages=packages,
    python_requires=">=3.7",
)
