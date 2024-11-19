from setuptools import setup, find_packages

setup(
    name="drl-aero-optimiser",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
) 