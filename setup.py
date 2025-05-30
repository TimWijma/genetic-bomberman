from setuptools import setup, find_packages

setup(
    name="genetic",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pommerman",
        "numpy",
        "gym",
    ],
    description="A genetic algorithm implementation for Pommerman agents",
    author="Tim Wijma",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)