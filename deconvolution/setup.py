from setuptools import setup, find_packages

setup(
    name="deconvolution",
    version="0.1",
    packages=find_packages(),
    description="A package for deconvolving partitioning delays",
    author="Demetrios Pagonis",
    author_email="demetriospagonis@weber.edu",
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'pandas',
        'numba'
    ]

)