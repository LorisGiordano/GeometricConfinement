from setuptools import setup, find_packages

setup(
    name='dataset',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'SimpleITK',
        'numpy',
        'matplotlib',
        'monai',
        'glob',
        'zipfile',
        'json'
    ],
)