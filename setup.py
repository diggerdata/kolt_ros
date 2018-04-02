#!/usr/bin/env python3

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['core'],
    package_dir={'': 'src'},
    install_requires=[
        'numpy', 
        'scipy',
        'scikit-learn',
        'keras',
        'opencv-python',
        'h5py',
        'imgaug',
        'tensorflow-gpu'
    ],
)

setup(**d)