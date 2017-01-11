from setuptools import setup, Extension
from Cython.Distutils import build_ext

import numpy as np

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(
    name='Kaggler',
    version='0.5.0',

    author='Jeong-Yoon Lee',
    author_email='jeongyoon.lee1@gmail.com',

    packages=['kaggler',
              'kaggler.feature_selection',
              'kaggler.model',
              'kaggler.metrics',
              'kaggler.online_model',
              'kaggler.preprocessing',
              'kaggler.test'],
    url='https://github.com/jeongyoonlee/Kaggler',
    license='LICENSE.txt',

    description='Code for Kaggle Data Science Competitions.',
    long_description=read_md('README.md'),

    install_requires=[
        'cython',
        'h5py',
        'ml_metrics',
        'numpy',
        'pandas',
        'scipy >= 0.14.0',
        'scikit-learn >= 0.15.0',
        'statsmodels >= 0.5.0',
    ],

    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension('kaggler.online_model.ftrl',
                           ['kaggler/online_model/ftrl.pyx',
                            'kaggler/online_model/murmurhash/MurmurHash3.cpp'],
                           libraries=[],
                           include_dirs=[np.get_include(), '.'],
                           extra_compile_args=['-O3']),
                 Extension('kaggler.online_model.sgd',
                           ['kaggler/online_model/sgd.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include(), '.'],
                           extra_compile_args=['-O3']),
                 Extension('kaggler.online_model.fm',
                           ['kaggler/online_model/fm.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include(), '.'],
                           extra_compile_args=['-O3']),
                 Extension('kaggler.online_model.nn',
                           ['kaggler/online_model/nn.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include(), '.'],
                           extra_compile_args=['-O3']),
                 Extension('kaggler.online_model.nn_h2',
                           ['kaggler/online_model/nn_h2.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include(), '.'],
                           extra_compile_args=['-O3']),
                 Extension('kaggler.util',
                           ['kaggler/util.pyx', 'kaggler/util.pxd'],
                           libraries=[],
                           include_dirs=[np.get_include(), '.'],
                           extra_compile_args=['-O3'])],
)
