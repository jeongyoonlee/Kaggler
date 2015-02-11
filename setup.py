from setuptools import setup, Extension
from Cython.Distutils import build_ext

import numpy as np


setup(
    name='Kaggler',
    version='0.3.3',

    author='Jeong-Yoon Lee',
    author_email='jeongyoon.lee1@gmail.com',

    packages=['kaggler',
              'kaggler.model',
              'kaggler.online_model',
              'kaggler.test'],
    url='https://github.com/jeongyoonlee/Kaggler',
    license='LICENSE.txt',

    description='Code for Kaggle Data Science Competitions.',
    long_description=open('README.md').read(),

    install_requires=[
        'cython',
        'numpy',
        'scipy >= 0.14.0',
        'scikit-learn >= 0.15.0',
        'statsmodels >= 0.5.0',
    ],

    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension('kaggler.online_model.ftrl',
                           ['kaggler/online_model/ftrl.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include()],
                           extra_compile_args=['-O3']),
                 Extension('kaggler.online_model.sgd',
                           ['kaggler/online_model/sgd.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include()],
                           extra_compile_args=['-O3']),
                 Extension('kaggler.online_model.fm',
                           ['kaggler/online_model/fm.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include()],
                           extra_compile_args=['-O3']),
                 Extension('kaggler.online_model.nn',
                           ['kaggler/online_model/nn.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include()],
                           extra_compile_args=['-O3']),
                 Extension('kaggler.online_model.nn_h2',
                           ['kaggler/online_model/nn_h2.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include()],
                           extra_compile_args=['-O3']),
                 Extension('kaggler.util',
                           ['kaggler/util.pyx'],
                           libraries=[],
                           include_dirs=[np.get_include()],
                           extra_compile_args=['-O3'])],
)
