from distutils.core import setup

setup(
    name='Kaggler',
    version='0.1.1',
    author='Jeong-Yoon Lee',
    author_email='jeongyoon.lee1@gmail.com',
    packages=['kaggler', 'kaggler.test'],
    url='http://pypi.python.org/pypi/Kaggler/',
    license='LICENSE.txt',
    description='Utility functions and common setups for Kaggle competitions.',
    long_description=open('README.txt').read(),
    install_requires=[
        "scipy >= 0.14.0",
        "scikit-learn >= 0.15.0",
        "statsmodels >= 0.5.0",
    ],
)
