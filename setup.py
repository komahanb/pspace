from setuptools import setup, find_packages

setup(
    name='pspace',
    version='2.0.0',
    description='Polynomial Chaos Expansion library for uncertainty quantification',
    author='Komahan Boopathy',
    author_email='komibuddy@gmail.com',
    packages=find_packages(exclude=['tests*', 'demos*']),
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'scipy',
        'sympy',
    ],
)
