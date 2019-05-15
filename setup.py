from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pchaos',
    version='0.1',
    description='Polynomial chaos package for stochastic galerkin and collocation methods of uncertainty quantification',
    long_description=readme,
    author='Komahan Boopathy',
    author_email='komibuddy@gmail.com',
    url='https://github.com/komahanb/pchaos',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

