from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pspace',
    version='0.1',
    description='Probabilistic space for stochastic galerkin and collocation methods of uncertainty quantification',
    long_description=readme,
    author='Komahan Boopathy',
    author_email='komibuddy@gmail.com',
    url='https://github.com/komahanb/pspace',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'examples'))
)

