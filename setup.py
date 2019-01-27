from setuptools import setup

"""
Description of how to make a python package

https://python-packaging.readthedocs.io/en/latest/command-line-scripts.html

"""


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='deepunion',
      version='0.1',
      long_description=readme(),
      description='A collection of drug discovery tools',
      url='https://github.com/zhenglz/deepunion',
      author='zhenglz',
      author_email='zhenglz@outlook.com',
      license='GPL-3.0',
      packages=['deepunion'],
      install_requires=[
          'numpy',
          'pandas',
          'pubchempy',
          'mdtraj',
#          'rdkit',
      ],
      include_package_data=True,
      zip_safe=False,
      python_requires='>=3.5',
      )
