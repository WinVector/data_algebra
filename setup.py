# setup.py
import setuptools

DESCRIPTION = """
data_algebra ( https://github.com/WinVector/data_algebra ) is a piped data wrangling system
based on Codd's relational algebra and experience working with dplyr at scale.  The primary 
purpose of the package is to support an easy to compose and maintain grammar of data processing
steps that in turn can be used to generate database specific SQL.  The package also is intended
to implement the same transforms for Pandas DataFrames. 

This package is still under initial development, so some parts are not yet implemented or tested, and APIs
are subject to change.

Recommended packages include: Pandas, PyYAML (supplies yaml), sqlparse, and black. 
"""

setuptools.setup(
    name='data_algebra',
    version='0.1.1',
    author='John Mount',
    author_email='jmount@win-vector.com',
    url='https://github.com/WinVector/data_algebra',
    packages=setuptools.find_packages(exclude=['tests', 'Examples']),
    install_requires=[
    ],
    platforms=['any'],
    license='License :: OSI Approved :: BSD 3-clause License',
    description=DESCRIPTION,
    classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'License :: OSI Approved :: BSD License',
    ],
)
