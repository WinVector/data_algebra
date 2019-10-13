# setup.py
import setuptools

DESCRIPTION=\
    "data_algebra is a data manipulation language that can both generate SQL queries and work on Pandas DataFrames. "

LONG_DESCRIPTION = """
[data_algebra](https://github.com/WinVector/data_algebra) is a piped data wrangling system
based on Codd's relational algebra and experience working with dplyr at scale, and a general 
data record transform engine.  The primary purpose of the package is to support an easy to 
compose and maintain grammar of data processing steps that in turn can be used to generate
database specific SQL. The package also implements the same transforms for Pandas DataFrames 
and Dask.

[R](https://www.r-project.org) versions of the system are available as 
the [rquery](https://github.com/WinVector/rquery) and 
[rqdatatable](https://github.com/WinVector/rqdatatable) packages.

Recommended packages include: PyYAML (supplies yaml), sqlparse, black, and dask. 
"""

setuptools.setup(
    name='data_algebra',
    version='0.2.9',
    author='John Mount',
    author_email='jmount@win-vector.com',
    url='https://github.com/WinVector/data_algebra',
    packages=setuptools.find_packages(exclude=['tests', 'Examples']),
    install_requires=[
        "numpy",
        "pandas"
    ],
    platforms=['any'],
    license='License :: OSI Approved :: BSD 3-clause License',
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'License :: OSI Approved :: BSD License',
    ],
    long_description=LONG_DESCRIPTION,
)
