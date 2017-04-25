"""OpenTDA: Open-source Python library for topological data analysis"""
import sys
import numpy as np
from os.path import join as pjoin
from setuptools import setup, Extension, find_packages
from tda import __name__, __version__

try:
    import Cython
    from Cython.Distutils import build_ext

    if Cython.__version__ < '0.18':
        raise ImportError()
except ImportError:
    print(
        'Cython version 0.18 or later is required. Try "conda install cython"')
    sys.exit(1)

NAME = __name__
VERSION = __version__


def read(filename):
    import os
    BASE_DIR = os.path.dirname(__file__)
    filename = os.path.join(BASE_DIR, filename)
    with open(filename, 'r') as fi:
        return fi.read()


def readlist(filename):
    rows = read(filename).split("\n")
    rows = [x.strip() for x in rows if x.strip()]
    return list(rows)


extensions = []
extensions.append(
    Extension('tda.snf',
              sources=[pjoin('tda', 'snf.pyx')],
              include_dirs=[np.get_include()]))


setup(
    name=NAME,
    version=VERSION,
    description=('Open-source Python library for topological data analysis (TDA).'),
    author="Brandon B",
    author_email="outlacedev@gmail.com",
    url='https://github.com/outlace/%s' % NAME,
    download_url='https://github.com/outlace/%s/tarball/master' % NAME,
    license='ALv2',
    packages=find_packages(),
    package_data={
        '': ['README.md'],
    },
    zip_safe=False,
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext}
)
