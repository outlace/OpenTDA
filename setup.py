"""OpenTDA: Open-source Python library for topological data analysis"""
from setuptools import setup, find_packages
from tda import __name__, __version__

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
        '': ['README.md',
             'requirements.txt'],
    },
    zip_safe=False,
    install_requires=readlist('requirements.txt'),
)
