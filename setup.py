import pathlib
from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install
import os

# current directory
HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()
DESC = ('A pytorch library that provides routines for translating, building, and'
        'optimizing procedural materials')

REQUIRES_PYTHON = '>=3.6.0'
VERSION = (HERE / "version.txt").read_text().strip()

# run setup
setup(
    name='diffmat',
    version=VERSION,
    description=DESC,
    long_description=README,
    long_description_content_type="text/markdown",
    author='Liang Shi; Beichen Li',
    author_email='liangs@mit.edu; beichen@mit.edu',
    python_requires=REQUIRES_PYTHON,
    url="https://github.com/mit-gfx/diffmat-legacy",
    keywords='procedural material, differentiable graph, SVBRDF, gradient descent, optimization',
    packages=find_packages(),
    install_requires=[
        'configargparse>=1.2.3',
        'torch>=1.7.0',
        'torchvision>=0.8.2',
        'torchsummary>=1.5.1',
        'numpy>=1.18.5',
        'matplotlib>=3.3.2',
        'scipy>=1.4.1',
        'ordered-set>=3.1.1',
        'imageio>=2.5.0',
        'gputil>=1.4.0',
        ],
    include_package_data=True,
    license='Massachusetts Institute of Technology Research License',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Topic :: Multimedia :: Graphics',
        'Intended Audience :: Product/Research',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    entry_points={
        'console_scripts': [
            'sbs_parse=diffmat.sbs_converter.sbs_parser:main',
        ],
    },
)