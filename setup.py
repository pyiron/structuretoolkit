"""
Setuptools based setup module
"""
from setuptools import setup, find_packages
import versioneer

setup(
    name='structuretoolkit',
    version=versioneer.get_version(),
    description='structuretoolkit - to analyse, build and visualise atomistic structures.',
    long_description='http://pyiron.org',

    url='https://github.com/pyiron/structuretoolkit',
    author='Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department',
    author_email='janssen@mpie.de',
    license='BSD',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],

    keywords='pyiron',
    packages=find_packages(exclude=["*tests*", "*docs*", "*binder*", "*conda*", "*notebooks*", "*.ci_support*"]),
    install_requires=[
        'aimsgb==0.1.1',
        'ase==3.22.1',
        'matplotlib==3.7.0',
        'numpy==1.24.2',
        'phonopy==2.17.1',
        'pymatgen==2022.11.7',
        'scipy==1.10.0',
        'scikit-learn==1.2.1',
        'spglib==2.0.2',
    ],
    cmdclass=versioneer.get_cmdclass(),
)
