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
        'ase==3.22.1',
        'matplotlib==3.7.2',  # ase already requires matplotlib
        'numpy==1.24.3',  # ase already requires numpy
        'scipy==1.11.1',  # ase already requires scipy
    ],
    extras_require={
        "grainboundary": ['aimsgb==1.0.1', 'pymatgen==2023.5.31'],
        "pyscal": ['pyscal2==2.10.18'],
        "nglview": ['nglview==3.0.5'],
        "plotly": ['plotly==5.15.0'],
        "clusters": ['scikit-learn==1.3.0'],
        "symmetry": ['spglib==2.0.2'],
        "surface": ['spglib==2.0.2', 'pymatgen==2023.5.31'],
        "phonopy": ['phonopy==2.20.0', 'spglib==2.0.2'],
    },
    cmdclass=versioneer.get_cmdclass(),
)
