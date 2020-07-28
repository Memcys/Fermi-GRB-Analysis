from setuptools import setup, find_namespace_packages
import mkconfig

with open('README.md', 'r') as ld:
    long_description = ld.read()

setup(
    name='grb',

    version = '0.7.28',  # July 28, 2020

    description='Python package for certain Fermi LAT data analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Memcys/Fermi-GRB-Analysis',
    
    author='Memcys',

    # Following https://packaging.python.org/guides/packaging-namespace-packages/#native-namespace-packages
    packages=find_namespace_packages(include=['grb.*']),

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
    python_requires='>=3.8',
    install_requires=[
        'astropy',
        'astroquery',
        'numba',
        'numpy',
        'matplotlib',
        'pandas',
        'retry',
        'scipy',
        'seaborn',
        'tables',
        'tqdm',
    ],
    # See Setting the zip_safe flag: https://setuptools.readthedocs.io/en/latest/setuptools.html#id31
    zip_safe=False,
)