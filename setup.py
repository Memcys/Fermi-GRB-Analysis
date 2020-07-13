from setuptools import setup, find_namespace_packages
from pathlib import Path
import fileinput
import logging
# from pkg.config import path

logging.basicConfig(level=logging.WARNING)

configpath = 'pkg/config/path.py'
p = Path()
# path.ROOT = p.resolve()
ROOT = f"Path('{p.resolve()}')"
d = {'FITS': "ROOT / 'data/fits'", 'TABLE': "ROOT / 'data/table'", \
    'IMAGE': "ROOT / 'data/image'"}
# macros = ['FITS', 'TABLE', 'IMAGE']
# varlist = ['path.' + n + f'.relative_to(Path("{p}"))' for n in macros]

print('Current/Working directory (default to ROOT):\n', p.resolve())
for key, value in d.items():
    # pathstr = 'path.' + key + f'.relative_to(Path("{path.ROOT}"))'
    # d[key] = f"'{eval(pathstr)}'"
    print(key, '(relative to ROOT):\n', d[key].split("'")[1])

d['ROOT'] = ROOT

while True:
    v = input("\nPlease type the name (ROOT, FITS, TABLE, or IMAGE) to modify:\n(type c to finish and continue)\n")
    if v == 'c':
        break
    elif v == 'ROOT':
        ROOT = input("Please assign the ROOT absolute path:\n")
        d[v] = f"Path('{ROOT}')"
        # check if ROOT exists:
        if not eval(d[v]).exists:
            logging.warning(f"{ROOT} not exists!\n")
    elif key in d.keys:
        # key is one of 'FITS', 'TABLES' and 'IMAGES'
        datapath = input("Please assign the data path relative to ROOT (e.g., data/new):\n")
        d[key] = 'ROOT / ' + f"'{datapath}'"
        # check if datapath exists:
        newpath = d['ROOT'] + '/' + datapath
        if not Path(newpath).exists:
            logging.warning(f"{datapath} not exists!\n")

pathlist = list(d.keys())
# read all lines
with open(configpath) as c:
    lines = c.readlines()

# do substitution
for i, line in enumerate(lines):
    # ! The walrus operator := is a new feature in Python 3.8
    if len(pathlist) > 0 and line.startswith(key := pathlist[0]):
        newline = f'{key} = {d[key]}\n'
        lines[i] = newline
        pathlist.pop(0)

# make change(s) to file
with open(configpath, 'w') as c:
    c.writelines(lines)

# prepared to install
with open('README.md', 'r') as ld:
    long_description = ld.read()

setup(
    name='grb',

    version = '0.7.13',  # July 13, 2020

    description='Python package for certain Fermi LAT data analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Memcys/Fermi-GRB-Analysis',
    
    author='Memcys',

    # Following https://packaging.python.org/guides/packaging-namespace-packages/#native-namespace-packages
    packages=find_namespace_packages(include=['pkg.*']),

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
        'tqdm',
    ]
)