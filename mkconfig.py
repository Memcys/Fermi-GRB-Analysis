"""Modify grb/config/path.py as needed.

There are four paths to assign:
- ROOT: the root directory for the repo, absolute path
- FITS: data directory for FITS file, relative to ROOT
- TABLE: data directory for tables ((e)csv, txt, etc.), relative to ROOT
- IMAGE: image directory, relative to ROOT
"""

from pathlib import Path
import logging

# def main():
logging.basicConfig(level=logging.WARNING)

configpath = 'grb/config/path.py'
p = Path()
ROOT = f"Path('{p.resolve()}')"
d = {'FITS': "ROOT / 'data/fits'", 'TABLE': "ROOT / 'data/table'", \
    'IMAGE': "ROOT / 'data/image'"}

print('Current/Working directory (default to ROOT):\n', p.resolve())
for key, value in d.items():
    print(key, '(relative to ROOT):\n', d[key].split("'")[1])

d['ROOT'] = ROOT

while True:
    try:
        key = input("\nPlease type the name (ROOT, FITS, TABLE, or IMAGE) to modify:\n(type 'c' or Enter to install the package: grb)\n")
        if key == 'c' or key == '':
            break
        elif key == 'ROOT':
            ROOT = input("Please assign the ROOT absolute path:\n")
            d[key] = f"Path('{ROOT}')"
            # check if ROOT exists:
            if not eval(d[key]).exists:
                logging.warning(f"{ROOT} not exists!\n")
        elif key in d.keys():
            # key is one of 'FITS', 'TABLES' and 'IMAGES'
            datapath = input("Please assign the data path relative to ROOT (e.g., data/new):\n")
            d[key] = 'ROOT / ' + f"'{datapath}'"
            # check if datapath exists:
            newpath = d['ROOT'] + '/' + datapath
            if not Path(newpath).exists:
                logging.warning(f"{datapath} not exists!\n")
    except EOFError:
        # Handle the EOFError in readthedocs.org
        break
    except Exception as exception:
        # Output unexpected Exceptions.
        logging.warning(exception)
        break

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