# configurations for directory paths
# Please assign YOUR paths (on or before installing pkg)
from pathlib import Path

# root directory
ROOT = Path('/Data/Repos/Fermi-GRB-Analysis')

# FITS data
FITS = ROOT / 'data/fits'

# tables, i.e., .(e)csv, .txt, .h5
TABLE = ROOT / 'data/table'

# images, e.g., .pdf, .jpg
IMAGE = ROOT / 'data/image'
