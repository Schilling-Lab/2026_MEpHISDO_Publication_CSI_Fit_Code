"""
Most of the utils functions are used for plotting inside jupyter-notebooks.

Available functions:

1. plot_colorbar

2. plot_image_series

3. plot_4d
"""

from mpl_toolkits.axes_grid1 import make_axes_locatable


common_cmaps = [
    "bone",  # Good general-purpose grayscale
    "gray",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "viridis",
    "jet",
    "hsv",
    "cool",
    "seismic",
    "twilight",
    "tab10",
    "prism",
    "gist_rainbow",
    "nipy_spectral",  # Perceptually uniform rainbow
    "cmc.lipari",  # Crameri's Lipari colormap
    "cmc.navia",  # Crameri's Navia colormap
]
