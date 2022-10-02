"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/guides.html?#writers

Replace code below according to your needs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union
from skm_pyutils.plot import GridFig
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import ceil
from pathlib import Path

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = Tuple[DataType, dict, str]


def write_single_image(path: str, data: Any, meta: dict) -> List[str]:
    """Writes a single image layer"""
    print(meta)

    mpl.rcParams["figure.subplot.left"] = 0.08
    mpl.rcParams["figure.subplot.right"] = 0.92
    mpl.rcParams["figure.subplot.bottom"] = 0.1
    mpl.rcParams["figure.subplot.top"] = 0.9

    num_images = data.shape[0]

    if num_images == 4:
        rows = 2
        cols = 2
    else:
        rows = ceil(num_images / 3)
        cols = min(num_images, 3)

    gf = GridFig(
        rows=rows,
        cols=cols,
        size_multiplier_x=2,
        size_multiplier_y=2,
        wspace=0.12,
        hspace=0.12,
        tight_layout=True,
    )
    for d in data:
        ax = gf.get_next()
        plt.axis("off")
        ax.imshow(d)

    gf.savefig(path, dpi=300)
    plt.close(gf.get_fig())

    return [path]
