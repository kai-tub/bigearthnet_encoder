from pathlib import Path
from typing import Dict, Optional

import fastcore.all as fc
import natsort
import numpy as np
import rasterio
from pydantic import DirectoryPath, FilePath, validate_arguments

__all__ = ["read_tif", "read_ben_tif", "read_ben_tiffs"]


@validate_arguments
def read_tif(file_path: FilePath, read_channels: Optional[int] = None) -> np.ndarray:
    """
    Path to `tif` file.

    If read_channel is `None` all channels/bands are read from the file.
    Otherwise the value of read_channel is used.
    """
    # https://gitlab.tubit.tu-berlin.de/rsim/bigearthnet-models-tf/blob/master/BigEarthNet.py
    # Needs to convert bands to
    with rasterio.open(file_path) as tif_image:
        num_channels = tif_image.count
        channels = range(1, num_channels + 1)
        read_channels = channels if read_channels is None else read_channels
        data = tif_image.read(read_channels)
    return data


def read_ben_tif(file_path: FilePath) -> np.ndarray:
    """
    A tiny wrapper around `read_tif`, which sets the default
    values to the ones required by BigEarthNet TIF files.
    """
    return read_tif(file_path, read_channels=1)


@validate_arguments
def read_ben_tiffs(folder_path: DirectoryPath) -> Dict[str, np.ndarray]:
    tiffs = folder_path.glob("*.tif")
    # I return in natural order
    tiffs = natsort.natsorted(tiffs, key=lambda p: p.name)
    return {fp.name: read_ben_tif(fp) for fp in tiffs}
