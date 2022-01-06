import re
from typing import Optional
from pathlib import Path
from bigearthnet_common.gdf_builder import get_patch_directories
from pydantic import validate_arguments, DirectoryPath, FilePath
from typing import Dict, Any, Callable, List
import pandas as pd
import lmdb

from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch

from ._tif_reader import read_ben_tiffs

__all__ = [
    "patch_name_to_metadata_func_builder_from_parquet",
    "write_S2_lmdb",
    "tiff_dir_to_ben_s2_patch",
]

BEN_S2_BAND_RE = re.compile(r".*(?P<band>B\d[0-9A])")


def _tiff_name_to_ben_patch_key(name: str) -> str:
    """
    Receive the tiff's file name, which contains a single band,
    and return the BigEarthNet_S2_Patch key.
    """
    band_name = BEN_S2_BAND_RE.search(name).group("band")
    return BigEarthNet_S2_Patch.short_to_long_band_name(band_name)


@validate_arguments
def tiff_dir_to_ben_s2_patch(patch_dir: DirectoryPath, **kwargs):
    name_np_dict = read_ben_tiffs(patch_dir)
    bands_dict = {_tiff_name_to_ben_patch_key(k): v for k, v in name_np_dict.items()}
    return BigEarthNet_S2_Patch(**bands_dict, **kwargs)


@validate_arguments
def patch_name_to_metadata_func_builder_from_parquet(
    parquet_path: FilePath,
    patch_name_col: str = "name",
    drop_cols: Optional[List[str]] = None,
):
    """
    Read from parquet and not directly from dataframe for two reasons:

    1. Enforce that the data is stored on disk to ensure reproducible builds
    2. There is often an encoding-difference between raw dataframes and the loaded parquet dataframes
        - For example, lists are encoded as np.ndarray of dtype='object' when stored as parquet
    """
    df = pd.read_parquet(parquet_path)
    indexed_df = df.set_index(patch_name_col).drop(drop_cols, axis=1)

    def patch_name_to_metdata_func(patch_name):
        return indexed_df.loc[patch_name].to_dict()

    return patch_name_to_metdata_func


@validate_arguments
def write_S2_lmdb(
    ben_s2_path: DirectoryPath,
    lmdb_path: Path = Path("S2_lmdb.db"),
    patch_name_to_metadata: Optional[Callable[[str], Dict[str, Any]]] = None,
):
    max_size = 2 ** 40  # 1TebiByte
    patch_paths = get_patch_directories(ben_s2_path)
    env = lmdb.open(str(lmdb_path), map_size=max_size, readonly=False)

    with env.begin(write=True) as txn:
        for patch_path in patch_paths:
            patch_name = patch_path.name
            if patch_name_to_metadata is not None:
                metadata = patch_name_to_metadata(patch_name)
                if not isinstance(metadata, dict):
                    raise TypeError(
                        "name to metadata converter has returned a wrong type!",
                        metadata,
                    )
            else:
                metadata = {}
            ben_patch = tiff_dir_to_ben_s2_patch(patch_path, **metadata)
            txn.put(patch_name.encode(), ben_patch.dumps())
    env.close()
