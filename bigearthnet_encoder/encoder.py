from pathlib import Path
import bigearthnet_common.constants as ben_constants
from bigearthnet_common.base import get_s2_patch_directories, get_s1_patch_directories
from pydantic import validate_arguments, DirectoryPath
from typing import Dict, Any, Callable, List, Optional
import typer
import lmdb
from rich.progress import track
import fastcore.all as fc

from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch
from bigearthnet_patch_interface.s1_interface import BigEarthNet_S1_Patch

from ._tif_reader import read_ben_tiffs

__all__ = [
    "write_S2_lmdb",
    "write_S1_lmdb",
    "tiff_dir_to_ben_s2_patch",
    "tiff_dir_to_ben_s1_patch",
]


def _tiff_name_to_ben_s2_patch_key(name: str) -> str:
    """
    Receive the tiff's file name, which contains a single band,
    and return the BigEarthNet_S2_Patch key.
    """
    band_name = ben_constants.BEN_S2_BAND_RE.search(name).group("band")
    return BigEarthNet_S2_Patch.short_to_long_band_name(band_name)


def _tiff_name_to_ben_s1_patch_key(name: str) -> str:
    """
    Receive the tiff's file name, which contains a single band,
    and return the BigEarthNet_S2_Patch key.
    """
    band_name = ben_constants.BEN_S1_BAND_RE.search(name).group("band")
    return BigEarthNet_S1_Patch.short_to_long_band_name(band_name)


@validate_arguments
def tiff_dir_to_ben_s2_patch(patch_dir: DirectoryPath, **kwargs):
    name_np_dict = read_ben_tiffs(patch_dir)
    bands_dict = {_tiff_name_to_ben_s2_patch_key(k): v for k, v in name_np_dict.items()}
    return BigEarthNet_S2_Patch(**bands_dict, **kwargs)


@validate_arguments
def tiff_dir_to_ben_s1_patch(patch_dir: DirectoryPath, **kwargs):
    name_np_dict = read_ben_tiffs(patch_dir)
    bands_dict = {_tiff_name_to_ben_s1_patch_key(k): v for k, v in name_np_dict.items()}
    return BigEarthNet_S1_Patch(**bands_dict, **kwargs)


@validate_arguments
def _write_lmdb(
    patch_paths: List[DirectoryPath],
    patch_builder: Callable,
    lmdb_path: Path = Path("S2_lmdb.db"),
    patch_name_to_metadata: Optional[Callable[[str], Dict[str, Any]]] = None,
):
    max_size = 2**40  # 1TebiByte
    env = lmdb.open(str(lmdb_path), map_size=max_size, readonly=False)

    with env.begin(write=True) as txn:
        for patch_path in track(patch_paths, description="Building LMDB archive"):
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
            ben_patch = patch_builder(patch_path, **metadata)
            txn.put(patch_name.encode(), ben_patch.dumps())
    env.close()


@validate_arguments
def write_S2_lmdb(
    ben_s2_path: DirectoryPath,
    lmdb_path: Path = Path("S2_lmdb.db"),
    patch_name_to_metadata: Optional[Callable[[str], Dict[str, Any]]] = None,
):
    patch_paths = get_s2_patch_directories(ben_s2_path)
    _write_lmdb(
        patch_paths,
        tiff_dir_to_ben_s2_patch,
        lmdb_path=lmdb_path,
        patch_name_to_metadata=patch_name_to_metadata,
    )


@validate_arguments
def write_S1_lmdb(
    ben_s1_path: DirectoryPath,
    lmdb_path: Path = Path("S1_lmdb.db"),
    patch_name_to_metadata: Optional[Callable[[str], Dict[str, Any]]] = None,
):
    patch_paths = get_s1_patch_directories(ben_s1_path)
    _write_lmdb(
        patch_paths,
        tiff_dir_to_ben_s1_patch,
        lmdb_path=lmdb_path,
        patch_name_to_metadata=patch_name_to_metadata,
    )


@fc.delegates(write_S1_lmdb, but=["patch_name_to_metadata"])
def write_simple_S1_lmdb(ben_s1_directory_path: Path, **kwargs):
    return write_S1_lmdb(ben_s1_directory_path, **kwargs)


@fc.delegates(write_S2_lmdb, but=["patch_name_to_metadata"])
def write_simple_S2_lmdb(ben_s2_directory_path: Path, **kwargs):
    return write_S2_lmdb(ben_s2_directory_path, **kwargs)


def encoder_cli():
    app = typer.Typer()
    app.command()(write_simple_S1_lmdb)
    app.command()(write_simple_S2_lmdb)
    app()


if __name__ == "__main__":
    encoder_cli()
