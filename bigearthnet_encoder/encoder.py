from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import bigearthnet_common.constants as ben_constants
import fastcore.all as fc
import lmdb
import typer
from bigearthnet_common.base import (
    get_s1_patch_directories,
    get_s2_patch_directories,
    s2_to_s1_patch_name,
)
from bigearthnet_patch_interface.merged_interface import BigEarthNet_S1_S2_Patch
from bigearthnet_patch_interface.s1_interface import BigEarthNet_S1_Patch
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch
from pydantic import DirectoryPath, validate_arguments
from rich.progress import Progress

from ._tif_reader import read_ben_tiffs
from .metadata_utils import load_labels_from_patch_path


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
def tiff_dir_to_ben_s2_patch(
    patch_dir: DirectoryPath, **kwargs
) -> BigEarthNet_S2_Patch:
    """
    Process a Sentinel-2 directory path and create a `BigEarthNet_S2_Patch`
    The keyword arguments will be passed in as extra metadata to the patch class.
    """
    name_np_dict = read_ben_tiffs(patch_dir)
    bands_dict = {_tiff_name_to_ben_s2_patch_key(k): v for k, v in name_np_dict.items()}
    return BigEarthNet_S2_Patch(**bands_dict, **kwargs)


@validate_arguments
def tiff_dir_to_ben_s1_patch(
    patch_dir: DirectoryPath, **kwargs
) -> BigEarthNet_S1_Patch:
    """
    Process a Sentinel-1 directory path and create a `BigEarthNet_S1_Patch`.
    The keyword arguments will be passed in as extra metadata to the patch class.
    """
    name_np_dict = read_ben_tiffs(patch_dir)
    bands_dict = {_tiff_name_to_ben_s1_patch_key(k): v for k, v in name_np_dict.items()}
    return BigEarthNet_S1_Patch(**bands_dict, **kwargs)


@validate_arguments
def build_tiff_dirs_to_ben_s1_s2_patch_func(ben_s1_path: DirectoryPath) -> Callable:
    """
    Given the path to the Sentinel-1 root directory, return a function that
    builds a `BigEarthNet_S1_S2_Path` given a Sentinel-2 path.

    This builder ensures that the same Sentinel-1 root path is used for all future
    calls of the resulting function.
    """

    @validate_arguments
    def tiff_dirs_to_ben_s1_s2_patch(
        s2_patch_dir: DirectoryPath, **kwargs
    ) -> BigEarthNet_S1_S2_Patch:
        s1_patch_name = s2_to_s1_patch_name(s2_patch_dir.name)
        s1_patch_dir = ben_s1_path / s1_patch_name

        s1_names_np_dict = read_ben_tiffs(s1_patch_dir)
        s2_names_np_dict = read_ben_tiffs(s2_patch_dir)
        s1_bands_dict = {
            _tiff_name_to_ben_s1_patch_key(k): v for k, v in s1_names_np_dict.items()
        }
        s2_bands_dict = {
            _tiff_name_to_ben_s2_patch_key(k): v for k, v in s2_names_np_dict.items()
        }
        return BigEarthNet_S1_S2_Patch(**s1_bands_dict, **s2_bands_dict, **kwargs)

    return tiff_dirs_to_ben_s1_s2_patch


@validate_arguments
def _write_lmdb(
    patch_paths: List[DirectoryPath],
    patch_builder: Callable,
    lmdb_path: Path = Path("S2_lmdb.db"),
    patch_path_to_metadata: Optional[Callable[[DirectoryPath], Dict[str, Any]]] = None,
    chunk_size: int = 50_000,
) -> None:
    """
    The function writes an LMDB archive.
    A list of patch paths is processed by calling the `patch_builder` individually.
    The output archive is written to `lmdb_path`.
    A `patch_path_to_metadata` function can be provided that returns a metadata dictionary,
    which will be embedded into the `patch_builder` call.
    The `chunk_size` ensure that after processing `chunk_size` patches the output
    is written to disk and the memory is freed.
    If the process fails due to memory issues, decrease the `chunk_size`!
    """
    if len(patch_paths) == 0:
        raise ValueError(
            "No patches were provided! Maybe provided wrong Sentinel directory?"
        )
    max_size = 2**40  # 1TebiByte
    env = lmdb.open(str(lmdb_path), map_size=max_size, readonly=False)

    chunks = fc.chunked(patch_paths, chunk_sz=chunk_size, drop_last=False)
    with Progress() as progress:
        task = progress.add_task("Building LMDB archive", total=len(patch_paths))
        for chunk in chunks:
            with env.begin(write=True) as txn:
                for patch_path in chunk:
                    patch_name = patch_path.name
                    if patch_path_to_metadata is not None:
                        metadata = patch_path_to_metadata(patch_path)
                        if not isinstance(metadata, dict):
                            raise TypeError(
                                "name to metadata converter has returned a wrong type!",
                                metadata,
                            )
                    else:
                        metadata = {}
                    ben_patch = patch_builder(patch_path, **metadata)
                    txn.put(patch_name.encode(), ben_patch.dumps())
                    progress.update(task, advance=1)
        env.close()


@fc.delegates(_write_lmdb, but=["lmdb_path"])
def write_S2_lmdb(
    ben_s2_path: DirectoryPath,
    /,
    lmdb_path: Path = Path("S2_lmdb.db"),
    **kwargs,
) -> None:
    patch_paths = get_s2_patch_directories(ben_s2_path)
    _write_lmdb(patch_paths, tiff_dir_to_ben_s2_patch, lmdb_path=lmdb_path, **kwargs)


@fc.delegates(_write_lmdb, but=["lmdb_path"])
def write_S1_lmdb(
    ben_s1_path: DirectoryPath, /, lmdb_path: Path = Path("S1_lmdb.db"), **kwargs
) -> None:
    patch_paths = get_s1_patch_directories(ben_s1_path)
    _write_lmdb(
        patch_paths,
        tiff_dir_to_ben_s1_patch,
        lmdb_path=lmdb_path,
        **kwargs,
    )


@fc.delegates(_write_lmdb, but=["lmdb_path"])
def write_S1_S2_lmdb(
    ben_s1_path: DirectoryPath,
    ben_s2_path: DirectoryPath,
    /,
    lmdb_path: Path = Path("S2_lmdb.db"),
    **kwargs,
) -> None:
    patch_paths_s2 = get_s2_patch_directories(ben_s2_path)
    _write_lmdb(
        patch_paths_s2,
        build_tiff_dirs_to_ben_s1_s2_patch_func(ben_s1_path),
        lmdb_path=lmdb_path,
        **kwargs,
    )


@fc.delegates(write_S1_lmdb, but=["patch_path_to_metadata"])
def write_S1_lmdb_raw(ben_s1_directory_path: Path, **kwargs) -> None:
    """
    Write an S1 lmdb file that only includes the patch name
    as the key and the patch array information as the value.

    If the process fails due to memory issues, decrease the `chunk_size`!
    """
    return write_S1_lmdb(ben_s1_directory_path, **kwargs)


@fc.delegates(write_S2_lmdb, but=["patch_path_to_metadata"])
def write_S2_lmdb_raw(ben_s2_directory_path: Path, **kwargs) -> None:
    """
    Write an S2 lmdb file that only includes the patch name
    as the key and the patch array information as the value.

    If the process fails due to memory issues, decrease the `chunk_size`!
    """
    return write_S2_lmdb(ben_s2_directory_path, **kwargs)


@fc.delegates(write_S2_lmdb, but=["patch_path_to_metadata"])
def write_S1_S2_lmdb_raw(
    ben_s1_directory_path: Path, ben_s2_directory_path: Path, **kwargs
) -> None:
    """
    Write a combined S1 and S2 lmdb file that only includes the patch name
    as the key and the patch array information as the value.

    If the process fails due to memory issues, decrease the `chunk_size`!
    """
    return write_S1_S2_lmdb(ben_s1_directory_path, ben_s2_directory_path, **kwargs)


@fc.delegates(write_S1_lmdb, but=["patch_path_to_metadata"])
def write_S1_lmdb_with_lbls(ben_s1_directory_path: Path, **kwargs) -> None:
    """
    Write an S1 lmdb file that includes the patch name
    as the key and the patch array information, as well as the original and new label data as the value.

    If the process fails due to memory issues, decrease the `chunk_size`!
    """
    load_lbl_func = fc.partialler(load_labels_from_patch_path, is_sentinel2=False)
    return write_S1_lmdb(
        ben_s1_directory_path, patch_path_to_metadata=load_lbl_func, **kwargs
    )


@fc.delegates(write_S2_lmdb, but=["patch_path_to_metadata"])
def write_S2_lmdb_with_lbls(ben_s2_directory_path: Path, **kwargs) -> None:
    """
    Write an S2 lmdb file that includes the patch name
    as the key and the patch array information, as well as the original and new label data as the value.

    If the process fails due to memory issues, decrease the `chunk_size`!
    """
    load_lbl_func = fc.partialler(load_labels_from_patch_path, is_sentinel2=True)
    return write_S2_lmdb(
        ben_s2_directory_path, patch_path_to_metadata=load_lbl_func, **kwargs
    )


@fc.delegates(write_S2_lmdb, but=["patch_path_to_metadata"])
def write_S1_S2_lmdb_with_lbls(
    ben_s1_directory_path: Path, ben_s2_directory_path: Path, **kwargs
) -> None:
    """
    Write a combined S1 and S2 lmdb file that includes the patch name
    as the key and the patch array information, as well as the original and new label data as the value.

    If the process fails due to memory issues, decrease the `chunk_size`!
    """
    load_lbl_func = fc.partialler(load_labels_from_patch_path, is_sentinel2=True)
    return write_S1_S2_lmdb(
        ben_s1_directory_path,
        ben_s2_directory_path,
        patch_path_to_metadata=load_lbl_func,
        **kwargs,
    )


def encoder_cli() -> None:
    app = typer.Typer()
    app.command()(write_S1_lmdb_raw)
    app.command()(write_S2_lmdb_raw)
    app.command()(write_S1_S2_lmdb_raw)
    app.command()(write_S1_lmdb_with_lbls)
    app.command()(write_S2_lmdb_with_lbls)
    app.command()(write_S1_S2_lmdb_with_lbls)
    app()


if __name__ == "__main__":
    encoder_cli()
