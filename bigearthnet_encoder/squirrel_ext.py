import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import fastcore.all as fc
import fsspec
import msgpack
import msgpack_numpy
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch
from pydantic import DirectoryPath, validate_arguments
from rich.progress import track
from squirrel.driver.store_driver import StoreDriver
from squirrel.serialization.msgpack import MessagepackSerializer
from squirrel.store import SquirrelStore

from bigearthnet_encoder.encoder import tiff_dir_to_ben_s2_patch


class MyMessagePackDriver(StoreDriver):
    name = "my_messagepack"

    def __init__(self, url: str, **kwargs):
        """Initializes MessagepackDriver with default store and custom serializer."""
        if "store" in kwargs:
            raise ValueError(
                "Store of MessagepackDriver is fixed, `store` cannot be provided."
            )
        super().__init__(store=SquirrelStore(url, MyMessagepackSerializer()), **kwargs)


class MyMessagepackSerializer(MessagepackSerializer):
    """
    Identical to `MessagepackSerializer` with only difference
    that I allow custom compression method!

    To minimize the required custom code, the data will
    still be suffixed with `.gz`, irrespective of the used compression.
    """

    @staticmethod
    def serialize_shard_to_file(shard, fp, fs=None, mode: str = "wb", **open_kwargs):
        # Allows to set custom compression
        print(open_kwargs)
        open_kwargs["mode"] = mode

        if fs is None:
            fs = fsspec

        with fs.open(fp, **open_kwargs) as f:
            for sample in shard:
                f.write(MyMessagepackSerializer.serialize(sample))

    @staticmethod
    def deserialize_shard_from_file(fp: str, fs=None, mode: str = "rb", **open_kwargs):
        open_kwargs["mode"] = mode

        if fs is None:
            fs = fsspec

        with fs.open(fp, **open_kwargs) as f:
            yield from msgpack.Unpacker(f, object_hook=msgpack_numpy.decode)


# Patch names to shards?
# would allow for custom mapping of each patch
# and would allow for 'random' chunking
def _enumerated_chunking(
    patch_names: List[str], chunk_sz: int = 6600, shuffle: bool = True
) -> Dict[str, List[str]]:
    if shuffle:
        random.shuffle(patch_names)
    chunks = fc.chunked(patch_names, chunk_sz=chunk_sz, drop_last=False)
    return {f"{i:05}": list(chunk) for i, chunk in enumerate(chunks)}


def _patch_interface_to_dict(
    s2_patch_interface: BigEarthNet_S2_Patch,
) -> Dict[str, Any]:
    s2_metadata_dict = s2_patch_interface.__stored_args__.copy()
    s2_data_dict = {b.name: b.data for b in s2_patch_interface.bands}
    return {**s2_metadata_dict, **s2_data_dict}


def _write_s2_msgpack(
    patch_paths: List[DirectoryPath], target_path: Path, compression=None
):
    msgpack_driver = MyMessagePackDriver(target_path)
    name_to_path_mapping = {p.name: p for p in patch_paths}
    # FUTURE: allow custom shard_names_mapping function!
    shard_names_mapping = _enumerated_chunking(
        list(name_to_path_mapping.keys()), chunk_sz=6600
    )

    for key_names_tup in track(shard_names_mapping.items()):
        key, names = key_names_tup
        # names to interface
        # interface to dict
        # dict to shard
        shard_data = [
            # FUTURE: Allow custom path_builder func
            tiff_dir_to_ben_s2_patch(name_to_path_mapping[name], patch_name=name)
            for name in names
        ]
        # FUTURE: extend the interface class
        # needs to be converted to dict because mspack doesn't allow
        # arbitrary python classes/objects
        shard_data = [_patch_interface_to_dict(patch) for patch in shard_data]
        msgpack_driver.store.set(shard_data, key=key, compression=compression)
