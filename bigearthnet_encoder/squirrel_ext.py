import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import fastcore.all as fc
import fsspec
import msgpack
import msgpack_numpy
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch
from pydantic import DirectoryPath, validate_arguments
from squirrel.driver.store_driver import StoreDriver
from squirrel.store.filesystem import get_random_key
from squirrel.serialization.msgpack import MessagepackSerializer

import msgpack
import msgpack_numpy
from squirrel.driver.store_driver import StoreDriver
from squirrel.serialization.msgpack import MessagepackSerializer

from enum import Enum

class Split(str, Enum):
    train = "train"
    validation = "validation"
    test = "test"


# When writing the dataset, it would be easiest
# to ensure that the dataset has the desired output format
class ConfigurableMessagePackDriver(StoreDriver):
    name = "configurable_messagepack"

    def __init__(self, url: str, **kwargs):
        """Initializes MessagepackDriver with default store and custom serializer."""
        if "store" in kwargs:
            raise ValueError(
                "Store of MessagepackDriver is fixed, `store` cannot be provided."
            )
        # This happens in top-super and nothing else:
        # self._catalog = catalog if catalog is not None else Catalog()
        # and
        # self._store = store is set
        super().__init__(url, ConfigurableMessagepackSerializer, **kwargs)

    # get_iter of super calls nothing else than self.get()
    # here it would use a sharded access store
    # Now, I am thinking that the SquirrelStore should be updated to include the split variable
    # Then the get/set methods of the SquirrelStore could include the split name
    # and auto prepend it... But where? And how?
    # I would have to write a custom get/set method that calls the super variant
    # Or... I could simply name the keys depending on the split
    # I think this would be the easiest approach
    def get_iter(
        self,
        split: Optional[Split] = None,
        **kwargs,
    ):
        """
        Returns an iterable of items in the form of a :py:class:`squirrel.iterstream.Composable`, which allows
        various stream manipulation functionalities.

        Items are fetched using the :py:meth:`get` method. The returned :py:class:`Composable` iterates over the items
        in the order of the keys returned by the :py:meth:`keys` method.

        Args:
            flatten (bool): Whether to flatten the returned iterable. Defaults to True.
            **kwargs: Other keyword arguments passed to `super().get_iter()`. For details, see
                :py:meth:`squirrel.driver.MapDriver.get_iter`.

        Returns:
            (squirrel.iterstream.Composable) Iterable over the items in the store.

        See map-driver.get_iter!
        """
        keys = super().keys()
        if split is not None:
            keys = [k for k in keys if k.startswith(split)]
        return super().get_iter(keys_iterable=keys, **kwargs)


class ConfigurableMessagepackSerializer(MessagepackSerializer):
    """
    Identical to `MessagepackSerializer` with only difference
    that I allow custom compression method!

    To minimize the required custom code, the data will
    still be suffixed with `.gz`, irrespective of the used compression.
    """

    @staticmethod
    def serialize_shard_to_file(shard, fp, fs=None, mode: str = "wb", **open_kwargs):
        # Allows to set custom compression
        open_kwargs["mode"] = mode

        if fs is None:
            fs = fsspec

        with fs.open(fp, **open_kwargs) as f:
            for sample in shard:
                f.write(ConfigurableMessagepackSerializer.serialize(sample))

    @staticmethod
    def deserialize_shard_from_file(fp: str, fs=None, mode: str = "rb", max_buffer_size=0, **open_kwargs):
        """
        Reset to max_buffer_size=0 as we trust our dataset source and out-of-buffer errors are not easy do debug for our users.
        """
        open_kwargs["mode"] = mode

        if fs is None:
            fs = fsspec

        with fs.open(fp, **open_kwargs) as f:
            yield from msgpack.Unpacker(f, object_hook=msgpack_numpy.decode, max_buffer_size=max_buffer_size)


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

# call _write_s2_msgpack internally from dataset split
def _prefixed_key(prefix: Optional[str] = None):
    random_suffix = get_random_key()
    prepend = f"{prefix}_"if prefix is not None else ""
    return f"{prepend}{random_suffix}"

def _write_prefixed_shard(data: List[Dict[str, Any]], msgpack_driver, prefix: Optional[None] = None, **kwargs):
    # needs to be strongly coupled to select what should be combined for hash value
    # names = [patch_dict["patch_name"] for patch_dict in data]
    # it would be better to hash the entire output!
    # So instead it would be better to re-write the output
    return msgpack_driver.store.set(data, key=_prefixed_key(prefix), **kwargs)
