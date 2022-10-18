import tempfile
from pathlib import Path

import lmdb
import pytest
from bigearthnet_common.base import s1_to_s2_patch_name
from bigearthnet_common.constants import BEN_S1_RE, BEN_S2_RE
from bigearthnet_common.example_data import (
    get_s1_example_folder_path,
    get_s1_example_patch_path,
    get_s2_example_folder_path,
    get_s2_example_patch_path,
)
from bigearthnet_patch_interface.merged_interface import BigEarthNet_S1_S2_Patch
from bigearthnet_patch_interface.s1_interface import BigEarthNet_S1_Patch
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch

from bigearthnet_encoder.encoder import *
from bigearthnet_encoder.encoder import (
    _write_lmdb,
    write_S1_lmdb,
    write_S1_lmdb_raw,
    write_S1_lmdb_with_lbls,
    write_S1_S2_lmdb_raw,
    write_S1_S2_lmdb_with_lbls,
    write_S2_lmdb,
    write_S2_lmdb_raw,
    write_S2_lmdb_with_lbls,
)

TEST_S2_ROOT = get_s2_example_folder_path()
TEST_S2_FOLDER = get_s2_example_patch_path()
TEST_S2_ENTRIES = len(list(TEST_S2_ROOT.iterdir()))

TEST_S1_ROOT = get_s1_example_folder_path()
TEST_S1_FOLDER = get_s1_example_patch_path()
TEST_S1_ENTRIES = len(list(TEST_S1_ROOT.iterdir()))


def test_s2_tiff_dir_to_ben():
    ben_patch = tiff_dir_to_ben_s2_patch(TEST_S2_FOLDER)
    assert isinstance(ben_patch, BigEarthNet_S2_Patch)


def test_s1_tiff_dir_to_ben():
    ben_patch = tiff_dir_to_ben_s1_patch(TEST_S1_FOLDER)
    assert isinstance(ben_patch, BigEarthNet_S1_Patch)


def test_empty_dir():
    with pytest.raises(ValueError, match="No patches"):
        _write_lmdb([], lambda x: x)


def _lmdb_tester(
    lmdb_path, is_key_sentinel2_dir, expected_class, size, test_attrs=None
):
    key_re_checker = BEN_S2_RE if is_key_sentinel2_dir else BEN_S1_RE
    env = lmdb.open(lmdb_path)
    with env.begin() as txn:
        assert env.stat()["entries"] == size
        for enc_patch_name, binary_patch_data in txn.cursor():
            patch_name = enc_patch_name.decode()
            assert key_re_checker.fullmatch(patch_name)
            patch_data = expected_class.loads(binary_patch_data)
            # just highlighting that the decoding has been succesfull
            assert isinstance(patch_data, expected_class)
            if test_attrs is not None:
                for attr in test_attrs:
                    assert hasattr(patch_data, attr)


def test_write_S2_lmdb_raw():
    with tempfile.TemporaryDirectory() as tmpdir:
        write_S2_lmdb_raw(TEST_S2_ROOT, lmdb_path=tmpdir)
        _lmdb_tester(tmpdir, True, BigEarthNet_S2_Patch, TEST_S2_ENTRIES)


def test_write_S1_lmdb_raw():
    with tempfile.TemporaryDirectory() as tmpdir:
        write_S1_lmdb_raw(TEST_S1_ROOT, lmdb_path=tmpdir)
        _lmdb_tester(tmpdir, False, BigEarthNet_S1_Patch, TEST_S1_ENTRIES)


def test_write_S1_S2_lmdb_raw():
    assert {s2.name for s2 in TEST_S2_ROOT.iterdir()} == {
        s1_to_s2_patch_name(s1.name) for s1 in TEST_S1_ROOT.iterdir()
    }, "Test needs to be updated to filter test-data!"
    with tempfile.TemporaryDirectory() as tmpdir:
        write_S1_S2_lmdb_raw(TEST_S1_ROOT, TEST_S2_ROOT, lmdb_path=tmpdir)
        _lmdb_tester(tmpdir, True, BigEarthNet_S1_S2_Patch, TEST_S2_ENTRIES)


def test_write_S2_lmdb_with_lbls():
    with tempfile.TemporaryDirectory() as tmpdir:
        write_S2_lmdb_with_lbls(TEST_S2_ROOT, lmdb_path=tmpdir)
        _lmdb_tester(
            tmpdir,
            True,
            BigEarthNet_S2_Patch,
            TEST_S2_ENTRIES,
            ["labels", "new_labels"],
        )


# TODO: Find the correctly matching data
def test_write_S1_lmdb_with_lbls():
    with tempfile.TemporaryDirectory() as tmpdir:
        write_S1_lmdb_with_lbls(TEST_S1_ROOT, lmdb_path=tmpdir)
        _lmdb_tester(
            tmpdir,
            False,
            BigEarthNet_S1_Patch,
            TEST_S1_ENTRIES,
            ["labels", "new_labels"],
        )


def test_write_S1_S2_lmdb_with_lbls():
    with tempfile.TemporaryDirectory() as tmpdir:
        write_S1_S2_lmdb_with_lbls(TEST_S1_ROOT, TEST_S2_ROOT, lmdb_path=tmpdir)
        _lmdb_tester(
            tmpdir,
            True,
            BigEarthNet_S1_S2_Patch,
            TEST_S2_ENTRIES,
            ["labels", "new_labels"],
        )
