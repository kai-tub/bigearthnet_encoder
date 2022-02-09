import lmdb
import pytest
from bigearthnet_common.constants import BEN_S1_RE, BEN_S2_RE
from bigearthnet_encoder.encoder import *
from pathlib import Path
import tempfile
from bigearthnet_patch_interface.s1_interface import BigEarthNet_S1_Patch
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch
from bigearthnet_patch_interface.merged_interface import BigEarthNet_S1_S2_Patch

from bigearthnet_encoder.encoder import (
    write_S1_lmdb_raw,
    write_S2_lmdb_raw,
    write_S1_S2_lmdb_raw,
    write_S2_lmdb,
    write_S1_lmdb,
    write_S1_lmdb_with_lbls,
    write_S2_lmdb_with_lbls,
    write_S1_S2_lmdb_with_lbls,
    _write_lmdb,
)

TEST_S2_ROOT = Path(__file__).parent / "s2-tiny"
TEST_S2_FOLDER = TEST_S2_ROOT / "S2A_MSIL2A_20170617T113321_4_55"
TEST_S2_FILE = TEST_S2_FOLDER / "S2A_MSIL2A_20170617T113321_4_55_B01.tif"
TEST_S2_ENTRIES = 3

TEST_S1_ROOT = Path(__file__).parent / "s1-tiny"
TEST_S1_FOLDER = TEST_S1_ROOT / "S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39"
TEST_S1_FILE = TEST_S1_FOLDER / "S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39_VV.tif"
TEST_S1_ENTRIES = 6


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
