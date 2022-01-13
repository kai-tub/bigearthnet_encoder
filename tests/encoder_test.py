import pytest
from bigearthnet_encoder.encoder import *
from pathlib import Path
from pydantic import ValidationError
import pandas as pd
import tempfile
from bigearthnet_patch_interface.s1_interface import BigEarthNet_S1_Patch
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch

TEST_S2_ROOT = Path(__file__).parent / "s2-tiny"
TEST_S2_FOLDER = TEST_S2_ROOT / "S2A_MSIL2A_20170617T113321_4_55"
TEST_S2_FILE = TEST_S2_FOLDER / "S2A_MSIL2A_20170617T113321_4_55_B01.tif"

TEST_S1_ROOT = Path(__file__).parent / "s1-tiny"
TEST_S1_FOLDER = TEST_S1_ROOT / "S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39"
TEST_S1_FILE = TEST_S1_FOLDER / "S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39_VV.tif"


def test_s2_tiff_dir_to_ben():
    ben_patch = tiff_dir_to_ben_s2_patch(TEST_S2_FOLDER)
    assert isinstance(ben_patch, BigEarthNet_S2_Patch)


def test_s1_tiff_dir_to_ben():
    ben_patch = tiff_dir_to_ben_s1_patch(TEST_S1_FOLDER)
    assert isinstance(ben_patch, BigEarthNet_S1_Patch)


def test_write_S2_lmdb():
    with tempfile.TemporaryDirectory() as tmpdir:
        write_S2_lmdb(TEST_S2_ROOT, tmpdir)


def test_write_S1_lmdb():
    with tempfile.TemporaryDirectory() as tmpdir:
        write_S1_lmdb(TEST_S1_ROOT, tmpdir)
