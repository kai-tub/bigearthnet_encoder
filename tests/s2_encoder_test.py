import pytest
from bigearthnet_encoder.s2_encoder import *
from pathlib import Path
from pydantic import ValidationError
import tempfile

TEST_ROOT = Path(__file__).parent / "tiny"
TEST_FOLDER = TEST_ROOT / "S2A_MSIL2A_20170617T113321_4_55"
TEST_FILE = TEST_FOLDER / "S2A_MSIL2A_20170617T113321_4_55_B01.tif"
# TODO


def test_tiff_dir_to_ben():
    ben_patch = tiff_dir_to_ben_s2_patch(TEST_FOLDER)
    # TODO: Check isinsatane


def test_write_lmdb():
    with tempfile.TemporaryDirectory() as tmpdir:
        write_S2_lmdb(TEST_ROOT, tmpdir)
