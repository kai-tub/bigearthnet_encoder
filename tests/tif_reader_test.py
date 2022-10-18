from pathlib import Path

import pytest
from bigearthnet_common.example_data import get_s2_example_patch_path
from pydantic import ValidationError

from bigearthnet_encoder._tif_reader import *

TEST_FOLDER = get_s2_example_patch_path()
TEST_FILE = TEST_FOLDER / f"{TEST_FOLDER.name}_B01.tif"


def test_read_tif():
    assert (read_tif(TEST_FILE) == read_tif(TEST_FILE, read_channels=1)).all()


def test_read_ben_tif():
    assert (read_ben_tif(TEST_FILE) == read_tif(TEST_FILE, read_channels=1)).all()


def test_read_ben_tiffs_len():
    tiffs_dict = read_ben_tiffs(TEST_FOLDER)
    assert len(tiffs_dict) == 12


def test_read_ben_tiffs_key():
    tiffs_dict = read_ben_tiffs(TEST_FOLDER)
    assert TEST_FILE.name in tiffs_dict.keys()


def test_read_tif_too_many_channels():
    with pytest.raises(IndexError):
        read_tif(TEST_FILE, read_channels=2)


@pytest.mark.parametrize("invalid_path", [".", TEST_FOLDER])
def test_read_tif_no_file(invalid_path):
    with pytest.raises(ValidationError):
        read_tif(invalid_path)
