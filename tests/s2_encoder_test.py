import pytest
from bigearthnet_encoder.s2_encoder import *
from pathlib import Path
from pydantic import ValidationError
import pandas as pd
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


def test_patch_name_to_metdata_func_builder():
    df = pd.DataFrame({"name": TEST_FOLDER.name, "labels": ["Marine waters"]})
    with tempfile.NamedTemporaryFile() as tmp_file:
        df.to_parquet(tmp_file.name)
        patch_name_to_metadata_func_builder_from_parquet(parquet_path=tmp_file.name)


def test_patch_name_to_metdata_func_builder():
    df = pd.DataFrame(
        {"name": TEST_FOLDER.name, "labels": ["Marine waters"], "drop_me": ["please"]}
    )
    with tempfile.NamedTemporaryFile() as tmp_file:
        df.to_parquet(tmp_file.name)
        patch_name_to_metadata_func_builder_from_parquet(
            parquet_path=tmp_file.name, drop_cols=["drop_me"]
        )
