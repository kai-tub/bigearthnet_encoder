import tempfile
from pathlib import Path

from bigearthnet_common.example_data import (
    get_s1_example_folder_path,
    get_s1_example_patch_path,
    get_s2_example_folder_path,
    get_s2_example_patch_path,
)

from bigearthnet_encoder.metadata_utils import *

TEST_S2_ROOT = get_s2_example_folder_path()
# inspected value to manually check label information
TEST_S2_FOLDER = TEST_S2_ROOT / "S2A_MSIL2A_20170617T113321_4_55"
assert TEST_S2_FOLDER.exists()
TEST_S1_ROOT = get_s1_example_folder_path()
TEST_S1_FOLDER = TEST_S1_ROOT / "S1A_IW_GRDH_1SDV_20170617T064724_29UPU_4_55"


def test_patch_path_to_metdata_func_builder_parquet():
    df = pd.DataFrame({"name": TEST_S2_FOLDER.name, "labels": ["Pastures"]})
    with tempfile.NamedTemporaryFile() as tmp_file:
        df.to_parquet(tmp_file.name)
        func = patch_path_to_metadata_func_builder_from_parquet(
            parquet_path=tmp_file.name
        )
    metadata = func(TEST_S2_FOLDER)
    assert isinstance(metadata, dict)
    assert metadata["labels"] == df.iloc[0]["labels"]


def test_patch_path_to_metdata_func_builder_parquet_drop():
    df = pd.DataFrame(
        {
            "name": TEST_S2_FOLDER.name,
            "labels": ["Pastures"],
            "drop_me": ["please"],
        }
    )
    with tempfile.NamedTemporaryFile() as tmp_file:
        df.to_parquet(tmp_file.name)
        func = patch_path_to_metadata_func_builder_from_parquet(
            parquet_path=tmp_file.name, drop_cols=["drop_me"]
        )
    metadata = func(TEST_S2_FOLDER)
    assert isinstance(metadata, dict)
    assert metadata["labels"] == df.iloc[0]["labels"]
    assert "drop_me" not in metadata.keys()


def test_load_labels_from_s2_patch_path():
    metadata = load_labels_from_patch_path(TEST_S2_FOLDER, is_sentinel2=True)
    assert metadata["labels"] == ["Pastures"]
    assert metadata["new_labels"] == ["Pastures"]
    metadata = load_labels_from_patch_path(
        TEST_S2_FOLDER, is_sentinel2=True, infer_new_labels=False
    )
    assert "new_labels" not in metadata.keys()


def test_load_labels_from_s1_patch_path():
    metadata = load_labels_from_patch_path(TEST_S1_FOLDER, is_sentinel2=False)
    assert metadata["labels"] == ["Pastures"]
    assert metadata["new_labels"] == ["Pastures"]
    metadata = load_labels_from_patch_path(
        TEST_S1_FOLDER, is_sentinel2=False, infer_new_labels=False
    )
    assert "new_labels" not in metadata.keys()
