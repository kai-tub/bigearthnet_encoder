import tempfile
from pathlib import Path

from bigearthnet_encoder.metadata_utils import *

TEST_S2_ROOT = Path(__file__).parent / "s2-tiny"
TEST_S2_FOLDER = TEST_S2_ROOT / "S2A_MSIL2A_20170617T113321_4_55"
TEST_S1_ROOT = Path(__file__).parent / "s1-tiny"
TEST_S1_FOLDER = TEST_S1_ROOT / "S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39"


def test_patch_path_to_metdata_func_builder_parquet():
    df = pd.DataFrame({"name": TEST_S2_FOLDER.name, "labels": ["Marine waters"]})
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
            "labels": ["Marine waters"],
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
    assert metadata["labels"] == [
        "Non-irrigated arable land",
        "Broad-leaved forest",
        "Water courses",
    ]
    assert metadata["new_labels"] == [
        "Arable land",
        "Broad-leaved forest",
        "Inland waters",
    ]
    metadata = load_labels_from_patch_path(
        TEST_S1_FOLDER, is_sentinel2=False, infer_new_labels=False
    )
    assert "new_labels" not in metadata.keys()
