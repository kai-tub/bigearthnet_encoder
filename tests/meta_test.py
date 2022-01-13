from pathlib import Path
import tempfile
from bigearthnet_encoder.metadata_utils import *

TEST_S2_ROOT = Path(__file__).parent / "s2-tiny"
TEST_S2_FOLDER = TEST_S2_ROOT / "S2A_MSIL2A_20170617T113321_4_55"


def test_patch_name_to_metdata_func_builder_parquet():
    df = pd.DataFrame({"name": TEST_S2_FOLDER.name, "labels": ["Marine waters"]})
    with tempfile.NamedTemporaryFile() as tmp_file:
        df.to_parquet(tmp_file.name)
        patch_name_to_metadata_func_builder_from_parquet(parquet_path=tmp_file.name)


def test_patch_name_to_metdata_func_builder_parquet_drop():
    df = pd.DataFrame(
        {
            "name": TEST_S2_FOLDER.name,
            "labels": ["Marine waters"],
            "drop_me": ["please"],
        }
    )
    with tempfile.NamedTemporaryFile() as tmp_file:
        df.to_parquet(tmp_file.name)
        patch_name_to_metadata_func_builder_from_parquet(
            parquet_path=tmp_file.name, drop_cols=["drop_me"]
        )
