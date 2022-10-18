from typing import Dict, List, Optional

import pandas as pd
from bigearthnet_common.base import old2new_labels, read_S1_json, read_S2_json
from pydantic import DirectoryPath, FilePath, validate_arguments


@validate_arguments
def patch_path_to_metadata_func_builder_from_parquet(
    parquet_path: FilePath,
    patch_name_col: str = "name",
    drop_cols: Optional[List[str]] = None,
):
    """
    Read from parquet and not directly from dataframe for two reasons:

    1. Enforce that the data is stored on disk to ensure reproducible builds
    2. There is often an encoding-difference between raw dataframes and the loaded parquet dataframes
        - For example, lists are encoded as np.ndarray of dtype='object' when stored as parquet
    """
    df = pd.read_parquet(parquet_path)
    indexed_df = df.set_index(
        patch_name_col,
    )
    if drop_cols is not None:
        indexed_df = indexed_df.drop(drop_cols, axis=1)

    @validate_arguments
    def patch_name_to_metdata_func(patch_path: DirectoryPath):
        patch_name = patch_path.name
        return indexed_df.loc[patch_name].to_dict()

    return patch_name_to_metdata_func


@validate_arguments
def load_labels_from_patch_path(
    patch_directory: DirectoryPath, is_sentinel2: bool, infer_new_labels: bool = True
) -> Dict[str, str]:
    read_json = read_S2_json if is_sentinel2 else read_S1_json
    metadata = read_json(
        patch_directory / f"{patch_directory.name}_labels_metadata.json"
    )
    result = {"labels": metadata["labels"]}
    if infer_new_labels:
        result["new_labels"] = old2new_labels(result["labels"])
    return result
