from typing import List, Optional

from pydantic import validate_arguments, FilePath
import pandas as pd


@validate_arguments
def patch_name_to_metadata_func_builder_from_parquet(
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

    def patch_name_to_metdata_func(patch_name):
        return indexed_df.loc[patch_name].to_dict()

    return patch_name_to_metdata_func
