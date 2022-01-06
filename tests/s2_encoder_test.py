import pytest
from bigearthnet_encoder._tif_reader import *
from pathlib import Path
from pydantic import ValidationError

TEST_FOLDER = Path(__file__).parent / Path("tiny/S2A_MSIL2A_20170617T113321_4_55")
TEST_FILE = TEST_FOLDER / "S2A_MSIL2A_20170617T113321_4_55_B01.tif"
# TODO
