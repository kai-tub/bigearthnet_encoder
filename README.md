# BigEarthNet Encoder

The goal of the BigEarthNet Encoder library is to quickly transform the original BigEarthNet archive to a deep-learning optimized format.
The long-term goal is to support multiple output formats.

To simplify the process of working with BigEarthNet, each patch is first converted to a [BigEarthNet-Patch-Interface](https://docs.kai-tub.tech/bigearthnet_patch_interface/).
This interface will guarantee that the data is complete and valid before moving on to creating desired format.
The patch-data is internally stored as an numpy array to keep the data in an deep-learning framework agnostic format.

The library should provide all the necessary functionality via a CLI to allow for quick conversion without requiring to understand the details of the conversion process.

As of now, the only supported target format is the LMDB archive format.
