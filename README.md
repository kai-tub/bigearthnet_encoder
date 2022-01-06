# BigEarthNet Encoder

Uses
- [BigEarthNet-Patch-Interface](https://github.com/kai-tub/bigearthnet_patch_interface)
    - Interface that is used to combine all data points into a single structure
    - Each instance will encode itself via pickle and is ready to be inserted into a database and unpickled without any dependencies
- LMDB to write the database with the patch-name as the key

Should be combined with:
- [ben_gdf_builder](https://github.com/kai-tub/bigearthnet_common)
    - Allows to easily extend the database with metadata
