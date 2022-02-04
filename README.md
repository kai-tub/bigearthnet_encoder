# BigEarthNet Encoder

Uses
- [BigEarthNet-Patch-Interface](https://github.com/kai-tub/bigearthnet_patch_interface)
    - Interface that is used to combine all data points into a single structure
    - Each instance will encode itself via pickle and is ready to be inserted into a database and unpickled without any dependencies
- LMDB to write the database with the patch-name as the key

```
ben_encoder write-s<1,2>-lmdb-raw <PATH TO BEN DIRECTORY> <OUTPUT PATH>
```

The library supports easy construction of an LMDB archive with the old 43-label and the newer 19-label nomenclature as the metadata.

```
ben_encoder write-s<1,2>-lmdb-with-labels <PATH TO BEN DIRECTORY> <OUTPUT PATH>
```

_Could_ be combined with:
- [ben_gdf_builder](https://github.com/kai-tub/bigearthnet_common)
    - Allows to easily extend the database with complex metadata

## Installation
While the library and the patch interface are still in development, install the library directly from git:

```sh
python -m pip install --upgrade git+https://github.com/kai-tub/bigearthnet_encoder.git
```
