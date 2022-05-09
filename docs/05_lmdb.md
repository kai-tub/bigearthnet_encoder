(lmdb)=
# LMDB
[LMDB](https://www.symas.com/lmdb) (Lightning Memory-Mapped Database) is a key-value storage format where each entry is stored as a byte array.
Like a Python dictionary, the key must be unique and the value can be any object.
In the Python domain, the keys/values are usually _pickled_ to serialize the data to the binary format expected by LMDB.

The encoder can create an S1/S2 or a combined S1 and S2 LMDB archive:
- The encoder uses the patch name as the key
    - The patch name is either the S1 or S2 name depending on the source dataset
    - If the combined S1 and S2 LMDB archive should be generated, the S2 patch name is used
- The _value_ is the BigEarthNet-Patch-Interface object
- The encoder provides a CLI to easily generate the required archive
    - Either a simple LMDB archive that only contains the image data as an numpy array (`raw`)
    - Or an archive that already adds the 43- and 19-class nomenclature labels as metadata (`with-lbls`)

```
ben_encoder write-s<1,2>-lmdb-raw <PATH TO BEN DIRECTORY> <OUTPUT PATH>
# OR
ben_encoder write-s<1,2>-lmdb-with-labels <PATH TO BEN DIRECTORY> <OUTPUT PATH>
```

To access the values, the pickled data has to be loaded with `BigEarthNet_S1_Patch.load`.

After creating the archive, the archive should have restriced write access, to ensure that the data isn't accidentally touched.

## Short-comings
There are a couple of issues with the LMDB approach.
- LMDB reads the entire data when given a key
    - If all 12 bands are encoded, all 12 bands are always read even if only 3 or 10 are required
    - To optimize read performance, there would have to be multiple archives with the _minimum_ data for the specific task
- Requires to _know_ what is being read
    - The data is encoded as a binary stream and requires the user to know how to _unpickle_ the data
    - Here, the {{ BenInterface }} library has to be installed
- Hard to get optimal read performance with _batched_ input data
    - PyTorch style data loading will create a new connection to the LMDB archive for _each_ image file and not use a single one for the entire batch
    - _Could_ be fixed by writing specialized dataloader but is not trivial
- Not easy to control caching behavior
    - No way to enforce to read entire archive into RAM for maximum read performance (at least I couldn't figure it out)
- Relatively hard-to-read documentation (personal opinion)
