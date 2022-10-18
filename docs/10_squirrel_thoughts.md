# Squirrel thoughts
> A couple of thoughts about the squirrel library

:::{warning}
This is an initial draft discussing the pros/cons of the {{Squirrel}} library for BigEarthNet!

Please skip this section if you are not actively working on the Encoder library!
:::

In general, the core idea of the Squirrel library is that it is an iterator-based dataset wrapper.
As a result, the encoded dataset is designed to be consumed entirely.
If only small random subsets are loaded, I expect the Squirrel data to perform poorly.

The key strategy to maximize the read-performance is to read _large_ files.
SSDs have a low random-read performance (~200--900 MB/s).
Frequently reading small chunks of data (i.e., images/patches) is nothing else than randomly reading from disk.
The data must be converted into a different format that is better suited to allow quick data access.

To maximize the read-performance, the data must be orders of magnitudes greater than 4KB (the smallest unit of data accessed by the SSD).
A possible solution is _sharding_.
_Sharding_ aggregates multiple _smaller_ files into _larger_ storage units (_shards_).
A simple way to generate shards is to convert the data into the GNU `tar` format.
Combining the sharding strategy with compression can reduce the data size, benefiting data accessed over the network.
The recommended size of each shard is between 128MB -- 1GB.
For more details, refer to the [High Performance I/O for large scale deep learning paper from NVIDIA](https://arxiv.org/abs/2001.01858).

Important questions that need to be answered:
- Can sharding be applied in such a way that it could be split by country?
    - Possibly with suffixes to hit the size sweet-spot
    - How hard is the performance impact, if I access a random subset?
        - This is not _too_ important because we could regenerate the dataset depending on the application
    - What is the impact on the randomness of the data?
        - How to tune the parameters to create _truly random_ batches?
        - We train with batch sizes from 128 -- 2048
- Is there a way to cache the entire dataset in memory?
- How to efficiently split train/validation/test set?
    - Their example code uses datasets that already provide train/validation/test splits as shards.
    - [CC100](https://squirrel-datasets-core.readthedocs.io/en/latest/dataset_links/cc100.html) from the Squirrel-Datasets documentation does not give a train/test/validation split
        - Sharded by language
    - It looks like the best approach would be to re-generate the dataset according to the used split to optimize the performance

## Shard size
Converting the patches to the intermediate {{BenInterface}} format, the resulting object size is ~16KB.
To create shards with a size of 1GB, we could combine ~6650 patches (without compression).
If we train with the usual batch sizes of around 1024 patches per batch, how strongly would the shard-size impact the randomness?

How much time does the decompression take?
Would it make sense to compress the data in a different format to GZIP, like LZ4 or ZSTD, which may decode faster than GZIP?

## Performance discussion
I am currently using `cProfile`:
```bash
python -m cProfile -o stats.dump reader.py
snakeviz stats.dump
```
To analyze where most time is spent.

In my first mini-tests, the Squirrel dataset is significantly slower than the LMDB variant.
The main issue is that most time is spent decompressing the archive.
Without the decompression, the performance would be roughly similar.

These results are done without applying any shuffling at all!

With GZIP compression:
- 2 * 6600 patches = 2 * 797MB
- Iteration over both shards takes ~14sek
    - Most of the time is required to unpack the archive
    - It would take ~10min to pass through the dataset
    - Could it potentially be minimized by prefetching?

LMDB:
- 2 * 6600 patches = 2.2GB
- Iteration of the entire LMDB file takes ~4sek
    - To iterate over BigEarthNet, it would take ~3min



## Data access in detail
- With `keys_iterable` only specific shards can be accessed.
- The keys themselves can be easily shuffled.
- So the shard access can be easily randomized.
- `shuffle_item_buffer` is size of the buffer used to shuffle items after items are fetched.
- Are multiple shards unpacked?

## Data shuffling in detail
```{figure} images/squirrel_shuffle.svg
Data shuffling in squirrel
```
