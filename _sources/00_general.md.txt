# General

:::{admonition} TL;DR
:class: important

Loading raw BigEarthNet data is slow and _has_ to be converted to a different deep-learning optimized archive for faster training.
The following points must be considered when converting the data:
- The bands have different resolutions
    - Pre-interpolating allows for faster reading _but_ less flexibility
- The bands are not equally important
    - It is common to not use the 60m bands as they _often_ contribute little to the overall performance
- The output format should be easy to work with
:::

One crucial component of reducing the training time is minimizing data access.
There are _many_ different approaches with their pros and cons.
This section assembles a couple of popular approaches and discusses possible shortcomings.

The assumption is that the data loading procedure is slow and becomes a bottleneck during training.
Looking at the structure of the data, we can identify three possible issues:

1. Provided patches are slow to be read
    - The data is split across multiple files. Each band is stored separately because of the different spatial resolution
    - Each `tif` ([GeoTIFF](https://en.wikipedia.org/wiki/GeoTIFF)) file contains unnecessary metadata for the training of deep neural networks
2. Images need to be converted before passing through common deep learning pipelines
    - Data is stored as `uint16`; most libraries require `float32` or `int`
3. Lower resolution bands need to be interpolated if they should be converted to a _tensor-like_ structure

## Requirements
Many tutorials compare and discuss libraries for converting the data into a different format or storing it in a database to optimize the access but are often not well-suited for our heterogeneous satellite data.

To allow us to work with the multi-spectral satellite data flexibly, we have the following requirements:

### Flexible loading of different channels
Initial experiments are often performed on the RGB-only bands to re-use code from other projects/papers that were originally created for the CV domain.
Some experiments may only load the 10m and 20m bands, as the 60m bands _often_ have negligible impact on the performance.
Dropping these bands allows training with larger batches, reducing training time.
This requirement is orthogonal to point `3` of the possible bottlenecks during data loading.

For large scale benchmark runs, it would be beneficial to pre-process the individual bands, i.e.:
- Interpolate to a common dimensionality
- Drop unused bands
- Convert from `uint16` to `float32` or `int`

But for most scenarios, the decreased flexibility is a big issue.
To keep the data as accessible as possible for future use, the data should be as close as possible to the _original_ satellite data as possible.

### Easy to use output archive
- Colleagues should be able to _just start_ working with the converted archive without having to know the details of the conversion process.
- Do not require heavy external binaries when accessing archive
  - rasterio is complicated to install/brings many hard dependencies with it
- Do not require any _real_ database connections/databases

### Deep-Learning Library agnostic
- Should work with tensor-flow, pytorch or even more exotic interfaces

### Embedding of metadata (Retracted)
Besides the 19-class nomenclature, some experiments could also be performed on the 43-class nomenclature for comparison with other papers.
Or geographic metadata about each patch could be used for _remote-sensing_ specific training strategies.
As satellite data has rich metadata, it would be nice to have an easy way to store the metadata _within_ the converted output format.

#### Reasons why metadata embedding is not important
After spending some more time working with satellite data and deep-learning, I personally think that this isn't as important as I initially thought.
It is often _easier_ and more flexible to keep the metadata separate.

For example: If we do some smart geographic location-based sampling, the metadata has to be available at the start of the epoch to calculate what images should be sampled.
If the metadata is _embedded_ with the image data in a common structure, the image data would probably have to be loaded even if it isn't necessary.
Some libraries allow metadata reading without loading the images, but this is generally not true for _key-value storages_.

## Candidates
- [LMDB](https://realpython.com/storing-images-in-python/#reading-from-lmdb)
- H5PY
- [FFCV](https://ffcv.io/)
