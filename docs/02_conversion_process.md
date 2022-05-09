# Conversion process

The general conversion process is as follows:

1. Locate _raw_ BigEarthNet-S1/2 TIFF files
1. Convert to {{ BenInterface }}
    - Strict input data reader that ensures that the data is in the expected format and valid
    - Has easy introspection features and can hold arbitrary metadata
1. Convert from {{ BenInterface }} to target format.

The encoder library provides a couple of functions to quickly convert the original BigEarthNet directories to the desired {{ BenInterface }}.

The specific functions are:
- {any}`tiff_dir_to_ben_s1_patch`
- {any}`tiff_dir_to_ben_s2_patch`

To read the TIFF files the data has to be read with `rasterio`.
