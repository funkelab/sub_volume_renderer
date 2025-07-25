# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "pathlib",
#     "tensorstore",
#     "zarr",
# ]
# ///
# entire file written by Claude Code with Claude Sonnet 4
from pathlib import Path

import numpy as np
import tensorstore as ts
import zarr


def is_power_of_2(n):
    """Check if a number is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def downsample_mean(data, factor=2):
    """Downsample data by taking the mean over factor^3 blocks."""
    print(
        f"DEBUG: downsample_mean called with data shape {data.shape}, factor {factor}"
    )
    shape = np.array(data.shape)
    print(f"DEBUG: shape array: {shape}")

    # Assert that all dimensions are divisible by factor
    for i, dim in enumerate(shape):
        print(
            f"DEBUG: checking dimension {i} ({dim}) divisible by {factor}: {dim % factor == 0}"
        )
        assert dim % factor == 0, (
            f"Dimension {i} ({dim}) is not divisible by factor {factor}"
        )

    new_shape = shape // factor
    print(f"DEBUG: new_shape after downsampling: {new_shape}")

    # Reshape and take mean
    print(
        f"DEBUG: reshaping data to {(new_shape[0], factor, new_shape[1], factor, new_shape[2], factor)}"
    )
    reshaped = data.reshape(
        new_shape[0], factor, new_shape[1], factor, new_shape[2], factor
    )
    print(f"DEBUG: reshaped data shape: {reshaped.shape}")

    result = reshaped.mean(axis=(1, 3, 5))
    print(f"DEBUG: mean result shape: {result.shape}")
    return result


def main():
    # Load the original zarr data
    print("DEBUG: Starting main function")
    print("DEBUG: Loading original zarr data...")
    data = ts.open(
        spec={
            "driver": "zarr",
            "kvstore": "file:///nrs/funke/data/lightsheet/160315_mouse/160315.zarr/raw",
        }
    ).result()
    print(f"DEBUG: Original data loaded, shape: {data.shape}, dtype: {data.dtype}")

    # Apply the specified slice
    print("DEBUG: Applying slice [100, 0, 117:629, 131:2179, 230:2278]...")
    sliced_data = data[100, 0, 117:629, 131:2179, 230:2278]
    print(f"DEBUG: Sliced data shape: {sliced_data.shape}")
    print(f"DEBUG: Sliced data type: {type(sliced_data)}")

    # Convert to numpy array for processing
    print("DEBUG: Converting to numpy array...")
    sliced_data = np.array(sliced_data)
    print(f"DEBUG: Numpy array shape: {sliced_data.shape}, dtype: {sliced_data.dtype}")

    # Convert to float32
    print("DEBUG: Converting to float32...")
    sliced_data = sliced_data.astype(np.float32)
    print(
        f"DEBUG: After float32 conversion: shape: {sliced_data.shape}, dtype: {sliced_data.dtype}"
    )

    # Assert that all dimensions are powers of 2
    print("DEBUG: Checking if all dimensions are powers of 2...")
    for i, dim in enumerate(sliced_data.shape):
        is_pow2 = is_power_of_2(dim)
        print(f"DEBUG: dimension {i} = {dim}, is_power_of_2: {is_pow2}")
        assert is_pow2, f"Dimension {i} ({dim}) is not a power of 2"
    print(f"DEBUG: Verified all dimensions are powers of 2: {sliced_data.shape}")

    # Create output directory
    print("DEBUG: Creating output directory...")
    output_path = Path("/nrs/funke/data/sub_volume/mouse")
    print(f"DEBUG: Output path: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    print("DEBUG: Output directory created successfully")

    # Create zarr group with sharding
    print("DEBUG: Creating zarr group...")
    raw_store = zarr.storage.LocalStore(str(output_path / "raw.zarr"))
    print(f"DEBUG: Store path: {output_path / 'raw.zarr'}")
    group = zarr.create_group(store=raw_store, overwrite=True)
    print("DEBUG: Zarr group created successfully")

    # Define chunk and shard sizes
    chunk_size = (16, 16, 16)
    shard_size = (64, 64, 64)  # 4x4x4 chunks per shard
    print(f"DEBUG: chunk_size: {chunk_size}, shard_size: {shard_size}")

    # Create codec with sharding
    print("DEBUG: Creating sharding codec...")
    sharding_codec = zarr.codecs.ShardingCodec(
        chunk_shape=chunk_size,
        codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()],
    )
    print("DEBUG: Sharding codec created successfully")

    # Save scale0 (original sliced data)
    print("DEBUG: Saving scale0 (original data)...")
    print(f"DEBUG: scale0 data shape: {sliced_data.shape}, dtype: {sliced_data.dtype}")
    scale0_array = group.create_array(
        name="scale0",
        shape=sliced_data.shape,
        chunks=chunk_size,
        dtype=sliced_data.dtype,
        shards=shard_size,
    )
    scale0_array[:] = sliced_data
    print(f"DEBUG: scale0 saved successfully, shape: {sliced_data.shape}")

    # Generate and save downsampled scales
    print("DEBUG: Starting downsampling loop...")
    current_data = sliced_data
    for scale_level in range(1, 5):
        print(f"DEBUG: Processing scale{scale_level} (downsampling by factor 2)...")
        print(f"DEBUG: Input data shape for scale{scale_level}: {current_data.shape}")
        current_data = downsample_mean(current_data, factor=2)
        print(
            f"DEBUG: Downsampled data shape for scale{scale_level}: {current_data.shape}"
        )

        print(f"DEBUG: Creating zarr array for scale{scale_level}...")
        scale_array = group.create_array(
            name=f"scale{scale_level}",
            shape=current_data.shape,
            chunks=chunk_size,
            dtype=current_data.dtype,
            shards=shard_size,
        )
        scale_array[:] = current_data
        print(
            f"DEBUG: scale{scale_level} saved successfully, shape: {current_data.shape}"
        )

    print(f"DEBUG: Multiscale zarr saved to: {output_path / 'raw.zarr'}")
    print("DEBUG: Raw scale summary:")
    for scale_name in sorted(group.array_keys()):
        arr = group[scale_name]
        print(f"DEBUG: {scale_name}: {arr.shape} (dtype: {arr.dtype})")

    # Create labels zarr with same structure but uint32 dtype and constant values
    print("\nDEBUG: Starting labels zarr creation...")
    print("DEBUG: Creating labels zarr store...")
    labels_store = zarr.storage.LocalStore(str(output_path / "labels.zarr"))
    print(f"DEBUG: Labels store path: {output_path / 'labels.zarr'}")
    labels_group = zarr.create_group(store=labels_store, overwrite=True)
    print("DEBUG: Labels zarr group created successfully")

    # Create sharding codec for uint32
    print("DEBUG: Creating labels sharding codec...")
    labels_sharding_codec = zarr.codecs.ShardingCodec(
        chunk_shape=chunk_size,
        codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()],
    )
    print("DEBUG: Labels sharding codec created successfully")

    # Create label arrays with constant values
    print("DEBUG: Starting labels creation loop...")
    current_data = sliced_data
    for scale_level in range(5):  # 0 to 4
        print(f"DEBUG: Processing labels scale{scale_level}...")
        if scale_level == 0:
            labels_shape = sliced_data.shape
            print(f"DEBUG: Using original shape for scale0: {labels_shape}")
        else:
            # Use the same downsampling logic to get the shape
            print(f"DEBUG: Downsampling to get shape for scale{scale_level}...")
            current_data = downsample_mean(current_data, factor=2)
            labels_shape = current_data.shape
            print(f"DEBUG: Downsampled shape for scale{scale_level}: {labels_shape}")

        print(
            f"DEBUG: Creating labels scale{scale_level} with value {scale_level}, shape {labels_shape}..."
        )
        labels_data = np.full(labels_shape, scale_level, dtype=np.uint32)
        print(
            f"DEBUG: Created np.full array with shape {labels_data.shape}, dtype {labels_data.dtype}, value {scale_level}"
        )

        print(f"DEBUG: Saving labels scale{scale_level} to zarr...")
        labels_array = labels_group.create_array(
            name=f"scale{scale_level}",
            shape=labels_shape,
            chunks=chunk_size,
            dtype=np.uint32,
            shards=shard_size,
        )
        labels_array[:] = labels_data
        print(
            f"DEBUG: labels scale{scale_level} saved successfully, shape: {labels_shape}"
        )

    print(f"DEBUG: Labels zarr saved to: {output_path / 'labels.zarr'}")
    print("DEBUG: Labels scale summary:")
    for scale_name in sorted(labels_group.array_keys()):
        arr = labels_group[scale_name]
        print(f"DEBUG: {scale_name}: {arr.shape} (dtype: {arr.dtype})")

    print("DEBUG: Script completed successfully!")


if __name__ == "__main__":
    main()
