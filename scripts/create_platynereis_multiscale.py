# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "pathlib",
#     "zarr",
# ]
# ///
# entire file written by Claude Code with Claude Sonnet 4
from collections import Counter
from pathlib import Path

import numpy as np
import zarr


def is_power_of_2(n):
    """Check if a number is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def center_data_in_target(data, target_shape):
    """Center data in a larger target shape array."""
    print(
        f"DEBUG: center_data_in_target called with data shape {data.shape}, target_shape {target_shape}"
    )

    # Create target array filled with zeros
    target = np.zeros(target_shape, dtype=data.dtype)
    print(
        f"DEBUG: Created target array with shape {target.shape}, dtype {target.dtype}"
    )

    # Calculate offset to center the data
    data_shape = np.array(data.shape)
    target_shape_arr = np.array(target_shape)
    offset = (target_shape_arr - data_shape) // 2
    print(f"DEBUG: Calculated offset: {offset}")

    # Calculate slice indices for placing data
    slices = tuple(
        slice(offset[i], offset[i] + data_shape[i]) for i in range(len(data_shape))
    )
    print(f"DEBUG: Placement slices: {slices}")

    # Place data in center of target array
    target[slices] = data
    print(f"DEBUG: Data centered successfully in target array")

    return target


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


def downsample_labels_majority(labels, factor=2):
    """Downsample labels using majority vote with tie-breaking (smallest non-zero label)."""
    print(
        f"DEBUG: downsample_labels_majority called with labels shape {labels.shape}, factor {factor}"
    )
    shape = np.array(labels.shape)
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

    # Reshape to group factor^3 blocks together
    reshaped = labels.reshape(
        new_shape[0], factor, new_shape[1], factor, new_shape[2], factor
    )
    print(f"DEBUG: reshaped labels shape: {reshaped.shape}")

    # Apply majority vote to each block
    result = np.zeros(new_shape, dtype=labels.dtype)

    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                # Get the factor^3 block
                block = reshaped[i, :, j, :, k, :]
                block_flat = block.flatten()

                # Count occurrences of each label
                label_counts = Counter(block_flat)

                # Find the maximum count
                max_count = max(label_counts.values())

                # Get all labels with maximum count
                tied_labels = [
                    label for label, count in label_counts.items() if count == max_count
                ]

                # Tie-breaking: choose smallest non-zero label, or zero if all tied labels are zero
                if len(tied_labels) == 1:
                    result[i, j, k] = tied_labels[0]
                else:
                    # Multiple labels tied - apply tie-breaking rule
                    non_zero_tied = [label for label in tied_labels if label != 0]
                    if non_zero_tied:
                        # Choose smallest non-zero label
                        result[i, j, k] = min(non_zero_tied)
                    else:
                        # All tied labels are zero
                        result[i, j, k] = 0

    print(f"DEBUG: majority vote result shape: {result.shape}")
    return result


def main():
    print("DEBUG: Starting main function")

    # Load the original zarr data
    print("DEBUG: Loading original platynereis zarr data...")
    raw_data = zarr.open_array(
        "/nrs/funke/data/lightsheet/130312_platynereis/13-03-12.zarr/raw"
    )
    print(f"DEBUG: Raw data loaded, shape: {raw_data.shape}, dtype: {raw_data.dtype}")

    segmentation_data = zarr.open_array(
        "/nrs/funke/data/lightsheet/130312_platynereis/13-03-12.zarr/segmentation"
    )
    print(
        f"DEBUG: Segmentation data loaded, shape: {segmentation_data.shape}, dtype: {segmentation_data.dtype}"
    )

    # Apply the specified slice [0, 378, :, :, :]
    print("DEBUG: Applying slice [0, 378, :, :, :]...")
    sliced_raw = raw_data[0, 378, :, :, :]
    sliced_seg = segmentation_data[0, 378, :, :, :]
    print(f"DEBUG: Sliced raw data shape: {sliced_raw.shape}")
    print(f"DEBUG: Sliced segmentation shape: {sliced_seg.shape}")

    # Convert to numpy arrays and proper dtypes
    print("DEBUG: Converting to numpy arrays...")
    sliced_raw = np.array(sliced_raw).astype(np.float32)
    sliced_seg = np.array(sliced_seg).astype(np.uint32)
    print(f"DEBUG: Raw array shape: {sliced_raw.shape}, dtype: {sliced_raw.dtype}")
    print(
        f"DEBUG: Segmentation array shape: {sliced_seg.shape}, dtype: {sliced_seg.dtype}"
    )

    # Center data in target 128x768x768 arrays
    target_shape = (128, 768, 768)
    print(f"DEBUG: Centering data in target shape: {target_shape}")

    centered_raw = center_data_in_target(sliced_raw, target_shape)
    centered_seg = center_data_in_target(sliced_seg, target_shape)

    print(f"DEBUG: Centered raw data shape: {centered_raw.shape}")
    print(f"DEBUG: Centered segmentation shape: {centered_seg.shape}")

    # Log target shape info (768 = 256 × 3, so not pure power of 2)
    print("DEBUG: Target shape analysis...")
    for i, dim in enumerate(target_shape):
        is_pow2 = is_power_of_2(dim)
        print(f"DEBUG: target dimension {i} = {dim}, is_power_of_2: {is_pow2}")
    print(f"DEBUG: Target shape: {target_shape} (note: 768 contains factor of 3)")

    # Create output directory
    print("DEBUG: Creating output directory...")
    output_path = Path("/nrs/funke/data/sub_volume/platynereis")
    print(f"DEBUG: Output path: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    print("DEBUG: Output directory created successfully")

    # Create raw zarr group
    print("DEBUG: Creating raw zarr group...")
    raw_store = zarr.storage.LocalStore(str(output_path / "raw.zarr"))
    print(f"DEBUG: Raw store path: {output_path / 'raw.zarr'}")
    raw_group = zarr.create_group(store=raw_store, overwrite=True)
    print("DEBUG: Raw zarr group created successfully")

    # Create labels zarr group
    print("DEBUG: Creating labels zarr group...")
    labels_store = zarr.storage.LocalStore(str(output_path / "labels.zarr"))
    print(f"DEBUG: Labels store path: {output_path / 'labels.zarr'}")
    labels_group = zarr.create_group(store=labels_store, overwrite=True)
    print("DEBUG: Labels zarr group created successfully")

    # Define chunk and shard sizes
    chunk_size = (16, 16, 16)
    shard_size = (64, 64, 64)  # 4x4x4 chunks per shard
    print(f"DEBUG: chunk_size: {chunk_size}, shard_size: {shard_size}")

    # Generate multiscale levels
    current_raw = centered_raw
    current_labels = centered_seg

    for scale_level in range(5):  # 0 to 4
        print(f"\nDEBUG: Processing scale{scale_level}...")

        if scale_level > 0:
            print(f"DEBUG: Downsampling for scale{scale_level}...")
            print(f"DEBUG: Input raw shape: {current_raw.shape}")
            print(f"DEBUG: Input labels shape: {current_labels.shape}")

            current_raw = downsample_mean(current_raw, factor=2)
            current_labels = downsample_labels_majority(current_labels, factor=2)

            print(f"DEBUG: Downsampled raw shape: {current_raw.shape}")
            print(f"DEBUG: Downsampled labels shape: {current_labels.shape}")

        # Save raw data
        print(f"DEBUG: Saving raw scale{scale_level}...")
        raw_array = raw_group.create_array(
            name=f"scale{scale_level}",
            shape=current_raw.shape,
            chunks=chunk_size,
            dtype=current_raw.dtype,
            shards=shard_size,
        )
        raw_array[:] = current_raw
        print(f"DEBUG: Raw scale{scale_level} saved, shape: {current_raw.shape}")

        # Save labels
        print(f"DEBUG: Saving labels scale{scale_level}...")
        labels_array = labels_group.create_array(
            name=f"scale{scale_level}",
            shape=current_labels.shape,
            chunks=chunk_size,
            dtype=current_labels.dtype,
            shards=shard_size,
        )
        labels_array[:] = current_labels
        print(f"DEBUG: Labels scale{scale_level} saved, shape: {current_labels.shape}")

    # Print final summaries
    print(f"\nDEBUG: Raw zarr saved to: {output_path / 'raw.zarr'}")
    print("DEBUG: Raw scale summary:")
    for scale_name in sorted(raw_group.array_keys()):
        arr = raw_group[scale_name]
        print(f"DEBUG: {scale_name}: {arr.shape} (dtype: {arr.dtype})")

    print(f"\nDEBUG: Labels zarr saved to: {output_path / 'labels.zarr'}")
    print("DEBUG: Labels scale summary:")
    for scale_name in sorted(labels_group.array_keys()):
        arr = labels_group[scale_name]
        print(f"DEBUG: {scale_name}: {arr.shape} (dtype: {arr.dtype})")

    print("DEBUG: Script completed successfully!")


if __name__ == "__main__":
    main()
