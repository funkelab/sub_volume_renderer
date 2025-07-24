// Multi-scale volume sampling with high-to-low fallthrough logic

$$ for i in range(num_scales)
fn try_sample_scale_{ { i }}(data_tex_coord: vec3<f32>, sizef: vec3<f32>) -> vec4<f32> {
    // Transform coordinates for this scale level
    let data_coord = data_tex_coord * sizef;
    let scale_factor = u_wrapping_buffer_{{ i }}.scale_factor;
    let scaled_data_coord = data_coord * scale_factor;

    // For same-sized voxels: scale_factor represents how to scale the texture coordinates
    // Scale 0: scale_factor=1.0, Scale 1: scale_factor=0.5 (to make voxels appear 2x larger)
    let ring_buffer_dimensions = vec3<f32>(textureDimensions(t_scale_,{{ i}}));

    let offset = u_wrapping_buffer_{{ i }}.current_logical_offset_in_pixels;
    let shape = u_wrapping_buffer_{{ i }}.current_logical_shape_in_pixels;

    let in_bounds = all(offset <= vec3<i32>(scaled_data_coord)) && all(vec3<i32>(scaled_data_coord) < offset + shape);
    if !in_bounds {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0); // Invalid sample - w=0 indicates no data
    }

    let wrapped_scaled_data_coord = scaled_data_coord % ring_buffer_dimensions;
    let result = textureLoad(t_scale_,{{ i,}}, vec3<i32>(wrapped_scaled_data_coord), 0);
    return vec4<f32>(result.rgb, 1.0); // Valid sample - w=1 indicates data available
}

fn try_sample_segmentations_scale_{ { i }}(data_tex_coord: vec3<f32>, sizef: vec3<f32>) -> vec4<u32> {
    // Transform coordinates for this scale level
    let data_coord = data_tex_coord * sizef;
    let scale_factor = u_wrapping_buffer_{{ i }}.scale_factor;
    let scaled_data_coord = data_coord * scale_factor;

    // For same-sized voxels: scale_factor represents how to scale the texture coordinates
    // Scale 0: scale_factor=1.0, Scale 1: scale_factor=0.5 (to make voxels appear 2x larger)
    let ring_buffer_dimensions = vec3<f32>(textureDimensions(t_segmentations_scale_,{{ i}}));

    let offset = u_wrapping_buffer_{{ i }}.current_logical_offset_in_pixels;
    let shape = u_wrapping_buffer_{{ i }}.current_logical_shape_in_pixels;

    let in_bounds = all(offset <= vec3<i32>(scaled_data_coord)) && all(vec3<i32>(scaled_data_coord) < offset + shape);
    if !in_bounds {
        return vec4<u32>(0, 0, 0, 0); // Invalid sample - w=0 indicates no data
    }

    let wrapped_scaled_data_coord = scaled_data_coord % ring_buffer_dimensions;
    let result = textureLoad(t_segmentations_scale_,{{ i,}}, vec3<i32>(wrapped_scaled_data_coord), 0);
    return vec4<u32>(result.rgb, 1); // Valid sample - w=1 indicates data available
}
$$ endfor

fn sample_vol_multi_scale(data_tex_coord: vec3<f32>, sizef: vec3<f32>) -> vec4<f32> {
    // Try scales from highest to lowest resolution (0 to num_scales-1)
    var result: vec4<f32>;
    $$ for iinrange(num_scales) result =try_sample_scale_ { { i }
}(data_tex_coord, sizef);
    if result.w > 0.0 { // Valid sample found
        return result;
        }
    $$ endfor
    
    // No valid sample found at any scale
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

fn sample_segmentations_vol_multi_scale(data_tex_coord: vec3<f32>, sizef: vec3<f32>) -> vec4<u32> {
    // Try scales from highest to lowest resolution (0 to num_scales-1)
    var result: vec4<u32>;
    $$ for iinrange(num_scales) result =try_sample_segmentations_scale_ { { i }
}(data_tex_coord, sizef);
    if result.w > 0 { // Valid sample found
        return result;
        }
    $$ endfor
    
    // No valid sample found at any scale
    return vec4<u32>(0, 0, 0, 0);
}

// Legacy single-scale functions for backward compatibility
fn sample_vol(data_tex_coord: vec3<f32>, sizef: vec3<f32>) -> vec4<f32> {
    $$ if num_scales > 1
        return sample_vol_multi_scale(data_tex_coord, sizef);
        $$ else
        return try_sample_scale_0(data_tex_coord, sizef);
        $$ endif
}

    fn sample_segmentations_vol(data_tex_coord,: vec3<f32>, sizef: vec3<f32>) -> vec4<u32>{
        $$ if num_scales > 1
            return sample_segmentations_vol_multi_scale(data_tex_coord, sizef);
            $$ else
            return try_sample_segmentations_scale_0(data_tex_coord, sizef);
            $$ endif
}