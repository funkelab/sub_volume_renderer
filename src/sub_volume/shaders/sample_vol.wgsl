fn sample_vol(data_tex_coord: vec3<f32>, sizef: vec3<f32>) -> vec4<f32> {
    let data_coord = data_tex_coord * sizef;
    let ring_buffer_dimensions = vec3<f32>(textureDimensions(t_ring_buffer));

    let offset = u_wrapping_buffer.current_logical_offset_in_pixels;
    let shape = u_wrapping_buffer.current_logical_shape_in_pixels;

    let in_bounds = all(offset <= vec3<i32>(data_coord)) && all(vec3<i32>(data_coord) < offset + shape);
    if !in_bounds {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let wrapped_data_coord = data_coord % ring_buffer_dimensions;
    let result = textureLoad(t_ring_buffer, vec3<i32>(wrapped_data_coord), 0);

    return vec4<f32>(result);
}

// We need to sample from a different texture to get the segmentations
// Passing in the texture as a parameter doesn't work because it is of type texture_3d<u32>, but the other texture
// is of type texture_3d<f32>. The WGSL type system constrains us here so we just define a seperate function. Another
// way to pass the texture as a parameter would be to define an overload which would get called.
fn sample_segmentations_vol(data_tex_coord: vec3<f32>, sizef: vec3<f32>) -> vec4<u32> {
    let data_coord = data_tex_coord * sizef;
    let ring_buffer_dimensions = vec3<f32>(textureDimensions(t_segmentations_ring_buffer));

    let offset = u_wrapping_buffer.current_logical_offset_in_pixels;
    let shape = u_wrapping_buffer.current_logical_shape_in_pixels;

    // Bounds check shouldn't be needed since this should only be called on valid points from sample_vol
    let in_bounds = all(offset <= vec3<i32>(data_coord)) && all(vec3<i32>(data_coord) < offset + shape);
    if !in_bounds {
        return vec4<u32>(0, 0, 0, 1);
    }

    let wrapped_data_coord = data_coord % ring_buffer_dimensions;
    let result = textureLoad(t_segmentations_ring_buffer, vec3<i32>(wrapped_data_coord), 0);

    return vec4<u32>(result);
}

