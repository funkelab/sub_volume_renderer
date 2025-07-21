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

    return result;
}

//
// fn sample_vol(
//     x: world coordinate
//     // this tells us how to wrap
//     offset: where it wraps
//     // defines where the texture is (this is the Roi)
//     b: start of the texture
//     size: size of texture
// ) -> vec4<f32>;
//
