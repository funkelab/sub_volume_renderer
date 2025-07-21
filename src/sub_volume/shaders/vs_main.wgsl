struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
};


@vertex
fn vs_main(in: VertexInput) -> Varyings {

    // Our geometry is implicitly defined by the volume dimensions.
    var geo = get_vol_geometry();

    // Select what face we're at
    let index = i32(in.vertex_index);
    let i0 = geo.indices[index];

    // Sample position, and convert to world pos, and then to ndc
    let data_pos = vec4<f32>(geo.positions[i0], 1.0);
    let world_pos = u_wobject.world_transform * data_pos;
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    // Prepare inverse matrix
    let ndc_to_data = u_wobject.world_transform_inv * u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv;

    var varyings: Varyings;

    // Store values for fragment shader
    varyings.position = vec4<f32>(ndc_pos);
    varyings.world_pos = vec3<f32>(world_pos.xyz);

    // The position on the face of the cube. We can say that it's the back face,
    // because we cull the front faces.
    // These positions are in data positions (voxels) rather than texcoords (0..1),
    // because distances make more sense in this space. In the fragment shader we
    // can consider it an isotropic volume, because any position, rotation,
    // and scaling of the volume is part of the world transform.
    varyings.data_back_pos = vec4<f32>(data_pos);

    // We calculate the NDC positions for the near and front clipping planes,
    // and transform these back to data coordinates. From these positions
    // we can construct the view vector in the fragment shader, which is then
    // resistant to perspective transforms. It also makes that if the camera
    // is inside the volume, only the part in front in rendered.
    // Note that the w component for these positions should be left intact.
    let ndc_pos1 = vec4<f32>(ndc_pos.xy, -ndc_pos.w, ndc_pos.w);
    let ndc_pos2 = vec4<f32>(ndc_pos.xy, ndc_pos.w, ndc_pos.w);
    varyings.data_near_pos = vec4<f32>(ndc_to_data * ndc_pos1);
    varyings.data_far_pos = vec4<f32>(ndc_to_data * ndc_pos2);

    return varyings;
}
