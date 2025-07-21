{$ include 'sub_volume.sample_vol.wgsl' $}
{$ include 'sub_volume.hsv_selection.wgsl' $}

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {

    // clipping planes
        {$ include 'pygfx.clipping_planes.wgsl' $}

    // Get size of the volume
    let sizef = vec3<f32>(u_wobject.volume_dimensions);

    // Determine the stepsize as a float in pixels.
    // This value should be between ~ 0.1 and 1. Smaller values yield better
    // results at the cost of performance. With larger values you may miss
    // small structures (and corners of larger structures) because the step
    // may skip over them.
    // We could make this a user-facing property. But for now we scale between
    // 0.1 and 0.8 based on the (sqrt of the) volume size.
    let relative_step_size = clamp(sqrt(max(sizef.x, max(sizef.y, sizef.z))) / 20.0, 0.1, 0.8);

    // Positions in data coordinates
    let back_pos = varyings.data_back_pos.xyz / varyings.data_back_pos.w;
    let far_pos = varyings.data_far_pos.xyz / varyings.data_far_pos.w;
    let near_pos = varyings.data_near_pos.xyz / varyings.data_near_pos.w;

    // Calculate unit vector pointing in the view direction through this fragment.
    let view_ray = normalize(far_pos - near_pos);

    // Calculate the (signed) distance, from back_pos to the first voxel
    // that must be sampled, expressed in data coords (voxels).
    var dist = dot(near_pos - back_pos, view_ray);
    dist = max(dist, min((-0.5 - back_pos.x) / view_ray.x, (sizef.x - 0.5 - back_pos.x) / view_ray.x));
    dist = max(dist, min((-0.5 - back_pos.y) / view_ray.y, (sizef.y - 0.5 - back_pos.y) / view_ray.y));
    dist = max(dist, min((-0.5 - back_pos.z) / view_ray.z, (sizef.z - 0.5 - back_pos.z) / view_ray.z));

    // Now we have the starting position. This is typically on a front face,
    // but it can also be incide the volume (on the near plane).
    let front_pos = back_pos + view_ray * dist;

    // Decide how many steps to take. If we'd not cul the front faces,
    // that would still happen here because nsteps would be negative.
    let nsteps = i32(-dist / relative_step_size + 0.5);
    if nsteps < 1 { discard; }

    // Get starting position and step vector in texture coordinates.
    let start_coord = (front_pos + vec3<f32>(0.5, 0.5, 0.5)) / sizef;
    let step_coord = ((back_pos - front_pos) / sizef) / f32(nsteps);

    // Render
    let render_out = raycast(sizef, nsteps, start_coord, step_coord);

    // Create fragment output.
    var out: FragmentOutput;

    if render_out.found {
        // render_out.coord is already in data coordinates, so we do not need to scale by sizef.
        // However, we need to subtract 0.5 to get to the center. When we sample (1,1,1), we are sampling from the corner
        // of the voxel. Our depth should deal with the center of the voxel, so we subtract half a unit to get to the
        // center.
        let data_pos = render_out.coord - vec3<f32>(0.5, 0.5, 0.5);
        let world_pos = u_wobject.world_transform * vec4<f32>(data_pos, 1.0);
        let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
        // ndc_pos.{x, y} contains the ndc pos on the screen, so this should match with the ndc_pos of this fragment shader
        // invocation. ndc_pos.z contains some distance value? And then ndc_pos.w encodes info about the perspective.
        // Note that even though we are using homogenous coordinates, w is *not* 1.0 since the projection transform hijacks
        // it to encode the perspective.
        // In case we sample from too close to the near plane, w becomes small enough to cause a division by zero.
        // To avoid this, we clamp w to a minimum of 0.001. I'm not too sure if this is the best way to do this,
        // but it stays within the intention of the math for the volume renderer. An alternative might be to return the
        // offset used to sample from the texture, but that would require a different calculation from the current one.
        let depth: f32 = ndc_pos.z / max(ndc_pos.w, 0.001);

        let i = render_out.segmentation;
        let n = u_wobject.max_segmentation_value;
        let hsv: vec3<f32> = vec3<f32>(sample_hs_color(i, n), render_out.color.r);
        let rgb: vec3<f32> = hsv_to_rgb(hsv);

        let fog_density: f32 = u_material.fog_density;
        let fog_color: vec3<f32> = u_material.fog_color;

        let offset = render_out.offset;
        let distance = length(offset);
        let fog_factor = exp(-fog_density * distance);
        let fogged_intensity = mix(fog_color, rgb, fog_factor);

        out.color = vec4<f32>(fogged_intensity, u_material.opacity);
        out.depth = depth;

        $$ if write_pick
            // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
            out.pick = (pick_pack(u32(u_wobject.id), 20) + pick_pack(u32(render_out.coord.x * 16383.0), 14) + pick_pack(u32(render_out.coord.y * 16383.0), 14) + pick_pack(u32(render_out.coord.z * 16383.0), 14));
            $$ endif
    } else {
        // If no significant value is found, we return black
            out.color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        // Not sure what a default depth should be
            out.depth = 0.0;
        }

        return out;
    }
