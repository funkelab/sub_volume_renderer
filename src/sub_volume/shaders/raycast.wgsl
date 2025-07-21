struct RenderOutput {
    color: vec4<f32>,
    coord: vec3<f32>,
};

// most of the LMIP algorithim written by Claude Sonnet 4
// watch out for issues, and be skeptical of the accuracy of the code
fn raycast(sizef: vec3<f32>, nsteps: i32, start_coord: vec3<f32>, step_coord: vec3<f32>) -> RenderOutput {
    let nstepsf = f32(nsteps);

    // Fog parameters
    var fog_density: f32 = 9;
    var fog_color = vec4<f32>(0.5, 0.5, 0.5, 1.0);
    // Classic LMIP (Local Maximum Intensity Projection) algorithm
    // The minimum intensity threshold to consider a sample significant
    var lmip_threshold: f32 = 50;
    // The percent of the local maximum intensity that we consider still significant
    var fall_off_factor: f32 = 0.2;
    // Number of samples to check for local max
    var max_samples_after_threshold = 100000;
    var local_max_sample: vec4<f32> = vec4<f32>(0.0);
    var local_max_offset: vec3<f32>;
    var local_max_coord: vec3<f32>;
    var local_max_intensity = 0.0;
    var found_significant_value = false;
    var samples_since_threshold = 0;

    for (var iter = 0.0; iter < nstepsf; iter = iter + 1.0) {
        let offset = iter * step_coord;
        let coord = start_coord + offset;
        let sample = sample_vol(coord, sizef);
        let sample_intensity = length(sample.rgb); // Use RGB magnitude as intensity

        if !found_significant_value {
            // Look for first sample above threshold
            if sample_intensity >= lmip_threshold {
                found_significant_value = true;
                local_max_intensity = sample_intensity;
                local_max_sample = sample;
                local_max_offset = offset;
                local_max_coord = coord;
                samples_since_threshold = 0;
            }
        } else {
            // We've found a significant value, now find the local maximum
            samples_since_threshold += 1;

            // Update local maximum
            if sample_intensity > local_max_intensity {
                local_max_intensity = sample_intensity;
                local_max_sample = sample;
                local_max_offset = offset;
                local_max_coord = coord;
            }

            // Stop if we've sampled enough after threshold or intensity drops significantly
            if samples_since_threshold >= max_samples_after_threshold || sample_intensity < local_max_intensity * fall_off_factor {
                break;
            }
        }
    }

    // Create LMIP result - use the actual sample with maximum intensity
    var final_color: vec4<f32>;
    if found_significant_value {
        var distance = length(local_max_offset);
        var fog_factor = exp(-fog_density * distance);
        final_color = mix(fog_color, local_max_sample, fog_factor);
    } else {
        // No significant value found, return transparent
        final_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Colormapping
    let color = sampled_value_to_color(final_color);
    // Move to physical colorspace (linear photon count) so we can do math
    $$ if colorspace == 'srgb'
        let physical_color = srgb2physical(color.rgb);
        $$ else
        let physical_color = color.rgb;
        $$ endif
        let opacity = color.a * u_material.opacity;
        let out_color = vec4<f32>(physical_color, opacity);

    // Produce result
        var out: RenderOutput;
        out.color = out_color;
        out.coord = local_max_coord;
        return out;
}