fn sample_hs_color(i: u32) -> vec2<f32> {
    return u_material.colors[i % u_material.color_count].rg;
}


// Written by Claude Sonnet 4
fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x;
    let s = hsv.y;
    let v = hsv.z;

    // If saturation is 0, it's grayscale
    if s == 0.0 {
        return vec3<f32>(v, v, v);
    }

    // Scale hue to 0-6 range and find which sector we're in
    let h_scaled = h * 6.0;
    let sector = i32(floor(h_scaled));
    let fractional = h_scaled - floor(h_scaled);

    // Calculate intermediate values
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * fractional);
    let t = v * (1.0 - s * (1.0 - fractional));

    // Return RGB based on which sector of the color wheel
    if sector == 0 {
        return vec3<f32>(v, t, p);  // Red to Yellow
    } else if sector == 1 {
        return vec3<f32>(q, v, p);  // Yellow to Green
    } else if sector == 2 {
        return vec3<f32>(p, v, t);  // Green to Cyan
    } else if sector == 3 {
        return vec3<f32>(p, q, v);  // Cyan to Blue
    } else if sector == 4 {
        return vec3<f32>(t, p, v);  // Blue to Magenta
    } else {
        return vec3<f32>(v, p, q);  // Magenta to Red
    }
}
