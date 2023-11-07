#version 330

uniform float env_sh[27];

in vec3 vert_color;
in float sh[9];

layout (location = 0) out vec4 light_color;
layout (location = 1) out vec4 textured_color;
layout (location = 2) out vec4 albedo_color;

vec4 gammaCorrection(vec4 vec, float g) {
    return vec4(pow(vec.x, 1.0/g), pow(vec.y, 1.0/g), pow(vec.z, 1.0/g), vec.w);
}

void main() {
    light_color = vec4(0.0);
    for (int i = 0; i < 9; i++) {
        light_color.x += sh[i] * env_sh[i];
        light_color.y += sh[i] * env_sh[9 + i];
        light_color.z += sh[i] * env_sh[18 + i];
    }
    light_color.w = 1.0;
    light_color = gammaCorrection(light_color, 2.2);
    textured_color = light_color * vec4(vert_color, 1.0);
    albedo_color = vec4(vert_color, 1.0);
}
