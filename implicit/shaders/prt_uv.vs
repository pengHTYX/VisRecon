#version 330

uniform mat4 mvp;

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec2 in_uv;
layout (location = 2) in float in_sh[9];

out vec2 uv;
out float sh[9];

void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    uv = in_uv;
    sh = in_sh;
}
