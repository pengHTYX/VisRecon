#version 330

uniform mat4 mvp;

layout (location = 0) in vec3 in_pos;
layout (location = 1) in float in_sh[9];

out float sh[9];

void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    sh = in_sh;
}
