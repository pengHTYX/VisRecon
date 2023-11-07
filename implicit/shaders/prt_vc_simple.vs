#version 330

uniform mat4 mvp;

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_vert_color;
layout (location = 2) in float in_sh[9];

out vec3 vert_color;
out float sh[9];

void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    vert_color = in_vert_color;
    sh = in_sh;
}
