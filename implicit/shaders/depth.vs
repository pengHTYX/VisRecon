#version 330
uniform mat4 mv;
uniform mat4 mvp;

layout(location = 0) in vec3 in_pos;

out float pos_z;

void main() {
    pos_z = (mv * vec4(in_pos, 1.0)).z;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
