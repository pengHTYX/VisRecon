#version 330
uniform mat4 mv;
uniform mat4 mvp;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_vert_normal;

out vec3 local_vert_normal;
out vec3 vert_normal;

void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    local_vert_normal = mat3(mv) * in_vert_normal;
    vert_normal = in_vert_normal;
}
