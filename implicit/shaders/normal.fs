#version 330

in vec3 local_vert_normal;
in vec3 vert_normal;

layout(location = 0) out vec3 frag_local_normal;
layout(location = 1) out vec3 frag_normal;

void main() {
    frag_local_normal = local_vert_normal;
    frag_normal = vert_normal;
}
