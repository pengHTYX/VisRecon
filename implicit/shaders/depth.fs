#version 330
in float pos_z;

layout(location = 0) out float frag_pos_z;

void main() {
    frag_pos_z = pos_z;
}
