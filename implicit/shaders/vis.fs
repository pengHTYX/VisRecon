#version 330

in float sh[9];

layout (location = 0) out vec3 coeff_0;
layout (location = 1) out vec3 coeff_1;
layout (location = 2) out vec3 coeff_2;

void main() {
    coeff_0 = vec3(sh[0], sh[1], sh[2]);
    coeff_1 = vec3(sh[3], sh[4], sh[5]);
    coeff_2 = vec3(sh[6], sh[7], sh[8]);
}
