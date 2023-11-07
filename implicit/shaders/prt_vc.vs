#version 330

uniform mat4 vp;
uniform mat4 model;

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_vert_color;
layout (location = 2) in float in_sh[9];

out vec3 vert_color;
out float sh[9];

float s_c3 = 0.94617469575; // (3*sqrt(5))/(4*sqrt(pi))
float s_c4 = -0.31539156525;// (-sqrt(5))/(4*sqrt(pi))
float s_c5 = 0.54627421529; // (sqrt(15))/(4*sqrt(pi))

float s_c_scale = 1.0/0.91529123286551084;
float s_c_scale_inv = 0.91529123286551084;

float s_rc2 = 1.5853309190550713*s_c_scale;
float s_c4_div_c3 = s_c4/s_c3;
float s_c4_div_c3_x2 = (s_c4/s_c3)*2.0;

float s_scale_dst2 = s_c3 * s_c_scale_inv;
float s_scale_dst4 = s_c5 * s_c_scale_inv;

void OptRotateBand0(float x[1], mat3 R, out float dst[1])
{
    dst[0] = x[0];
}

// 9 multiplies
void OptRotateBand1(float x[3], mat3 R, out float dst[3])
{
    // derived from  SlowRotateBand1
    dst[0] = ( R[1][1])*x[0] + (-R[1][2])*x[1] + ( R[1][0])*x[2];
    dst[1] = (-R[2][1])*x[0] + ( R[2][2])*x[1] + (-R[2][0])*x[2];
    dst[2] = ( R[0][1])*x[0] + (-R[0][2])*x[1] + ( R[0][0])*x[2];
}

// 48 multiplies
void OptRotateBand2(float x[5], mat3 R, out float dst[5])
{
    // Sparse matrix multiply
    float sh0 =  x[3] + x[4] + x[4] - x[1];
    float sh1 =  x[0] + s_rc2*x[2] +  x[3] + x[4];
    float sh2 =  x[0];
    float sh3 = -x[3];
    float sh4 = -x[1];

    // Rotations.  R0 and R1 just use the raw matrix columns
    float r2x = R[0][0] + R[0][1];
    float r2y = R[1][0] + R[1][1];
    float r2z = R[2][0] + R[2][1];

    float r3x = R[0][0] + R[0][2];
    float r3y = R[1][0] + R[1][2];
    float r3z = R[2][0] + R[2][2];

    float r4x = R[0][1] + R[0][2];
    float r4y = R[1][1] + R[1][2];
    float r4z = R[2][1] + R[2][2];

    // dense matrix multiplication one column at a time

    // column 0
    float sh0_x = sh0 * R[0][0];
    float sh0_y = sh0 * R[1][0];
    float d0 = sh0_x * R[1][0];
    float d1 = sh0_y * R[2][0];
    float d2 = sh0 * (R[2][0] * R[2][0] + s_c4_div_c3);
    float d3 = sh0_x * R[2][0];
    float d4 = sh0_x * R[0][0] - sh0_y * R[1][0];

    // column 1
    float sh1_x = sh1 * R[0][2];
    float sh1_y = sh1 * R[1][2];
    d0 += sh1_x * R[1][2];
    d1 += sh1_y * R[2][2];
    d2 += sh1 * (R[2][2] * R[2][2] + s_c4_div_c3);
    d3 += sh1_x * R[2][2];
    d4 += sh1_x * R[0][2] - sh1_y * R[1][2];

    // column 2
    float sh2_x = sh2 * r2x;
    float sh2_y = sh2 * r2y;
    d0 += sh2_x * r2y;
    d1 += sh2_y * r2z;
    d2 += sh2 * (r2z * r2z + s_c4_div_c3_x2);
    d3 += sh2_x * r2z;
    d4 += sh2_x * r2x - sh2_y * r2y;

    // column 3
    float sh3_x = sh3 * r3x;
    float sh3_y = sh3 * r3y;
    d0 += sh3_x * r3y;
    d1 += sh3_y * r3z;
    d2 += sh3 * (r3z * r3z + s_c4_div_c3_x2);
    d3 += sh3_x * r3z;
    d4 += sh3_x * r3x - sh3_y * r3y;

    // column 4
    float sh4_x = sh4 * r4x;
    float sh4_y = sh4 * r4y;
    d0 += sh4_x * r4y;
    d1 += sh4_y * r4z;
    d2 += sh4 * (r4z * r4z + s_c4_div_c3_x2);
    d3 += sh4_x * r4z;
    d4 += sh4_x * r4x - sh4_y * r4y;

    // extra multipliers
    dst[0] = d0;
    dst[1] = -d1;
    dst[2] = d2 * s_scale_dst2;
    dst[3] = -d3;
    dst[4] = d4 * s_scale_dst4;
}

void main() {
    mat3 R = mat3(model);
    float PRT0, PRT1[3], PRT2[5];
    PRT0 = in_sh[0];
    PRT1[0] = in_sh[1];
    PRT1[1] = in_sh[2];
    PRT1[2] = in_sh[3];
    PRT2[0] = in_sh[4];
    PRT2[1] = in_sh[5];
    PRT2[2] = in_sh[6];
    PRT2[3] = in_sh[7];
    PRT2[4] = in_sh[8];

    OptRotateBand1(PRT1, R, PRT1);
    OptRotateBand2(PRT2, R, PRT2);

    gl_Position = vp * vec4(in_pos, 1.0);
    vert_color = in_vert_color;

    sh[0] = PRT0;
    sh[1] = PRT1[0];
    sh[2] = PRT1[1];
    sh[3] = PRT1[2];
    sh[4] = PRT2[0];
    sh[5] = PRT2[1];
    sh[6] = PRT2[2];
    sh[7] = PRT2[3];
    sh[8] = PRT2[4];

    // sh = in_sh;
}
