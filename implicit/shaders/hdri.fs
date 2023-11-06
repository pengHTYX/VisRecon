#version 330 core
layout(location = 0) out vec4 hdri;

uniform float env_sh[27];
uniform mat3 R;

in vec2 uv;

#define PI 3.1415926538
#define k01 0.2820947918  // sqrt(1 / PI) / 2
#define k02 0.4886025119  // sqrt(3 / (4 * PI))
#define k03 1.0925484306  // sqrt(15 / PI) / 2
#define k04 0.3153915652  // sqrt(5 / PI) / 4
#define k05 0.5462742153  // sqrt(15 / PI) / 4

float SH_00() {
  return k01;
}
float SH_1_1(in vec3 s) {
  return k02 * s.y;
}
float SH_10(in vec3 s) {
  return k02 * s.z;
}
float SH_11(in vec3 s) {
  return k02 * s.x;
}
float SH_2_2(in vec3 s) {
  return k03 * s.x * s.y;
}
float SH_2_1(in vec3 s) {
  return k03 * s.y * s.z;
}
float SH_20(in vec3 s) {
  return k04 * (3.0 * s.z * s.z - 1.0);
}
float SH_21(in vec3 s) {
  return k03 * s.x * s.z;
}
float SH_22(in vec3 s) {
  return k05 * (s.x * s.x - s.y * s.y);
}

void main() {
  float phi = uv.y * PI;
  float theta = 2.0 * (1.0 - uv.x) * PI + PI;
  vec3 dir = vec3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
  dir = dir.xzy;
  dir = R * dir;

  float sh_basis[9];
  sh_basis[0] = SH_00();
  sh_basis[1] = SH_1_1(dir);
  sh_basis[2] = SH_10(dir);
  sh_basis[3] = SH_11(dir);
  sh_basis[4] = SH_2_2(dir);
  sh_basis[5] = SH_2_1(dir);
  sh_basis[6] = SH_20(dir);
  sh_basis[7] = SH_21(dir);
  sh_basis[8] = SH_22(dir);

  hdri = vec4(0.0);
  for (int i = 0; i < 9; i++) {
    hdri.x += sh_basis[i] * env_sh[i];
    hdri.y += sh_basis[i] * env_sh[9 + i];
    hdri.z += sh_basis[i] * env_sh[18 + i];
  }
  hdri.w = 1.0;
}
