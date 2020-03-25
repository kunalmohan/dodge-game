#version 450

layout(location=0) in vec3 v_color;
layout(location=1) in vec3 v_normal;

layout(location=0) out vec4 f_color;

layout(set=0, binding=2)
uniform Lights {
	vec3 u_light;
};

void main() {
	vec3 ambient = vec3(0.04, 0.04, 0.04);
	float diff = max(dot(normalize(v_normal), normalize(u_light)), 0.0);
	vec3 diffuse = vec3(diff, diff, diff);
	f_color = vec4((ambient + diffuse) * v_color, 1.0);
}