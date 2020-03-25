#version 450

layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_color;

layout(location=0) out vec3 v_color;
layout(location=1) out vec3 v_normal;

layout(set=0, binding=0)
uniform Instances {
	mat4 s_models[20];
};

layout(set=0, binding=1)
uniform Uniforms {
	mat4 u_view_proj;
};

void main() {
	v_color = a_color;
	mat4 model = s_models[gl_InstanceIndex];
	v_normal = transpose(inverse(mat3(model))) * vec3(1.0, 0.0, 0.0);
	gl_Position = u_view_proj * model * vec4(a_position, 1.0);
}