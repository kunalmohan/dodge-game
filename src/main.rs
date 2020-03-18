use winit::{
    event_loop::{EventLoop, ControlFlow},
    event::*,
    window::Window,
    dpi::PhysicalSize,
};
use std::time::{Instant, Duration};
use cgmath::prelude::*;

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

const NUM_INSTANCES_PER_ROW: u32 = 5;
const NUM_INSTANCES: u32 = NUM_INSTANCES_PER_ROW * NUM_INSTANCES_PER_ROW;
const INSTANCE_DISPLACEMENT: cgmath::Vector3<f32> = cgmath::Vector3::new(NUM_INSTANCES_PER_ROW as f32, 0.0, NUM_INSTANCES_PER_ROW as f32);

pub const INDICES: &[u16] = &[
    //Top
    0, 2, 1,
    1, 2, 3,
    //Right
    3, 5, 4,
    4, 1, 3,
    //Front
    3, 2, 5,
    5, 2, 7,
    //Left
    7, 2 ,6,
    6, 2, 0,
    //Back
    0, 1, 4,
    4, 6, 0,
    //Bottom
    6, 4, 5,
    5, 7, 6,
];

fn main() {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();

    let time = Instant::now() + Duration::from_millis(400);

    let mut state = State::new(&window);

    let id = window.id();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == id => if state.input(event) {
                *control_flow = ControlFlow::WaitUntil(time);
            } else {
                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::KeyboardInput {
                        input: i,
                        ..
                    } => match i {
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        _ => *control_flow = ControlFlow::WaitUntil(time),
                    },
                    _ => *control_flow = ControlFlow::WaitUntil(time),
                }
            }
            Event::MainEventsCleared => {
            	state.update();
            	state.render();
                *control_flow = ControlFlow::WaitUntil(time);
            },
            _ => *control_flow = ControlFlow::WaitUntil(time),
        }
    });
}

struct State {
	device: wgpu::Device,
	swap_chain: wgpu::SwapChain,
	queue: wgpu::Queue,
	player_render_pipeline: wgpu::RenderPipeline,
	obstacle_render_pipeline: wgpu::RenderPipeline,
	obstacle_bind_group: wgpu::BindGroup,
	index_buffer: wgpu::Buffer,
	player_vertex_buffer: wgpu::Buffer,
	obstacle_vertex_buffer: wgpu::Buffer,
	player_controller: PlayerController,
	uniform_bind_group: wgpu::BindGroup,
	player_position: PlayerPosition,
	position_buffer: wgpu::Buffer,
	instances: Vec<Instance>,
}

impl State {
	fn new(window: &Window) -> Self {
		let inner_size = window.inner_size();
		let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions{
			..Default::default()
		}).unwrap();
		let surface = wgpu::Surface::create(window);

		let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: Default::default(),
        });

        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: inner_size.width,
            height: inner_size.height,
            present_mode: wgpu::PresentMode::Vsync,
        };

        let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        let player = Block {
        	vertices: [
        		Vertex { position: [-0.5, 0.3, 0.7], color: [0.2, 0.2, 0.9] },
        		Vertex { position: [0.5, 0.3, 0.7], color: [0.2, 0.2, 0.9] },
        		Vertex { position: [-0.5, 0.3, -0.7], color: [0.2, 0.2, 0.9] },
        		Vertex { position: [0.5, 0.3, -0.7], color: [0.2, 0.2, 0.9] },
        		Vertex { position: [0.5, -0.3, 0.7], color: [0.2, 0.2, 0.9] },
        		Vertex { position: [0.5, -0.3, -0.7], color: [0.2, 0.2, 0.9] },
        		Vertex { position: [-0.5, -0.3, 0.7], color: [0.2, 0.2, 0.9] },
        		Vertex { position: [-0.5, -0.3, -0.7], color: [0.2, 0.2, 0.9] },
        	],
        };

        let obstacle = Block {
        	vertices: [
        		Vertex { position: [-0.2, 0.2, 0.2], color: [0.6, 0.6, 0.9] },
        		Vertex { position: [0.2, 0.2, 0.2], color: [0.6, 0.6, 0.9] },
        		Vertex { position: [-0.2, 0.2, -0.2], color: [0.6, 0.6, 0.9] },
        		Vertex { position: [0.2, 0.2, -0.2], color: [0.6, 0.6, 0.9] },
        		Vertex { position: [0.2, -0.2, 0.2], color: [0.6, 0.6, 0.9] },
        		Vertex { position: [0.2, -0.2, -0.2], color: [0.6, 0.6, 0.9] },
        		Vertex { position: [-0.2, -0.2, 0.2], color: [0.6, 0.6, 0.9] },
        		Vertex { position: [-0.2, -0.2, -0.2], color: [0.6, 0.6, 0.9] },
        	]
        };

        let instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|z| {
		    (0..NUM_INSTANCES_PER_ROW).map(move |x| {
		        let position = cgmath::Vector3 { x: x as f32, y: 0.0, z: z as f32 } - INSTANCE_DISPLACEMENT;
		
		        let rotation = if position.is_zero() {
		            cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
		        } else {
		            cgmath::Quaternion::from_axis_angle(position.clone().normalize(), cgmath::Deg(45.0))
		        };
		
		        Instance {
		            position, rotation,
		        }
		    })
		}).collect::<Vec<_>>();

		let instance_data = instances.iter().map(Instance::to_matrix).collect::<Vec<_>>();
		let instance_buffer_size = instance_data.len() * std::mem::size_of::<cgmath::Matrix4<f32>>();
		let instance_buffer = device.create_buffer_mapped(instance_data.len(), wgpu::BufferUsage::STORAGE_READ).fill_from_slice(&instance_data);

        let player_position = PlayerPosition {
        	position: 0.0f32,
        };

        let position_buffer = device.create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST).fill_from_slice(&[player_position]);

        let player_controller = PlayerController::new(0.3);

        let camera = Camera {
    	    eye: (0.0, 6.0, -4.0).into(),
    	    target: (0.0, 0.0, 5.0).into(),
    	    up: cgmath::Vector3::unit_y(),
    	    aspect: sc_desc.width as f32 / sc_desc.height as f32,
    	    fovy: 75.0,
    	    znear: 0.1,
    	    zfar: 100.0,
    	};

    	let obstacle_vertex_buffer = device.create_buffer_mapped(obstacle.vertices.len(), wgpu::BufferUsage::VERTEX).fill_from_slice(&obstacle.vertices);

        let player_vertex_buffer = device.create_buffer_mapped(player.vertices.len(), wgpu::BufferUsage::VERTEX).fill_from_slice(&player.vertices);

        let index_buffer = device.create_buffer_mapped(INDICES.len(), wgpu::BufferUsage::INDEX).fill_from_slice(INDICES);

        let mut player_uniforms = Uniforms::new();
        player_uniforms.update_view_proj(&camera);

        let player_uniform_buffer = device.create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST).fill_from_slice(&[player_uniforms]);

        let mut obstacle_uniforms = Uniforms::new();
        obstacle_uniforms.update_view_proj(&camera);

        let obstacle_uniform_buffer = device.create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM).fill_from_slice(&[obstacle_uniforms]);

        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        	bindings: &[
        		wgpu::BindGroupLayoutBinding {
        			binding: 0,
        			visibility: wgpu::ShaderStage::VERTEX,
        			ty: wgpu::BindingType::UniformBuffer {
        				dynamic: false,
        			},
        		},
        		wgpu::BindGroupLayoutBinding {
        			binding: 1,
        			visibility: wgpu::ShaderStage::VERTEX,
        			ty: wgpu::BindingType::UniformBuffer {
        				dynamic: false,
        			},
        		},
        	],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        	layout: &uniform_bind_group_layout,
        	bindings: &[
        		wgpu::Binding {
        			binding: 0,
        			resource: wgpu::BindingResource::Buffer {
        				buffer: &player_uniform_buffer,
        				range: 0..std::mem::size_of_val(&player_uniforms) as wgpu::BufferAddress,
        			},
        		},
        		wgpu::Binding {
        			binding: 1,
        			resource: wgpu::BindingResource::Buffer {
        				buffer: &position_buffer,
        				range: 0..std::mem::size_of_val(&player_position) as wgpu::BufferAddress,
        			},
        		},
        	],
        });

        let obstacle_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        	bindings: &[
        		wgpu::BindGroupLayoutBinding {
        			binding: 0,
        			visibility: wgpu::ShaderStage::VERTEX,
        			ty: wgpu::BindingType::StorageBuffer {
        				dynamic: false,
        				readonly: true,
        			},
        		},
        		wgpu::BindGroupLayoutBinding {
        			binding: 1,
        			visibility: wgpu::ShaderStage::VERTEX,
        			ty: wgpu::BindingType::UniformBuffer {
        				dynamic: false,
        			},
        		},
        	],
        });

        let obstacle_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        	layout: &obstacle_bind_group_layout,
        	bindings: &[
        		wgpu::Binding {
        			binding: 0,
        			resource: wgpu::BindingResource::Buffer {
        				buffer: &instance_buffer,
        				range: 0..instance_buffer_size as wgpu::BufferAddress,
        			},
        		},
        		wgpu::Binding {
        			binding: 1,
        			resource: wgpu::BindingResource::Buffer {
        				buffer: &obstacle_uniform_buffer,
        				range: 0..std::mem::size_of_val(&obstacle_uniforms) as wgpu::BufferAddress,
        			},
        		},
        	],
        });

        let vs_src = include_str!("shader.vert");
        let fs_src = include_str!("shader.frag");
        let vs_src2 = include_str!("shader2.vert");

        let vs_spriv = glsl_to_spirv::compile(vs_src, glsl_to_spirv::ShaderType::Vertex).unwrap();
        let fs_spirv = glsl_to_spirv::compile(fs_src, glsl_to_spirv::ShaderType::Fragment).unwrap();
        let vs_spriv2 = glsl_to_spirv::compile(vs_src2, glsl_to_spirv::ShaderType::Vertex).unwrap();

        let vs_data = wgpu::read_spirv(vs_spriv).unwrap();
        let vs_data2 = wgpu::read_spirv(vs_spriv2).unwrap();
        let fs_data = wgpu::read_spirv(fs_spirv).unwrap();

        let vs_module = device.create_shader_module(&vs_data);
        let vs_module2 = device.create_shader_module(&vs_data2);
        let fs_module = device.create_shader_module(&fs_data);

        let player_render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        	bind_group_layouts: &[&uniform_bind_group_layout],
        });

        let player_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        	layout: &player_render_pipeline_layout,
        	vertex_stage: wgpu::ProgrammableStageDescriptor {
        		module: &vs_module,
        		entry_point: "main",
        	},
        	fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
        		module: &fs_module,
        		entry_point: "main",
        	}),
        	rasterization_state: Some(wgpu::RasterizationStateDescriptor {
        		front_face: wgpu::FrontFace::Ccw,
        		cull_mode: wgpu::CullMode::None,
        		depth_bias: 0,
        		depth_bias_slope_scale: 0.0,
        		depth_bias_clamp: 0.0,
        	}),
        	primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        	color_states: &[
        		wgpu::ColorStateDescriptor {
        			format: wgpu::TextureFormat::Bgra8UnormSrgb,
        			alpha_blend: wgpu::BlendDescriptor::REPLACE,
        			color_blend: wgpu::BlendDescriptor::REPLACE,
        			write_mask: wgpu::ColorWrite::ALL,
        		},
        	],
        	depth_stencil_state: None,
        	index_format: wgpu::IndexFormat::Uint16,
        	vertex_buffers: &[
        		Vertex::desc(),
        	],
        	sample_count: 1,
        	sample_mask: !0,
        	alpha_to_coverage_enabled: false,
        });

        let obstacle_render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        	bind_group_layouts: &[&obstacle_bind_group_layout],
        });

        let obstacle_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        	layout: &obstacle_render_pipeline_layout,
        	vertex_stage: wgpu::ProgrammableStageDescriptor {
        		module: &vs_module2,
        		entry_point: "main",
        	},
        	fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
        		module: &fs_module,
        		entry_point: "main",
        	}),
        	rasterization_state: Some(wgpu::RasterizationStateDescriptor {
        		front_face: wgpu::FrontFace::Ccw,
        		cull_mode: wgpu::CullMode::None,
        		depth_bias: 0,
        		depth_bias_slope_scale: 0.0,
        		depth_bias_clamp: 0.0,
        	}),
        	primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        	color_states: &[
        		wgpu::ColorStateDescriptor {
        			format: wgpu::TextureFormat::Bgra8UnormSrgb,
        			alpha_blend: wgpu::BlendDescriptor::REPLACE,
        			color_blend: wgpu::BlendDescriptor::REPLACE,
        			write_mask: wgpu::ColorWrite::ALL,
        		},
        	],
        	depth_stencil_state: None,
        	index_format: wgpu::IndexFormat::Uint16,
        	vertex_buffers: &[
        		Vertex::desc(),
        	],
        	sample_count: 1,
        	sample_mask: !0,
        	alpha_to_coverage_enabled: false,
        });

        Self {
        	device,
        	queue,
        	swap_chain,
        	player_render_pipeline,
        	obstacle_render_pipeline,
        	index_buffer,
        	player_vertex_buffer,
        	obstacle_vertex_buffer,
        	uniform_bind_group,
        	obstacle_bind_group,
        	player_controller,
        	player_position,
        	position_buffer,
        	instances,
        }
	}

	fn input(&mut self ,event: &WindowEvent) -> bool {
		self.player_controller.control(event)
	}

	fn update(&mut self) {
		self.player_controller.update_player_position(&mut self.player_position);

		let tmp_buffer = self.device.create_buffer_mapped(1, wgpu::BufferUsage::COPY_SRC).fill_from_slice(&[self.player_position]);

		let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			todo: 0,
		});

		encoder.copy_buffer_to_buffer(&tmp_buffer, 0, &self.position_buffer, 0, std::mem::size_of::<f32>() as wgpu::BufferAddress);

		self.queue.submit(&[encoder.finish()]);
	}

	fn render(&mut self) {
		let frame = self.swap_chain.get_next_texture();

		let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			todo: 0,
		});

		{
			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				color_attachments: &[
					wgpu::RenderPassColorAttachmentDescriptor {
						attachment: &frame.view,
						resolve_target: None,
						load_op: wgpu::LoadOp::Clear,
						store_op: wgpu::StoreOp::Store,
						clear_color: wgpu::Color::WHITE,
					},
				],
				depth_stencil_attachment: None,
			});

			render_pass.set_pipeline(&self.obstacle_render_pipeline);
			render_pass.set_vertex_buffers(0, &[(&self.obstacle_vertex_buffer, 0)]);
			render_pass.set_index_buffer(&self.index_buffer, 0);
			render_pass.set_bind_group(0, &self.obstacle_bind_group, &[]);
			render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..self.instances.len() as u32);
		}

		{
			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				color_attachments: &[
					wgpu::RenderPassColorAttachmentDescriptor {
						attachment: &frame.view,
						resolve_target: None,
						load_op: wgpu::LoadOp::Load,
						store_op: wgpu::StoreOp::Store,
						clear_color: wgpu::Color::WHITE,
					},
				],
				depth_stencil_attachment: None,
			});

			render_pass.set_pipeline(&self.player_render_pipeline);
			render_pass.set_vertex_buffers(0, &[(&self.player_vertex_buffer, 0)]);
			render_pass.set_index_buffer(&self.index_buffer, 0);
			render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
			render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
		}

		self.queue.submit(&[encoder.finish()]);
	}
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Vertex {
	position: [f32; 3],
	color: [f32; 3],
}

impl Vertex {
	fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
		wgpu::VertexBufferDescriptor {
			stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
			step_mode: wgpu::InputStepMode::Vertex,
			attributes: &[
				wgpu::VertexAttributeDescriptor {
					offset: 0,
					format: wgpu::VertexFormat::Float3,
					shader_location: 0,
				},
				wgpu::VertexAttributeDescriptor {
					offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
					format: wgpu::VertexFormat::Float3,
					shader_location: 1,
				},
			],
		}
	}
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Block {
	vertices: [Vertex; 8],
}

#[derive(Debug)]
struct PlayerController {
	sensitivity: f32,
	left: bool,
	right: bool,
}

impl PlayerController {
	fn new(sensitivity: f32) -> Self {
		Self {
			sensitivity,
			left: false,
			right: false,
		}
	}

	fn control(&mut self, event: &WindowEvent) -> bool {
		match event {
			WindowEvent::KeyboardInput {
				input: KeyboardInput {
					state,
					virtual_keycode: Some(key_code),
					..
				},
				..
			} => {
				let pressed = *state == ElementState::Pressed;
				match key_code {
					VirtualKeyCode::Left => {
						self.left = pressed;
						true
					},
					VirtualKeyCode::Right => {
						self.right = pressed;
						true
					}
					_ => false,
				}
			},
			_ => false,
		}
	}

	fn update_player_position(&self, player_position: &mut PlayerPosition) {
		if self.right {
			if player_position.position > -3.0 {
				player_position.position -= self.sensitivity;
			}
		}
		else if self.left {
			if player_position.position < 3.0 {
				player_position.position += self.sensitivity;
			}
		}
	}
}

struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Uniforms {
	view_proj: cgmath::Matrix4<f32>,
}

impl Uniforms {
	fn new() -> Self {
		Self {
			view_proj: cgmath::Matrix4::identity(),
		}
	}

	fn update_view_proj(&mut self, camera: &Camera) {
		self.view_proj = camera.build_view_projection_matrix();
	}
}

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

impl Instance {
    fn to_matrix(&self) -> cgmath::Matrix4<f32> {
        cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct PlayerPosition {
	position: f32,
}