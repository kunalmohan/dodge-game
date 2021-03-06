use winit::{
    event_loop::{EventLoop, ControlFlow},
    event::*,
    window::Window,
    dpi::PhysicalSize,
};
use std::time::{Instant, Duration};
use cgmath::prelude::*;
use wgpu_glyph::{Section, GlyphBrushBuilder};
use std::thread::sleep;

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

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
            	state.check_collision();
            	state.update_instances();
            	state.update();
            	state.render();
                *control_flow = ControlFlow::WaitUntil(time);
                if state.collided {
                    state.render_finish();
                    sleep(Duration::from_millis(5000));
                    *control_flow = ControlFlow::Exit;
                }
            },
            _ => *control_flow = ControlFlow::WaitUntil(time),
        }
    });
}

struct State<'a> {
	device: wgpu::Device,
	swap_chain: wgpu::SwapChain,
	queue: wgpu::Queue,
	player_render_pipeline: wgpu::RenderPipeline,
	obstacle_render_pipeline: wgpu::RenderPipeline,
	road_render_pipeline: wgpu::RenderPipeline,
	obstacle_bind_group: wgpu::BindGroup,
	player_vertex_buffer: wgpu::Buffer,
	obstacle_vertex_buffer: wgpu::Buffer,
	player_controller: PlayerController,
	uniform_bind_group: wgpu::BindGroup,
	player_position: PlayerPosition,
	position_buffer: wgpu::Buffer,
	instances: Vec<Instance>,
	speed: f32,
	instance_buffer: wgpu::Buffer,
	road_vertex_buffer: wgpu::Buffer,
	instant: Instant,
	num_road: u32,
	num_player: u32,
	num_obstacle: u32,
    score: u32,
    font: &'a[u8],
    inner_size: PhysicalSize<u32>,
    collided: bool,
    multisampled_framebuffer: wgpu::TextureView,
}

impl State<'_> {
	fn new(window: &Window) -> Self {
		let instant = Instant::now();
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

        let multisampled_framebuffer = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: inner_size.width,
                height: inner_size.height,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 8,
            dimension: wgpu::TextureDimension::D2,
            format: sc_desc.format,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        }).create_default_view();

        let score: u32 = 0;
        let font: &[u8] = include_bytes!("font.ttf");

        let road = &[
        	Vertex { position: [-4.0, -1.0, -1.0], color: [0.2, 0.2, 0.2], normal: [0.0, 1.0, 0.0] },
        	Vertex { position: [4.0, -1.0, -1.0], color: [0.2, 0.2, 0.2], normal: [0.0, 1.0, 0.0] },
        	Vertex { position: [4.0, -1.0, 150.0], color: [0.2, 0.2, 0.2], normal: [0.0, 1.0, 0.0] },
        	Vertex { position: [4.0, -1.0, 150.0], color: [0.2, 0.2, 0.2], normal: [0.0, 1.0, 0.0] },
        	Vertex { position: [-4.0, -1.0, 150.0], color: [0.2, 0.2, 0.2], normal: [0.0, 1.0, 0.0] },
        	Vertex { position: [-4.0, -1.0, -1.0], color: [0.2, 0.2, 0.2], normal: [0.0, 1.0, 0.0] },
        ];

        let road_vertex_buffer = device.create_buffer_mapped(road.len(), wgpu::BufferUsage::VERTEX).fill_from_slice(road);

        let player = &[
       		//Top
       		Vertex { position: [-0.4, 0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 1.0, 0.0] },
       		Vertex { position: [-0.4, 0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 1.0, 0.0] },
       		Vertex { position: [0.4, 0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 1.0, 0.0] },
       		Vertex { position: [0.4, 0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 1.0, 0.0] },
       		Vertex { position: [-0.4, 0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 1.0, 0.0] },
       		Vertex { position: [0.4, 0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 1.0, 0.0] },
       		//Right
       		Vertex { position: [0.4, 0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [1.0, 0.0, 0.0] },
       		Vertex { position: [0.4, -0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [1.0, 0.0, 0.0] },
       		Vertex { position: [0.4, -0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [1.0, 0.0, 0.0] },
       		Vertex { position: [0.4, -0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [1.0, 0.0, 0.0] },
       		Vertex { position: [0.4, 0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [1.0, 0.0, 0.0] },
       		Vertex { position: [0.4, 0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [1.0, 0.0, 0.0] },
       		//Front
       		Vertex { position: [0.4, 0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 0.0, -1.0] },
       		Vertex { position: [-0.4, 0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 0.0, -1.0] },
       		Vertex { position: [0.4, -0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 0.0, -1.0] },
       		Vertex { position: [0.4, -0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 0.0, -1.0] },
       		Vertex { position: [-0.4, 0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 0.0, -1.0] },
       		Vertex { position: [-0.4, -0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 0.0, -1.0] },
       		//Left
       		Vertex { position: [-0.4, -0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [-1.0, 0.0, 0.0] },
       		Vertex { position: [-0.4, 0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [-1.0, 0.0, 0.0] },
       		Vertex { position: [-0.4, -0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [-1.0, 0.0, 0.0] },
       		Vertex { position: [-0.4, -0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [-1.0, 0.0, 0.0] },
       		Vertex { position: [-0.4, 0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [-1.0, 0.0, 0.0] },
       		Vertex { position: [-0.4, 0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [-1.0, 0.0, 0.0] },
       		//Back
       		Vertex { position: [-0.4, 0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 0.0, 1.0] },
       		Vertex { position: [0.4, 0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 0.0, 1.0] },
       		Vertex { position: [0.4, -0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 0.0, 1.0] },
       		Vertex { position: [0.4, -0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 0.0, 1.0] },
       		Vertex { position: [-0.4, -0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 0.0, 1.0] },
       		Vertex { position: [-0.4, 0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [0.0, 0.0, 1.0] },
       		//Bottom
       		Vertex { position: [-0.4, -0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [0.0, -1.0, 0.0] },
       		Vertex { position: [0.4, -0.4, 0.4], color: [0.2, 0.3, 1.0], normal: [0.0, -1.0, 0.0] },
       		Vertex { position: [0.4, -0.4, -0.4], color: [0.6, 0.3, 1.0], normal: [0.0, -1.0, 0.0] },
       		Vertex { position: [0.4, -0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [0.0, -1.0, 0.0] },
       		Vertex { position: [-0.4, -0.4, -0.4], color: [0.2, 0.3, 1.0], normal: [0.0, -1.0, 0.0] },
       		Vertex { position: [-0.4, -0.4, 0.4], color: [0.8, 0.3, 1.0], normal: [0.0, -1.0, 0.0] },
       	];

        let obstacle = &[
        	//Top
        	Vertex { position: [-0.2, 0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 1.0, 0.0] },
        	Vertex { position: [-0.2, 0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 1.0, 0.0] },
        	Vertex { position: [0.2, 0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 1.0, 0.0] },
        	Vertex { position: [0.2, 0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 1.0, 0.0] },
        	Vertex { position: [-0.2, 0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 1.0, 0.0] },
        	Vertex { position: [0.2, 0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 1.0, 0.0] },
        	//Right
        	Vertex { position: [0.2, 0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [1.0, 0.0, 0.0] },
        	Vertex { position: [0.2, -0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [1.0, 0.0, 0.0] },
        	Vertex { position: [0.2, -0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [1.0, 0.0, 0.0] },
        	Vertex { position: [0.2, -0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [1.0, 0.0, 0.0] },
        	Vertex { position: [0.2, 0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [1.0, 0.0, 0.0] },
        	Vertex { position: [0.2, 0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [1.0, 0.0, 0.0] },
        	//Front
        	Vertex { position: [0.2, 0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 0.0, -1.0] },
        	Vertex { position: [-0.2, 0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 0.0, -1.0] },
        	Vertex { position: [0.2, -0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 0.0, -1.0] },
        	Vertex { position: [0.2, -0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 0.0, -1.0] },
        	Vertex { position: [-0.2, 0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 0.0, -1.0] },
        	Vertex { position: [-0.2, -0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 0.0, -1.0] },
        	//Left
        	Vertex { position: [-0.2, -0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [-1.0, 0.0, 0.0] },
        	Vertex { position: [-0.2, 0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [-1.0, 0.0, 0.0] },
        	Vertex { position: [-0.2, -0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [-1.0, 0.0, 0.0] },
        	Vertex { position: [-0.2, -0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [-1.0, 0.0, 0.0] },
        	Vertex { position: [-0.2, 0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [-1.0, 0.0, 0.0] },
        	Vertex { position: [-0.2, 0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [-1.0, 0.0, 0.0] },
        	//Back
        	Vertex { position: [-0.2, 0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 0.0, 1.0] },
        	Vertex { position: [0.2, 0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 0.0, 1.0] },
        	Vertex { position: [0.2, -0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 0.0, 1.0] },
        	Vertex { position: [0.2, -0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 0.0, 1.0] },
        	Vertex { position: [-0.2, -0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 0.0, 1.0] },
        	Vertex { position: [-0.2, 0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [0.0, 0.0, 1.0] },
        	//Bottom
        	Vertex { position: [-0.2, -0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [0.0, -1.0, 0.0] },
        	Vertex { position: [0.2, -0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [0.0, -1.0, 0.0] },
        	Vertex { position: [0.2, -0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [0.0, -1.0, 0.0] },
        	Vertex { position: [0.2, -0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [0.0, -1.0, 0.0] },
        	Vertex { position: [-0.2, -0.2, -0.2], color: [0.9, 0.2, 0.2], normal: [0.0, -1.0, 0.0] },
        	Vertex { position: [-0.2, -0.2, 0.2], color: [0.9, 0.2, 0.2], normal: [0.0, -1.0, 0.0] },
        ];

        let num_obstacle = obstacle.len() as u32;
        let num_player = player.len() as u32;
        let num_road = road.len() as u32;

        let speed = 0.01f32;

        let mut zpos = 15.0f32;

		let mut instances  = vec![];
		for _i in 0..10 {
			for _j in 0..2 {
				let xpos = rand::random::<f32>() * 6.0 - 3.0;
		        let position = cgmath::Vector3 { x: xpos as f32, y: 0.0, z: zpos as f32 };
		
		        let rotation = if position.is_zero() {
		            cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
		        } else {
		            cgmath::Quaternion::from_axis_angle(position.clone().normalize(), cgmath::Deg(0.0))
		        };
		
		        instances.push(Instance {
		            position, rotation,
		        });
			}
			zpos += 15.0f32;
		}

		let instance_data = instances.iter().map(Instance::to_matrix).collect::<Vec<_>>();
		let instance_buffer_size = instance_data.len() * std::mem::size_of::<cgmath::Matrix4<f32>>();
		let instance_buffer = device.create_buffer_mapped(instance_data.len(), wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST).fill_from_slice(&instance_data);

        let player_position = PlayerPosition {
        	position: 0.0f32,
        };

        let position_buffer = device.create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST).fill_from_slice(&[player_position]);

        let player_controller = PlayerController::new(0.1);

        let camera = Camera {
    	    eye: (0.0, 6.0, -5.0).into(),
    	    target: (0.0, 0.0, 5.0).into(),
    	    up: cgmath::Vector3::unit_y(),
    	    aspect: sc_desc.width as f32 / sc_desc.height as f32,
    	    fovy: 50.0,
    	    znear: 0.1,
    	    zfar: 100.0,
    	};

    	let light = Light {
		    direction: (-1.0, -0.4, -0.9).into(),
		};
		let light_buffer = device.create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM).fill_from_slice(&[light]);

    	let obstacle_vertex_buffer = device.create_buffer_mapped(obstacle.len(), wgpu::BufferUsage::VERTEX).fill_from_slice(obstacle);

        let player_vertex_buffer = device.create_buffer_mapped(player.len(), wgpu::BufferUsage::VERTEX).fill_from_slice(player);

        let mut uniforms = Uniforms::new();
       	uniforms.update_view_proj(&camera);

        let uniform_buffer = device.create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST).fill_from_slice(&[uniforms]);

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
        		wgpu::BindGroupLayoutBinding {
        			binding: 2,
        			visibility: wgpu::ShaderStage::FRAGMENT,
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
        				buffer: &uniform_buffer,
        				range: 0..std::mem::size_of_val(&uniforms) as wgpu::BufferAddress,
        			},
        		},
        		wgpu::Binding {
        			binding: 1,
        			resource: wgpu::BindingResource::Buffer {
        				buffer: &position_buffer,
        				range: 0..std::mem::size_of_val(&player_position) as wgpu::BufferAddress,
        			},
        		},
        		wgpu::Binding {
        			binding: 2,
        			resource: wgpu::BindingResource::Buffer {
        				buffer: &light_buffer,
        				range: 0..std::mem::size_of_val(&light) as wgpu::BufferAddress,
        			},
        		},
        	],
        });

        let obstacle_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        		wgpu::BindGroupLayoutBinding {
        			binding: 2,
        			visibility: wgpu::ShaderStage::FRAGMENT,
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
        				buffer: &uniform_buffer,
        				range: 0..std::mem::size_of_val(&uniforms) as wgpu::BufferAddress,
        			},
        		},
        		wgpu::Binding {
        			binding: 2,
        			resource: wgpu::BindingResource::Buffer {
        				buffer: &light_buffer,
        				range: 0..std::mem::size_of_val(&light) as wgpu::BufferAddress,
        			},
        		},
        	],
        });

        let vs_src = include_str!("player.vert");
        let fs_src = include_str!("light.frag");
        let vs_src2 = include_str!("obstacle.vert");
        let vs_src3 = include_str!("road.vert");

        let vs_spriv = glsl_to_spirv::compile(vs_src, glsl_to_spirv::ShaderType::Vertex).unwrap();
        let fs_spirv = glsl_to_spirv::compile(fs_src, glsl_to_spirv::ShaderType::Fragment).unwrap();
        let vs_spriv2 = glsl_to_spirv::compile(vs_src2, glsl_to_spirv::ShaderType::Vertex).unwrap();
        let vs_spriv3 = glsl_to_spirv::compile(vs_src3, glsl_to_spirv::ShaderType::Vertex).unwrap();

        let vs_data = wgpu::read_spirv(vs_spriv).unwrap();
        let vs_data2 = wgpu::read_spirv(vs_spriv2).unwrap();
        let vs_data3 = wgpu::read_spirv(vs_spriv3).unwrap();
        let fs_data = wgpu::read_spirv(fs_spirv).unwrap();

        let vs_module = device.create_shader_module(&vs_data);
        let vs_module2 = device.create_shader_module(&vs_data2);
        let vs_module3 = device.create_shader_module(&vs_data3);
        let fs_module = device.create_shader_module(&fs_data);

        let road_render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        	bind_group_layouts: &[&uniform_bind_group_layout],
        });

        let road_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        	layout: &road_render_pipeline_layout,
        	vertex_stage: wgpu::ProgrammableStageDescriptor {
        		module: &vs_module3,
        		entry_point: "main",
        	},
        	fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
        		module: &fs_module,
        		entry_point: "main",
        	}),
        	rasterization_state: Some(wgpu::RasterizationStateDescriptor {
        		front_face: wgpu::FrontFace::Cw,
        		cull_mode: wgpu::CullMode::Back,
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
        	sample_count: 8,
        	sample_mask: !0,
        	alpha_to_coverage_enabled: false,
        });

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
        		front_face: wgpu::FrontFace::Cw,
        		cull_mode: wgpu::CullMode::Back,
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
        	sample_count: 8,
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
        		front_face: wgpu::FrontFace::Cw,
        		cull_mode: wgpu::CullMode::Back,
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
        	sample_count: 8,
        	sample_mask: !0,
        	alpha_to_coverage_enabled: false,
        });

        let collided = false;

        Self {
        	device,
        	queue,
        	swap_chain,
        	player_render_pipeline,
        	obstacle_render_pipeline,
        	road_render_pipeline,
        	player_vertex_buffer,
        	obstacle_vertex_buffer,
        	uniform_bind_group,
        	obstacle_bind_group,
        	player_controller,
        	player_position,
        	position_buffer,
        	instances,
        	instance_buffer,
        	speed,
        	road_vertex_buffer,
        	instant,
        	num_road,
        	num_player,
        	num_obstacle,
            score,
            font,
            inner_size,
            collided,
            multisampled_framebuffer,
        }
	}

	fn input(&mut self ,event: &WindowEvent) -> bool {
		self.player_controller.control(event)
	}

	fn update(&mut self) {
		self.player_controller.update_player_position(&mut self.player_position);

        self.score = Instant::now().duration_since(self.instant).as_secs() as u32;

		let tmp_buffer = self.device.create_buffer_mapped(1, wgpu::BufferUsage::COPY_SRC).fill_from_slice(&[self.player_position]);

		let instance_data = self.instances.iter().map(Instance::to_matrix).collect::<Vec<_>>();
		let tmp_buffer2 = self.device.create_buffer_mapped(instance_data.len(), wgpu::BufferUsage::COPY_SRC).fill_from_slice(&instance_data);

		let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			todo: 0,
		});

		let copy_size = std::mem::size_of::<cgmath::Matrix4<f32>>() * 20usize;

		encoder.copy_buffer_to_buffer(&tmp_buffer, 0, &self.position_buffer, 0, std::mem::size_of::<f32>() as wgpu::BufferAddress);
		encoder.copy_buffer_to_buffer(&tmp_buffer2, 0, &self.instance_buffer, 0, copy_size as wgpu::BufferAddress);

		self.queue.submit(&[encoder.finish()]);
	}

	fn update_instances(&mut self) {
		for i in 0..20 {
			self.instances[i].position[2] -= self.speed;
			if self.instances[i].position[2] <= -1.0 {
				self.instances[i].position[2] = 149.0f32;
				self.instances[i].position[0] = rand::random::<f32>() * 6.0 - 3.0;
			}
		}

		if self.speed < 5.0f32 {
			self.speed += 0.0005f32;
		}
		else if self.speed < 10.0f32 {
			self.speed += 0.00005f32;
		}
		else {
			self.speed += 0.000001f32;
		}
	}

	fn check_collision(&mut self) {
		for i in 0..20 {
			if self.instances[i].position[2] <= 1.0 {
				let z1 = self.instances[i].position[2];
				let x1 = self.instances[i].position[0];
				let x2 = self.player_position.position;
				let d = ((x1 - x2) * (x1 - x2) + z1 * z1).sqrt() as f32;
				if d < 0.5 {
                    self.collided = true;
				}
			}
		}
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
						attachment: &self.multisampled_framebuffer,
						resolve_target: Some(&frame.view),
						load_op: wgpu::LoadOp::Clear,
						store_op: wgpu::StoreOp::Store,
						clear_color: wgpu::Color {
							r: 0.2,
							g: 0.55,
							b: 0.15,
							a: 1.0,
						},
					},
				],
				depth_stencil_attachment: None,
			});

			render_pass.set_pipeline(&self.road_render_pipeline);
			render_pass.set_vertex_buffers(0, &[(&self.road_vertex_buffer, 0)]);
			render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
			render_pass.draw(0..self.num_road, 0..1);
		}

		{
			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				color_attachments: &[
					wgpu::RenderPassColorAttachmentDescriptor {
						attachment: &self.multisampled_framebuffer,
						resolve_target: Some(&frame.view),
						load_op: wgpu::LoadOp::Load,
						store_op: wgpu::StoreOp::Store,
						clear_color: wgpu::Color {
							r: 0.2,
							g: 0.55,
							b: 0.15,
							a: 1.0,
						},
					},
				],
				depth_stencil_attachment: None,
			});

			render_pass.set_pipeline(&self.obstacle_render_pipeline);
			render_pass.set_vertex_buffers(0, &[(&self.obstacle_vertex_buffer, 0)]);
			render_pass.set_bind_group(0, &self.obstacle_bind_group, &[]);
			render_pass.draw(0..self.num_obstacle, 0..self.instances.len() as u32);
		}

		{
			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				color_attachments: &[
					wgpu::RenderPassColorAttachmentDescriptor {
						attachment: &self.multisampled_framebuffer,
						resolve_target: Some(&frame.view),
						load_op: wgpu::LoadOp::Load,
						store_op: wgpu::StoreOp::Store,
						clear_color: wgpu::Color {
							r: 0.2,
							g: 0.55,
							b: 0.15,
							a: 1.0,
						},
					},
				],
				depth_stencil_attachment: None,
			});

			render_pass.set_pipeline(&self.player_render_pipeline);
			render_pass.set_vertex_buffers(0, &[(&self.player_vertex_buffer, 0)]);
			render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
			render_pass.draw(0..self.num_player, 0..1);
		}

        let render_text = format!("Your Score: {}", self.score);
        let section = Section {
            text: &render_text,
            scale: wgpu_glyph::Scale::uniform(34.0),
            color: [1.0, 1.0, 1.0, 1.0],
            ..Section::default()
        };
        let mut glyph_brush = GlyphBrushBuilder::using_font_bytes(self.font).unwrap().build(&self.device, wgpu::TextureFormat::Bgra8UnormSrgb);

        glyph_brush.queue(section);
        glyph_brush.draw_queued(&self.device, &mut encoder, &frame.view, self.inner_size.width, self.inner_size.height);

		self.queue.submit(&[encoder.finish()]);
	}

    fn render_finish(&mut self) {
        let frame = self.swap_chain.get_next_texture();

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            todo: 0,
        });

        let render_text = format!("Game Over!\nYour Score:\n         {}", self.score);
        let section = Section {
            text: &render_text,
            scale: wgpu_glyph::Scale::uniform(50.0),
            color: [1.0, 1.0, 1.0, 1.0],
            screen_position: (280.0, 220.0),
            ..Section::default()
        };

        let font: &[u8] = include_bytes!("font2.ttf");
        let mut glyph_brush = GlyphBrushBuilder::using_font_bytes(font).unwrap().build(&self.device, wgpu::TextureFormat::Bgra8UnormSrgb);

        glyph_brush.queue(section);
        glyph_brush.draw_queued(&self.device, &mut encoder, &frame.view, self.inner_size.width, self.inner_size.height);

        self.queue.submit(&[encoder.finish()]);
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Vertex {
	position: [f32; 3],
	color: [f32; 3],
	normal: [f32; 3],
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
				wgpu::VertexAttributeDescriptor {
					offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
					format: wgpu::VertexFormat::Float3,
					shader_location: 2,
				},
			],
		}
	}
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

#[derive(Debug)]
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

#[repr(C)]
#[derive(Copy, Clone)]
struct Light {
    direction: cgmath::Vector3<f32>,
}