use winit::{
    event_loop::{EventLoop, ControlFlow},
    event::*,
    window::Window,
    dpi::PhysicalSize,
};
use std::time::{Instant, Duration};

fn main() {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();

    let time = Instant::now() + Duration::from_millis(40);

    let state = State::new(&window);

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
                *control_flow = ControlFlow::WaitUntil(time);
            },
            _ => *control_flow = ControlFlow::WaitUntil(time),
        }
    });
}

#[allow(dead_code)]
struct State {
	adapter: wgpu::Adapter,
	device: wgpu::Device,
	surface: wgpu::Surface,
	queue: wgpu::Queue,
	swap_chain: wgpu::SwapChain,
	sc_desc: wgpu::SwapChainDescriptor,
}

impl State {
	fn new(window: &Window) -> Self {
		let inner_size = window.inner_size();
		let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions{
			..Default::default()
		}).unwrap();
		let surface = wgpu::Surface::create(window);

		let (device, mut queue) = adapter.request_device(&wgpu::DeviceDescriptor {
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

        Self {
        	adapter,
        	device,
        	queue,
        	swap_chain,
        	surface,
        	sc_desc,
        }
	}

	fn input(&self ,_event: &WindowEvent) -> bool {
		false
	}
}