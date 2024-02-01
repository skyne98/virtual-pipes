use notan::draw::*;
use notan::prelude::*;

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
const SCALE: usize = 10;

#[derive(AppState)]
struct State {
    sediment: Vec<f32>,
    water: Vec<f32>,
}

impl State {
    fn new() -> Self {
        Self {
            sediment: vec![1.0; WIDTH * HEIGHT],
            water: vec![1.0; WIDTH * HEIGHT],
        }
    }
}

#[notan_main]
fn main() -> Result<(), String> {
    // Check the documentation for more options
    let window_config = WindowConfig::new()
        .set_title("Virtual Pipes Demo - Notan")
        .set_size(WIDTH as u32 * SCALE as u32, HEIGHT as u32 * SCALE as u32)
        .set_vsync(true);

    notan::init_with(setup)
        .add_config(window_config)
        .draw(draw)
        .update(update)
        .add_config(DrawConfig)
        .build()
}

fn setup(gfx: &mut Graphics) -> State {
    State::new()
}

fn update(app: &mut App, state: &mut State) {}

fn draw(gfx: &mut Graphics, state: &mut State) {
    let mut draw = gfx.create_draw();
    draw.clear(Color::BLACK);
    // draw.triangle((400.0, 100.0), (100.0, 500.0), (700.0, 500.0));
    // gfx.render(&draw);
    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            let i = x + y * WIDTH;
            let sediment = state.sediment[i];
            let water = state.water[i];
            let color = Color::new(0.0, sediment, water, 1.0);
            draw.rect(
                (x as f32 * SCALE as f32, y as f32 * SCALE as f32),
                (SCALE as f32, SCALE as f32),
            )
            .color(color);
        }
    }
    gfx.render(&draw);
}
