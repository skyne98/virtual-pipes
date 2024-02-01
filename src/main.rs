/*
Xing Mei, Philippe Decaudin, Bao-Gang Hu. Fast Hydraulic Erosion Simulation and Visualization
on GPU. PG ’07 - 15th Pacific Conference on Computer Graphics and Applications, Oct 2007, Maui,
United States. pp.47-56, ff10.1109/PG.2007.15ff. ffinria-00402079
*/

use std::cmp::max;
use std::time::Instant;

use notan::draw::*;
use notan::prelude::*;
use ultraviolet::Vec2;

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
const SCALE: usize = 10;

#[derive(AppState)]
struct State {
    // Cell
    sediment: Vec<f32>,
    water: Vec<f32>,
    suspended_sediment: Vec<f32>,
    flux_left: Vec<f32>,
    flux_right: Vec<f32>,
    flux_bottom: Vec<f32>,
    flux_top: Vec<f32>,
    velocity: Vec<Vec2>,
    // External
    increment: Vec<f32>,
    delta: f32,
    pipe_area: f32,
    pipe_length: f32,
}

impl State {
    fn new() -> Self {
        Self {
            sediment: vec![0.0; WIDTH * HEIGHT],
            water: vec![0.0; WIDTH * HEIGHT],
            suspended_sediment: vec![0.0; WIDTH * HEIGHT],
            flux_left: vec![0.0; WIDTH * HEIGHT],
            flux_right: vec![0.0; WIDTH * HEIGHT],
            flux_bottom: vec![0.0; WIDTH * HEIGHT],
            flux_top: vec![0.0; WIDTH * HEIGHT],
            velocity: vec![Vec2::zero(); WIDTH * HEIGHT],
            increment: vec![0.0; WIDTH * HEIGHT],
            delta: 0.1,
            pipe_area: 0.1,
            pipe_length: 0.1,
        }
    }
    fn ix(&self, x: usize, y: usize) -> usize {
        if x >= WIDTH || y >= HEIGHT {
            return usize::MAX;
        }
        x + y * WIDTH
    }
    fn ix2(&self, ix: usize) -> (usize, usize) {
        (ix % WIDTH, ix / WIDTH)
    }

    // Getters
    fn get_sediment(&self, index: usize) -> f32 {
        if index >= WIDTH * HEIGHT {
            return 0.0;
        }
        self.sediment[index]
    }
    fn get_water(&self, index: usize) -> f32 {
        if index >= WIDTH * HEIGHT {
            return 0.0;
        }
        self.water[index]
    }
    fn get_flux_left(&self, index: usize) -> f32 {
        if index >= WIDTH * HEIGHT {
            return 0.0;
        }
        self.flux_left[index]
    }
    fn get_flux_right(&self, index: usize) -> f32 {
        if index >= WIDTH * HEIGHT {
            return 0.0;
        }
        self.flux_right[index]
    }
    fn get_flux_bottom(&self, index: usize) -> f32 {
        if index >= WIDTH * HEIGHT {
            return 0.0;
        }
        self.flux_bottom[index]
    }
    fn get_flux_top(&self, index: usize) -> f32 {
        if index >= WIDTH * HEIGHT {
            return 0.0;
        }
        self.flux_top[index]
    }

    fn step(&mut self) {
        for index in 0..WIDTH * HEIGHT {
            let (x, y) = self.ix2(index);

            // 3.1 Water Increment
            let d1 = self.water[index] + self.delta * self.increment[index];
            // 3.2 Flow Simulation
            // 3.2.1 Output Flux Computation
            let g = 9.81;
            // ...left
            let delta_h_l = self.sediment[index] + self.water[index]
                - self.get_sediment(self.ix(x - 1, y))
                - self.get_water(self.ix(x - 1, y));
            let flux_l_prev = self.flux_left[index];
            let flux_l = f32::max(
                0.0,
                flux_l_prev + self.delta * self.pipe_area * ((g * delta_h_l) / self.pipe_length),
            );
            // ...right
            let delta_h_r = self.sediment[index] + self.water[index]
                - self.get_sediment(self.ix(x + 1, y))
                - self.get_water(self.ix(x + 1, y));
            let flux_r_prev = self.flux_right[index];
            let flux_r = f32::max(
                0.0,
                flux_r_prev + self.delta * self.pipe_area * ((g * delta_h_r) / self.pipe_length),
            );
            // ...bottom
            let delta_h_b = self.sediment[index] + self.water[index]
                - self.get_sediment(self.ix(x, y - 1))
                - self.get_water(self.ix(x, y - 1));
            let flux_b_prev = self.flux_bottom[index];
            let flux_b = f32::max(
                0.0,
                flux_b_prev + self.delta * self.pipe_area * ((g * delta_h_b) / self.pipe_length),
            );
            // ...top
            let delta_h_t = self.sediment[index] + self.water[index]
                - self.get_sediment(self.ix(x, y + 1))
                - self.get_water(self.ix(x, y + 1));
            let flux_t_prev = self.flux_top[index];
            let flux_t = f32::max(
                0.0,
                flux_t_prev + self.delta * self.pipe_area * ((g * delta_h_t) / self.pipe_length),
            );
            let lx = 1.0;
            let ly = 1.0;
            let k = f32::min(
                1.0,
                (d1 * lx * ly) / ((flux_l + flux_r + flux_b + flux_t) * self.delta),
            );
            let flux_l = k * flux_l;
            let flux_r = k * flux_r;
            let flux_b = k * flux_b;
            let flux_t = k * flux_t;
            // 3.2.2 Water Surface and Velocity Field Update
            // Additional variables we might need, assuming they are defined similarly to your existing variables
            let mut inflow_sum = 0.0;
            let outflow_sum = flux_l + flux_r + flux_b + flux_t;
            // Ensure we do not go out of bounds
            inflow_sum += self.get_flux_right(self.ix(x - 1, y));
            inflow_sum += self.get_flux_left(self.ix(x + 1, y));
            inflow_sum += self.get_flux_top(self.ix(x, y - 1));
            inflow_sum += self.get_flux_bottom(self.ix(x, y + 1));
            // Calculate ΔV(x, y) according to the provided formula
            let delta_v = self.delta * (inflow_sum - outflow_sum);
            // Update the water surface with the calculated delta_v
            let d2 = d1 + (delta_v / (lx * ly));
            let delta_w_x = 0.5
                * (self.get_flux_right(self.ix(x - 1, y)) - self.flux_left[index]
                    + self.flux_right[index]
                    - self.get_flux_left(self.ix(x + 1, y)));
            let d_avg = 0.5 * (d1 + d2);
            let u = delta_w_x / (ly * d_avg);
            let delta_w_y = 0.5
                * (self.get_flux_top(self.ix(x, y - 1)) - self.flux_bottom[index]
                    + self.flux_top[index]
                    - self.get_flux_bottom(self.ix(x, y + 1)));
            let v = delta_w_y / (lx * d_avg);
            // Update the velocity field with the calculated u and v
            self.velocity[index] = Vec2::new(u, v);
            self.flux_left[index] = flux_l;
            self.flux_right[index] = flux_r;
            self.flux_bottom[index] = flux_b;
            self.flux_top[index] = flux_t;

            // Temporarily, just set the water to the new value
            self.water[index] = d2;
        }
    }
}

#[notan_main]
fn main() -> Result<(), String> {
    // Check the documentation for more options
    let window_config = WindowConfig::new()
        .set_title("Virtual Pipes Demo")
        .set_size(WIDTH as u32 * SCALE as u32, HEIGHT as u32 * SCALE as u32);

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

fn update(app: &mut App, state: &mut State) {
    let fps = app.timer.fps();
    app.window()
        .set_title(&format!("Virtual Pipes Demo - FPS: {}", fps));
    // Input
    if app.mouse.down.contains_key(&MouseButton::Left) {
        let x = (app.mouse.x / SCALE as f32) as usize;
        let y = (app.mouse.y / SCALE as f32) as usize;
        let ix = state.ix(x, y);
        if ix != usize::MAX {
            state.water[ix] += 1.0;
        }
    }
    // Simulation
    let start_time = Instant::now();
    state.step();
    let elapsed = start_time.elapsed();
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ix_and_ix2() {
        let state = State::new();
        for x in 0..WIDTH {
            for y in 0..HEIGHT {
                let ix = state.ix(x, y);
                let (x2, y2) = state.ix2(ix);
                assert_eq!(x, x2);
                assert_eq!(y, y2);
            }
        }
    }

    #[test]
    fn test_getters() {
        let state = State::new();
        for x in 0..WIDTH {
            for y in 0..HEIGHT {
                let ix = state.ix(x, y);
                assert_eq!(state.get_sediment(ix), 0.0);
                assert_eq!(state.get_water(ix), 0.0);
                assert_eq!(state.get_flux_left(ix), 0.0);
                assert_eq!(state.get_flux_right(ix), 0.0);
                assert_eq!(state.get_flux_bottom(ix), 0.0);
                assert_eq!(state.get_flux_top(ix), 0.0);
            }
        }
    }
}
