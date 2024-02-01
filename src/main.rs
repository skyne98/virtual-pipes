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
    fixed_delta: f32,
    pipe_area: f32,
    pipe_length: f32,
    last_step: Instant,
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
            fixed_delta: 0.01,
            pipe_area: 1.0,
            pipe_length: 0.1,
            last_step: Instant::now(),
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
        let delta = self.fixed_delta;
        for index in 0..WIDTH * HEIGHT {
            let (x, y) = self.ix2(index);

            // 3.1 Water Increment
            let d1 = self.water[index] + delta * self.increment[index];
            // 3.2 Flow Simulation
            // 3.2.1 Output Flux Computation
            let g = 9.81;
            // ...left
            let flux_l = if x == 0 {
                0.0
            } else {
                let delta_h_l = self.sediment[index] + self.water[index]
                    - self.get_sediment(self.ix(x - 1, y))
                    - self.get_water(self.ix(x - 1, y));
                let flux_l_prev = self.flux_left[index];
                f32::max(
                    0.0,
                    flux_l_prev + delta * self.pipe_area * ((g * delta_h_l) / self.pipe_length),
                )
            };
            // ...right
            let flux_r = if x == WIDTH - 1 {
                0.0
            } else {
                let delta_h_r = self.sediment[index] + self.water[index]
                    - self.get_sediment(self.ix(x + 1, y))
                    - self.get_water(self.ix(x + 1, y));
                let flux_r_prev = self.flux_right[index];
                f32::max(
                    0.0,
                    flux_r_prev + delta * self.pipe_area * ((g * delta_h_r) / self.pipe_length),
                )
            };
            // ...bottom
            let flux_b = if y == 0 {
                0.0
            } else {
                let delta_h_b = self.sediment[index] + self.water[index]
                    - self.get_sediment(self.ix(x, y - 1))
                    - self.get_water(self.ix(x, y - 1));
                let flux_b_prev = self.flux_bottom[index];
                f32::max(
                    0.0,
                    flux_b_prev + delta * self.pipe_area * ((g * delta_h_b) / self.pipe_length),
                )
            };
            // ...top
            let flux_t = if y == HEIGHT - 1 {
                0.0
            } else {
                let delta_h_t = self.sediment[index] + self.water[index]
                    - self.get_sediment(self.ix(x, y + 1))
                    - self.get_water(self.ix(x, y + 1));
                let flux_t_prev = self.flux_top[index];
                f32::max(
                    0.0,
                    flux_t_prev + delta * self.pipe_area * ((g * delta_h_t) / self.pipe_length),
                )
            };
            let lx = 1.0;
            let ly = 1.0;
            let mut k = f32::min(
                1.0,
                (d1 * lx * ly) / ((flux_l + flux_r + flux_b + flux_t) * delta),
            );
            if k.is_nan() || k.is_infinite() {
                k = 0.0;
            }
            let flux_l = k * flux_l;
            let flux_r = k * flux_r;
            let flux_b = k * flux_b;
            let flux_t = k * flux_t;
            // 3.2.2 Water Surface and Velocity Field Update
            // Additional variables we might need, assuming they are defined similarly to your existing variables
            let mut inflow_sum = 0.0;
            let outflow_sum = flux_l + flux_r + flux_b + flux_t;
            // Ensure we do not go out of bounds
            if x > 0 {
                inflow_sum += self.get_flux_right(self.ix(x - 1, y));
            }
            if x < WIDTH - 1 {
                inflow_sum += self.get_flux_left(self.ix(x + 1, y));
            }
            if y > 0 {
                inflow_sum += self.get_flux_top(self.ix(x, y - 1));
            }
            if y < HEIGHT - 1 {
                inflow_sum += self.get_flux_bottom(self.ix(x, y + 1));
            }
            // Calculate ΔV(x, y) according to the provided formula
            let delta_v = delta * (inflow_sum - outflow_sum);
            // Update the water surface with the calculated delta_v
            let left_flux_right = if x == 0 {
                0.0
            } else {
                self.get_flux_right(self.ix(x - 1, y))
            };
            let right_flux_left = if x == WIDTH - 1 {
                0.0
            } else {
                self.get_flux_left(self.ix(x + 1, y))
            };
            let bottom_flux_top = if y == 0 {
                0.0
            } else {
                self.get_flux_top(self.ix(x, y - 1))
            };
            let top_flux_bottom = if y == HEIGHT - 1 {
                0.0
            } else {
                self.get_flux_bottom(self.ix(x, y + 1))
            };
            let d2 = d1 + (delta_v / (lx * ly));
            let delta_w_x = 0.5
                * (left_flux_right - self.flux_left[index] + self.flux_right[index]
                    - right_flux_left);
            let d_avg = 0.5 * (d1 + d2);
            let u = delta_w_x / (ly * d_avg);
            let delta_w_y = 0.5
                * (bottom_flux_top - self.flux_bottom[index] + self.flux_top[index]
                    - top_flux_bottom);
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
    fn sediment_step(&mut self, threshold: f32) {
        // Create a new buffer by copying the old one
        let mut new_sediment = self.sediment.clone();

        // Average out the sediment with its neighbors
        for index in 0..WIDTH * HEIGHT {
            let (x, y) = self.ix2(index);
            let mut sediment_sum = self.sediment[index];
            let mut neighbor_count = 1;
            if x > 0 && (self.sediment[index] - self.sediment[self.ix(x - 1, y)]).abs() > threshold
            {
                sediment_sum += self.sediment[self.ix(x - 1, y)];
                neighbor_count += 1;
            }
            if x < WIDTH - 1
                && (self.sediment[index] - self.sediment[self.ix(x + 1, y)]).abs() > threshold
            {
                sediment_sum += self.sediment[self.ix(x + 1, y)];
                neighbor_count += 1;
            }
            if y > 0 && (self.sediment[index] - self.sediment[self.ix(x, y - 1)]).abs() > threshold
            {
                sediment_sum += self.sediment[self.ix(x, y - 1)];
                neighbor_count += 1;
            }
            if y < HEIGHT - 1
                && (self.sediment[index] - self.sediment[self.ix(x, y + 1)]).abs() > threshold
            {
                sediment_sum += self.sediment[self.ix(x, y + 1)];
                neighbor_count += 1;
            }
            new_sediment[index] = sediment_sum / neighbor_count as f32;
        }

        // Replace the old buffer with the new one
        self.sediment = new_sediment;
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
    let water_sum: f32 = state.water.iter().sum();
    app.window().set_title(&format!(
        "Virtual Pipes Demo - FPS: {}; Water: {}",
        fps, water_sum
    ));
    // Input
    if app.mouse.down.contains_key(&MouseButton::Left) {
        let x = (app.mouse.x / SCALE as f32) as usize;
        let y = (app.mouse.y / SCALE as f32) as usize;
        let ix = state.ix(x, y);
        if ix != usize::MAX {
            state.water[ix] += 5.0;
        }
    }
    if app.mouse.down.contains_key(&MouseButton::Right) {
        let x = (app.mouse.x / SCALE as f32) as usize;
        let y = (app.mouse.y / SCALE as f32) as usize;
        let ix = state.ix(x, y);
        if ix != usize::MAX {
            state.sediment[ix] += 0.5;
        }
    }
    // Simulation
    let current_time = Instant::now();
    if current_time.duration_since(state.last_step).as_secs_f32() > state.fixed_delta {
        state.last_step = current_time;
        state.step();
        state.sediment_step(1.5);
    }
    // Check for NaNs
    for i in 0..WIDTH * HEIGHT {
        if state.water[i].is_nan() {
            panic!("NaN in water");
        }
        if state.flux_left[i].is_nan() {
            panic!("NaN in flux_left");
        }
        if state.flux_right[i].is_nan() {
            panic!("NaN in flux_right");
        }
        if state.flux_bottom[i].is_nan() {
            panic!("NaN in flux_bottom");
        }
        if state.flux_top[i].is_nan() {
            panic!("NaN in flux_top");
        }
        if state.sediment[i].is_nan() {
            panic!("NaN in sediment");
        }
    }
}

fn draw(gfx: &mut Graphics, state: &mut State) {
    let mut draw = gfx.create_draw();
    draw.clear(Color::BLACK);

    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            let i = x + y * WIDTH;
            let sediment = state.sediment[i];
            let water = state.water[i];
            let color = Color::new(sediment / 5.0, sediment / 5.0, water / 5.0, 1.0);
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
