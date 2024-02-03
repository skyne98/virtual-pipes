/*
Xing Mei, Philippe Decaudin, Bao-Gang Hu. Fast Hydraulic Erosion Simulation and Visualization
on GPU. PG ’07 - 15th Pacific Conference on Computer Graphics and Applications, Oct 2007, Maui,
United States. pp.47-56, ff10.1109/PG.2007.15ff. ffinria-00402079
*/

use std::time::Instant;

use nalgebra::*;
use noise::{NoiseFn, Perlin};
use notan::draw::*;
use notan::prelude::*;
use palette::Mix;
use palette::Srgb;
use palette::Srgba;

const WIDTH: usize = 128;
const HEIGHT: usize = 128;
const SCALE: usize = 10;
const CELL_SIZE: f32 = 1.0;
const CAPACITY_K: f32 = 0.025;

#[derive(AppState)]
struct State {
    perlin: Perlin,
    // Cell
    sediment: Vec<f32>,
    water: Vec<f32>,
    suspended_sediment: Vec<f32>,
    suspended_sediment_1: Vec<f32>,
    flux: Vec<Vector4<f32>>, // left, right, bottom, top
    velocity: Vec<Vector2<f32>>,
    // External
    increment: Vec<f32>,
    fixed_delta: f32,
    pipe_area: f32,
    pipe_length: f32,
    last_step: Instant,
    light_direction: Vector3<f32>,
}

impl State {
    fn new() -> Self {
        let perlin = Perlin::new(1);
        let mut sediment = vec![0.0; WIDTH * HEIGHT];
        for x in 0..WIDTH {
            for y in 0..HEIGHT {
                let i = x + y * WIDTH;
                let nx = x as f64 / WIDTH as f64 - 0.5;
                let nx = nx * 4.0;
                let ny = y as f64 / HEIGHT as f64 - 0.5;
                let ny = ny * 4.0;
                let nz = 0.0;
                let value = perlin.get([nx, ny, nz]);

                // second, more detailed noise
                let nx = x as f64 / WIDTH as f64 - 0.5;
                let nx = nx * 8.0;
                let ny = y as f64 / HEIGHT as f64 - 0.5;
                let ny = ny * 8.0;
                let nz = 10.0;
                let value2 = perlin.get([nx, ny, nz]);
                // scale it down
                let value2 = value2 * 2.5;

                sediment[i] = (value * 10.0 + 10.0).max(0.0).min(20.0) as f32;
                sediment[i] += value2 as f32;
            }
        }

        Self {
            perlin,
            sediment,
            water: vec![0.0; WIDTH * HEIGHT],
            suspended_sediment: vec![0.0; WIDTH * HEIGHT],
            suspended_sediment_1: vec![0.0; WIDTH * HEIGHT],
            flux: vec![Vector4::zeros(); WIDTH * HEIGHT],
            velocity: vec![Vector2::zeros(); WIDTH * HEIGHT],
            increment: vec![0.0; WIDTH * HEIGHT],
            fixed_delta: 0.01,
            pipe_area: 1.0,
            pipe_length: 0.1,
            last_step: Instant::now(),
            light_direction: Vector3::new(1.0, 1.0, 1.0).normalize(),
        }
    }
    fn ix(&self, x: usize, y: usize) -> usize {
        if x >= WIDTH || y >= HEIGHT {
            // tile the edges
            return self.ix(x % WIDTH, y % HEIGHT);
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
        self.flux[index].x
    }
    fn set_flux_left(&mut self, index: usize, value: f32) {
        self.flux[index].x = value;
    }
    fn get_flux_right(&self, index: usize) -> f32 {
        self.flux[index].y
    }
    fn set_flux_right(&mut self, index: usize, value: f32) {
        self.flux[index].y = value;
    }
    fn get_flux_bottom(&self, index: usize) -> f32 {
        self.flux[index].z
    }
    fn set_flux_bottom(&mut self, index: usize, value: f32) {
        self.flux[index].z = value;
    }
    fn get_flux_top(&self, index: usize) -> f32 {
        self.flux[index].w
    }
    fn set_flux_top(&mut self, index: usize, value: f32) {
        self.flux[index].w = value;
    }
    /// Get the four neighbors of a cell:
    /// left, right, bottom, top
    fn get_neighbors(&self, index: usize) -> Vector4<usize> {
        let (x, y) = self.ix2(index);
        let left = self.ix(x - 1, y);
        let right = self.ix(x + 1, y);
        let bottom = self.ix(x, y - 1);
        let top = self.ix(x, y + 1);
        Vector4::new(left, right, bottom, top)
    }
    fn get_tilt(&self, index: usize) -> f32 {
        // use the normal to calculate the tilt
        let normal = self.get_normal(index);
        let tilt = f32::acos(normal.z);
        assert!(!tilt.is_nan(), "tilt is NaN");
        assert!(!tilt.is_infinite(), "tilt is infinite");
        assert!(tilt >= 0.0, "tilt is negative");
        assert!(
            tilt <= std::f32::consts::PI / 2.0,
            "tilt is greater than PI"
        );
        tilt
    }
    fn get_normal(&self, index: usize) -> Vector3<f32> {
        if index >= WIDTH * HEIGHT {
            return Vector3::zeros();
        }
        let (x, y) = self.ix2(index);
        let diff_left = if x == 0 {
            0.0
        } else {
            self.sediment[index] - self.sediment[self.ix(x - 1, y)]
        };
        let diff_right = if x == WIDTH - 1 {
            0.0
        } else {
            self.sediment[index] - self.sediment[self.ix(x + 1, y)]
        };
        let diff_bottom = if y == 0 {
            0.0
        } else {
            self.sediment[index] - self.sediment[self.ix(x, y - 1)]
        };
        let diff_top = if y == HEIGHT - 1 {
            0.0
        } else {
            self.sediment[index] - self.sediment[self.ix(x, y + 1)]
        };

        let normal_x = Vector3::new(1.0, 0.0, diff_left - diff_right);
        let normal_y = Vector3::new(0.0, 1.0, diff_bottom - diff_top);
        normal_x.cross(&normal_y).normalize()
    }
    fn get_normal_color(&self, index: usize) -> Srgb {
        let normal = self.get_normal(index);
        /*
         X: -1 to +1 :  Red:     0 to 255
         Y: -1 to +1 :  Green:   0 to 255
         Z:  0 to 1 :  Blue:  128 to 255
        */
        let r = ((normal.x + 1.0) * 0.5 * 255.0) as u8;
        let g = ((normal.y + 1.0) * 0.5 * 255.0) as u8;
        let b = ((normal.z * 128.0) + 128.0) as u8;
        Srgb::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
    }
    fn raycast_terrain(&self, x: f32, y: f32, direction: Vector3<f32>) -> f32 {
        if direction.x == 0.0 && direction.y == 0.0 {
            return 0.0;
        }
        let start_ix = self.ix(x as usize, y as usize);
        let start_sediment = self.sediment[start_ix];
        let mut t = start_sediment;
        let mut current_x = x;
        let mut current_y = y;
        while current_x >= 0.0
            && current_x < WIDTH as f32
            && current_y >= 0.0
            && current_y < HEIGHT as f32
        {
            let ix = self.ix(current_x as usize, current_y as usize);
            let sediment = self.sediment[ix];
            let height = sediment;
            if height > t {
                return t;
            }
            t += direction.z;
            current_x += direction.x;
            current_y += direction.y;
        }
        0.0
    }

    fn step(&mut self) {
        let delta = self.fixed_delta;
        for index in 0..WIDTH * HEIGHT {
            let (x, y) = self.ix2(index);

            // 3.1 Water Increment
            let neighbors_ix = self.get_neighbors(index);
            let d1 = self.water[index] + delta * self.increment[index];
            // 3.2 Flow Simulation
            // 3.2.1 Output Flux Computation
            let g = 9.81;
            // ...left
            let delta_h = neighbors_ix.map(|ix| {
                self.sediment[index] + self.water[index] - self.sediment[ix] - self.water[ix]
            });
            let flux_ix = Vector4::new(0usize, 1, 2, 3);
            let flux_prev = self.flux[index];
            let flux = flux_ix.map(|i| {
                f32::max(
                    0.0,
                    flux_prev[i] + delta * self.pipe_area * ((g * delta_h[i]) / self.pipe_length),
                )
            });
            let lx = CELL_SIZE;
            let ly = CELL_SIZE;
            let k = f32::min(1.0, (d1 * lx * ly) / ((flux.sum()) * delta));
            if k.is_nan() || k.is_infinite() {
                panic!("k is NaN or infinite; d1: {}; lx: {}; ly: {}; flux_l: {}; flux_r: {}; flux_b: {}; flux_t: {}; delta: {}",
                    d1, lx, ly, flux.x, flux.y, flux.z, flux.w, delta
                );
            }
            let mut flux = flux * k;

            // Boundary conditions
            if x == 0 {
                flux.x = 0.0;
            }
            if x == WIDTH - 1 {
                flux.y = 0.0;
            }
            if y == 0 {
                flux.z = 0.0;
            }
            if y == HEIGHT - 1 {
                flux.w = 0.0;
            }

            // 3.2.2 Water Surface and Velocity Field Update
            // Additional variables we might need, assuming they are defined similarly to your existing variables
            let mut inflow_sum = 0.0;
            let outflow_sum = flux.sum();
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
                * (left_flux_right - self.get_flux_left(index) + self.get_flux_right(index)
                    - right_flux_left);
            let delta_w_y = 0.5
                * (bottom_flux_top - self.get_flux_top(index) + self.get_flux_bottom(index)
                    - top_flux_bottom);
            // Update the velocity field with the calculated u and v
            let u = delta_w_x;
            let v = delta_w_y;
            assert!(!u.is_nan(), "u is NaN");
            assert!(!v.is_nan(), "v is NaN");
            let velocity = Vector2::new(u, v);
            self.velocity[index] = velocity.clone();
            self.flux[index] = flux.clone();

            // 3.3 Erosion and Deposition
            let k_c = CAPACITY_K;
            // calculate based on difference in sediment level with neighbors
            let tilt_angle = self.get_tilt(index);
            if velocity.x.is_nan() || velocity.x.is_infinite() {
                panic!("velocity.x is NaN or infinite");
            }
            if velocity.y.is_nan() || velocity.y.is_infinite() {
                panic!("velocity.y is NaN or infinite");
            }
            let c = k_c * f32::sin(tilt_angle) * velocity.magnitude();
            if c.is_nan() || c.is_infinite() {
                panic!(
                    "c is NaN or infinite; tilt_angle: {}; velocity.mag(): {}; velocity: {:?}",
                    tilt_angle,
                    velocity.magnitude(),
                    velocity
                );
            }
            if c < 0.0 {
                panic!(
                    "c is negative; tilt_angle: {}; velocity.mag(): {}; velocity: {:?}",
                    tilt_angle,
                    velocity.magnitude(),
                    velocity
                );
            }
            let k_s = 0.005;
            let k_d = 0.02;
            let (b, s1) = if c > self.suspended_sediment[index] {
                let change_amount = k_s * (c - self.suspended_sediment[index]);
                let change_amount = f32::min(change_amount, self.sediment[index]);
                (
                    self.sediment[index] - change_amount,
                    self.suspended_sediment[index] + change_amount,
                )
            } else {
                let change_amount = k_d * (self.suspended_sediment[index] - c);
                let change_amount = f32::min(change_amount, self.suspended_sediment[index]);
                (
                    self.sediment[index] + change_amount,
                    self.suspended_sediment[index] - change_amount,
                )
            };
            if b.is_nan() || b.is_infinite() {
                panic!("b is NaN or infinite");
            }
            self.sediment[index] = b;
            self.suspended_sediment_1[index] = s1;
            // self.suspended_sediment[index] = s1;

            // Temporarily
            self.water[index] = f32::max(0.0, d2);
        }

        // 3.4 Sediment Transportation
        for index in 0..WIDTH * HEIGHT {
            let (x, y) = self.ix2(index);
            let velocity = self.velocity[index];
            let s_x = x as f32 - velocity.x * delta;
            let s_y = y as f32 - velocity.y * delta;
            if s_x < 0.0 || s_x >= WIDTH as f32 || s_y < 0.0 || s_y >= HEIGHT as f32 {
                continue;
            }
            // get four closest grid points
            let s_x_floor = s_x.floor() as usize;
            let s_x_ceil = s_x.ceil() as usize;
            let s_y_floor = s_y.floor() as usize;
            let s_y_ceil = s_y.ceil() as usize;
            assert!(s_x_floor == s_x_ceil || s_x_floor + 1 == s_x_ceil);
            assert!(s_y_floor == s_y_ceil || s_y_floor + 1 == s_y_ceil);

            let bottom_left_index = self.ix(s_x_floor, s_y_floor);
            let bottom_right_index = self.ix(s_x_ceil, s_y_floor);
            let top_left_index = self.ix(s_x_floor, s_y_ceil);
            let top_right_index = self.ix(s_x_ceil, s_y_ceil);

            let bottom_left_suspended = self.suspended_sediment_1[bottom_left_index];
            let bottom_right_suspended = self.suspended_sediment_1[bottom_right_index];
            let top_left_suspended = self.suspended_sediment_1[top_left_index];
            let top_right_suspended = self.suspended_sediment_1[top_right_index];

            let left_fraction = s_x_ceil as f32 - s_x;
            let right_fraction = s_x - s_x_floor as f32;
            let bottom_interpolated =
                bottom_left_suspended * left_fraction + bottom_right_suspended * right_fraction;
            let top_interpolated =
                top_left_suspended * left_fraction + top_right_suspended * right_fraction;
            let bottom_fraction = s_y_ceil as f32 - s_y;
            let top_fraction = s_y - s_y_floor as f32;
            let interpolated =
                bottom_interpolated * bottom_fraction + top_interpolated * top_fraction;
            self.suspended_sediment[index] = interpolated;
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
    let sediment_sum: f32 = state.sediment.iter().sum();
    let suspended_sediment_sum: f32 = state.suspended_sediment.iter().sum();
    let sediment_and_suspend_sediment_sum = sediment_sum + suspended_sediment_sum;
    // Display stats under the mouse cursor
    let mouse_x = (app.mouse.x / SCALE as f32) as usize;
    let mouse_y = (app.mouse.y / SCALE as f32) as usize;
    let mouse_ix = state.ix(mouse_x, mouse_y);
    if mouse_ix != usize::MAX {
        let mouse_ix = mouse_ix;
        let mouse_sediment = state.get_sediment(mouse_ix);
        let mouse_water = state.get_water(mouse_ix);
        let mouse_suspend_sediment = state.suspended_sediment[mouse_ix];
        let mouse_tilt = state.get_tilt(mouse_ix);
        let mouse_normal = state.get_normal(mouse_ix);
        app.window().set_title(&format!(
            "Virtual Pipes Demo - FPS: {:.2}; Sediment: {:.2}; Suspended: {:.2}; Sum: {:.2}; Water: {:.2}; [Sediment: {:.2}; Water: {:.2}; Suspended Sediment: {:.2}, Tilt: {:.2}, Normal: {:?}]",
            fps, sediment_sum, suspended_sediment_sum, sediment_and_suspend_sediment_sum, water_sum, mouse_sediment, mouse_water, mouse_suspend_sediment, mouse_tilt, mouse_normal
        ));
    }
    // Input
    if app.mouse.down.contains_key(&MouseButton::Left) {
        let x = (app.mouse.x / SCALE as f32) as usize;
        let y = (app.mouse.y / SCALE as f32) as usize;
        let ix = state.ix(x, y);
        if ix != usize::MAX {
            state.water[ix] += 1.0;
        }
    }
    if app.mouse.down.contains_key(&MouseButton::Right) {
        let x = (app.mouse.x / SCALE as f32) as usize;
        let y = (app.mouse.y / SCALE as f32) as usize;
        let ix = state.ix(x, y);
        if ix != usize::MAX {
            state.sediment[ix] += 2.0;
        }
    }
    // Simulation
    let current_time = Instant::now();
    if current_time.duration_since(state.last_step).as_secs_f32() > state.fixed_delta {
        state.last_step = current_time;
        state.step();
        // state.sediment_step(2.0);
    }
    // Check for NaNs
    for i in 0..WIDTH * HEIGHT {
        if state.water[i].is_nan() {
            panic!("NaN in water");
        }
        if state.get_flux_left(i).is_nan() {
            panic!("NaN in flux_left");
        }
        if state.get_flux_right(i).is_nan() {
            panic!("NaN in flux_right");
        }
        if state.get_flux_bottom(i).is_nan() {
            panic!("NaN in flux_bottom");
        }
        if state.get_flux_top(i).is_nan() {
            panic!("NaN in flux_top");
        }
        if state.sediment[i].is_nan() {
            panic!("NaN in sediment");
        }
    }
}

fn srgb_to_color(srgb: Srgb) -> Color {
    Color::new(srgb.red, srgb.green, srgb.blue, 1.0)
}
fn srgba_to_color(srgba: Srgba) -> Color {
    Color::new(srgba.red, srgba.green, srgba.blue, srgba.alpha)
}

fn draw(gfx: &mut Graphics, state: &mut State) {
    let mut draw = gfx.create_draw();
    draw.clear(Color::BLACK);

    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            let i = x + y * WIDTH;
            let sediment = state.sediment[i];
            let water = state.water[i];

            let ambient_light = 0.35;
            let sand_color: Srgb<f32> = Srgb::new(246u8, 215u8, 176u8).into_format();
            let grass_color: Srgb<f32> = Srgb::new(65u8, 152u8, 10u8).into_format();
            let stone_color: Srgb<f32> = Srgb::new(97u8, 84u8, 65u8).into_format();
            let grass_start = 10.0;
            let stone_start = 15.0;
            let terrain_max = 20.0;
            // do linear interpolation
            let sediment_color = if sediment < grass_start {
                sand_color.mix(grass_color, sediment / grass_start)
            } else if sediment < stone_start {
                grass_color.mix(
                    stone_color,
                    (sediment - grass_start) / (stone_start - grass_start),
                )
            } else {
                stone_color.mix(
                    Srgb::new(1.0, 1.0, 1.0),
                    (sediment - stone_start) / (terrain_max - stone_start),
                )
            };
            let normal = state.get_normal(i);
            let light_direction = state.light_direction.clone();
            let sky_light_direction = Vector3::new(0.0, 0.0, 1.0).normalize();
            let diffuse = f32::max(ambient_light, normal.dot(&light_direction));
            let sky_diffuse = f32::max(ambient_light, normal.dot(&sky_light_direction));
            let shadow_raycast = state.raycast_terrain(x as f32, y as f32, light_direction);
            let shadow = if shadow_raycast != 0.0 {
                ambient_light
            } else {
                1.0
            };
            let color = sediment_color.into_format() * f32::min(diffuse, shadow) * sky_diffuse;
            let water_color = Srgb::new(0.0, 0.0, 1.0);
            let color = color.mix(water_color, water / 5.0);
            let color = srgba_to_color(color.into());
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
