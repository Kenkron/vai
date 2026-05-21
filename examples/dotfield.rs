#![allow(clippy::needless_return)]

use std::io::BufRead;
use std::sync::{Arc, Mutex};

use macroquad::prelude::{
    draw_line, is_key_pressed, mouse_position, vec2, Color, Conf, KeyCode, Vec2,
};
use macroquad::prelude::{GRAY, GREEN, PURPLE, RED, WHITE, YELLOW};
use macroquad::{shapes::draw_circle, text::draw_text, window::next_frame};
use nalgebra::{self as na, SVector, Vector2, Vector3};
use rand::rngs::StdRng;
use rand::{self, Rng, SeedableRng};
use rayon::prelude::*;
use vai::softmax;

const EXTRA_LAYERS: usize = 0;
const LAYER_SIZE: usize = 12;

fn relu(x: f32) -> f32 {
    return x.max(0.);
}

fn outside(x: f32, y: f32) -> f32 {
    // Overlap of two circles
    // x, y, r
    let c1 = (0.2, 0.3, 0.1);
    let c2 = (0.6, 0.6, 0.2);
    // distsq
    let d1 = (c1.0 - x) * (c1.0 - x) + (c1.1 - y) * (c1.1 - y);
    let d2 = (c2.0 - x) * (c2.0 - x) + (c2.1 - y) * (c2.1 - y);
    // weight
    let w1 = relu(d1 - c1.2 * c1.2);
    let w2 = relu(d2 - c2.2 * c2.2);
    if w1 * w2 > 0.0 {
        return 1.0;
    } else {
        return 0.0;
    }
}

// Categorize as inside or outside
fn categorize(x: f32, y: f32) -> SVector<f32, 2> {
    let out = outside(x, y);
    Vector2::new(1.0 - out, out)
}

fn train<const I: usize, const C: usize, const E: usize>(
    ai: &vai::VAI<I, 2, C, E>,
    tests: usize,
    random: &mut crate::rand::rngs::StdRng,
) -> vai::VAI<I, 2, C, E> {
    let test_points: Vec<_> = (0..tests)
        .map(|_| (random.gen::<f32>(), random.gen::<f32>()))
        .collect();
    let training_data: Vec<_> = test_points
        .iter()
        .map(|(x, y)| {
            let expected_output = categorize(*x, *y);
            let mut input = na::SMatrix::<f32, I, 1>::zeros();
            input[0] = 1.0;
            input[1] = *x;
            input[2] = *y;
            (input, expected_output)
        })
        .collect();
    ai.train_categorizer(training_data, 0.001)
}

fn test_point<const I: usize, const C: usize, const E: usize>(
    ai: &vai::VAI<I, 2, C, E>,
    x: f32,
    y: f32,
) -> f32 {
    let mut input = na::SMatrix::<f32, I, 1>::zeros();
    input[0] = 1.0;
    input[1] = x;
    input[2] = y;
    // This line of code, along with an extra input node, unsurprisingly makes the neural
    // network run a lot better.
    // input[3] = ((x*std::f32::consts::PI).sin() + (y*std::f32::consts::PI).sin()) * 0.5;
    let out = ai.process(&input);
    let softmaxed = vai::softmax_slice(out.as_slice());
    let expected = categorize(x, y);
    let costs = softmaxed
        .iter()
        .zip(expected.as_slice())
        .map(|(actual, expected)| expected - actual);
    costs.fold(0.0_f32, |acc, x| acc.max(x))
}

fn test<const I: usize, const C: usize, const E: usize>(
    ai: &vai::VAI<I, 2, C, E>,
    random: &mut crate::rand::rngs::StdRng,
    debug: impl Send + Fn(f32, f32, usize),
) -> f32 {
    let tests = 2000;
    let random_lock = Arc::new(Mutex::new(random));
    let debug_lock = Arc::new(Mutex::new(debug));
    let outer = Arc::new(Mutex::new(0.));
    let miss_outer = Arc::new(Mutex::new(0.));
    let inner = Arc::new(Mutex::new(0.));
    let miss_inner = Arc::new(Mutex::new(0.));
    let total_cost = Arc::new(Mutex::new(0.));
    (0..tests).into_par_iter().for_each(|_| {
        let x: f32;
        let y: f32;
        {
            let mut random = random_lock.lock().unwrap();
            x = random.gen();
            y = random.gen();
        }
        let mut input = na::SMatrix::<f32, I, 1>::zeros();
        input[0] = 1.0;
        input[1] = x;
        input[2] = y;
        // This line of code, along with an extra input node, unsurprisingly makes the neural
        // network run a lot better.
        // input[3] = ((x*std::f32::consts::PI).sin() + (y*std::f32::consts::PI).sin()) * 0.5;
        let out = ai.process(&input);
        let softmaxed = vai::softmax(out);
        let expected = categorize(x, y);
        let mut cost: f32 = 0.0;
        for i in 0..softmaxed.len() {
            cost = cost.max(expected[i] - softmaxed[i])
        }
        *total_cost.lock().unwrap() += cost;
        let path: usize;
        if expected[1] > expected[0] {
            *outer.lock().unwrap() += 1.;
            if out[1] > out[0] {
                path = 2;
            } else {
                path = 1;
                *miss_outer.lock().unwrap() += 1.;
            }
        } else {
            *inner.lock().unwrap() += 1.;
            if out[1] > out[0] {
                path = 3;
                *miss_inner.lock().unwrap() += 1.;
            } else {
                path = 4;
            }
        }
        debug_lock.lock().unwrap()(x, y, path);
    });
    let mut outer_cost = 0.;
    if *outer.lock().unwrap() > 0. {
        outer_cost = *miss_outer.lock().unwrap() / *outer.lock().unwrap();
        outer_cost *= outer_cost;
    }
    let mut inner_cost = 0.;
    if *inner.lock().unwrap() > 0. {
        inner_cost = *miss_inner.lock().unwrap() / *inner.lock().unwrap();
        inner_cost *= inner_cost;
    }
    // println!(
    //     "total_cost: {}, inner/outer cost: {}",
    //     *total_cost.lock().unwrap(),
    //     (inner_cost + outer_cost) * 0.5,
    // );
    //return *total_cost.lock().unwrap();
    return (inner_cost + outer_cost) * 0.5;
}

fn window_conf() -> Conf {
    Conf {
        window_title: "World's Worst AI".to_owned(),
        fullscreen: false,
        window_width: 800,
        window_height: 600,
        ..Default::default()
    }
}

fn draw_nn<const I: usize, const O: usize, const C: usize, const E: usize>(
    nn: &vai::VAI<I, O, C, E>,
    input: &SVector<f32, I>,
    location: Vec2,
    size: Vec2,
) {
    let (xray, output) = nn.process_transparent(input);
    let font = 16.0;
    let x_spacing = size.x / (xray.len() as f32 + 3.0);
    // Draw input nodes
    let max_input = input.max().max(-input.min());
    let y_spacing = size.y / (input.len() as f32 + 1.0);
    let x = location.x + x_spacing;
    let inner_x = x + x_spacing;
    for (j, val) in input.iter().enumerate() {
        let y = location.y + (j + 1) as f32 * y_spacing;
        let proportion = val / max_input;
        for i in 1..C {
            let intensity =
                10. * nn.input_connections[(i, j)] * proportion / nn.input_connections.max();
            let inner_y_spacing = size.y / (C as f32 + 1.0);
            let inner_y = location.y + (i + 1) as f32 * inner_y_spacing;
            let line_color = if intensity > 0. { GREEN } else { RED };
            draw_line(
                x,
                y,
                inner_x,
                inner_y,
                (intensity.abs() + 0.5).ln(),
                line_color,
            );
        }
        let color = Color::new(1.0 - proportion, proportion, 0.0, 1.0);
        draw_circle(x, y, x_spacing / 4.0, color);
        let tx = x - x_spacing / 8.0;
        draw_text(&format!("{:.2}", val), tx, y, font, WHITE);
    }
    // Draw hidden nodes
    for (i, layer) in xray.iter().enumerate() {
        let layer_max = layer.max().max(-layer.min());
        let y_spacing = size.y / (layer.len() as f32 + 1.0);
        let x = location.x + x_spacing * (i + 2) as f32;
        for (j, val) in layer.iter().enumerate() {
            let y = location.y + (j + 1) as f32 * y_spacing;
            let proportion = val / max_input;
            let inner_x = x + x_spacing;
            if i < xray.len() - 1 {
                for c in 1..C {
                    let intensity = 10.0 * nn.hidden_connections[i][(c, j)] * proportion
                        / nn.hidden_connections[i].max();
                    let inner_y_spacing = size.y / (C as f32 + 1.0);
                    let inner_y = location.y + (c + 1) as f32 * inner_y_spacing;
                    let line_color = if intensity > 0. { GREEN } else { RED };
                    draw_line(
                        x,
                        y,
                        inner_x,
                        inner_y,
                        (intensity.abs() + 0.5).ln(),
                        line_color,
                    );
                }
            } else {
                for c in 0..O {
                    let intensity = 10.0 * nn.output_connections[(c, j)] * proportion
                        / nn.output_connections.max();
                    let inner_y_spacing = size.y / (O as f32 + 1.0);
                    let inner_y = location.y + (c + 1) as f32 * inner_y_spacing;
                    let line_color = if intensity > 0. { GREEN } else { RED };
                    draw_line(
                        x,
                        y,
                        inner_x,
                        inner_y,
                        (intensity.abs() + 0.5).ln(),
                        line_color,
                    );
                }
            }
            let color = if *val > 0.0 { GREEN } else { RED };
            draw_circle(x, y, x_spacing / 4.0, color);
            let tx = x - x_spacing / 8.0;
            draw_text(&format!("{:.2}", val), tx, y, font, WHITE);
        }
    }
    // Draw output nodes
    let softmax_output = vai::softmax(output.clone());
    let y_spacing = size.y / (output.len() as f32 + 1.0);
    let x = location.x + x_spacing * (E + 3) as f32;
    for (j, softmax_val) in softmax_output.iter().enumerate() {
        let y = location.y + (j + 1) as f32 * y_spacing;
        let color = Color::new(1.0 - *softmax_val, *softmax_val, 0.0, 1.0);
        draw_circle(x, y, x_spacing / 4.0, color);
        let tx = x - x_spacing / 8.0;
        draw_text(&format!("{:.2}", softmax_val), tx, y, font, WHITE);
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut rng = StdRng::seed_from_u64(0);
    let mut best_ai = vai::VAI::<3, 2, LAYER_SIZE, EXTRA_LAYERS>::new();
    best_ai = best_ai.create_variant(1.0, &mut rng);
    let mut score = test(&best_ai, &mut rng, |_, _, _| ());

    println!("Starting ai:\n{}", best_ai);
    println!("Starting score: {}", score);

    let mut test_ai = best_ai.clone();
    let mut tweaking = false;
    let mut generation = 0;
    let mut paused = true;
    let mut step = false;
    let mut quiet = false;
    let mut show_best = true;
    let mut training = false;
    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
        training ^= is_key_pressed(KeyCode::T);
        step ^= is_key_pressed(KeyCode::Enter);
        paused ^= is_key_pressed(KeyCode::Space);
        quiet ^= is_key_pressed(KeyCode::Q);
        show_best ^= is_key_pressed(KeyCode::B);
        if is_key_pressed(KeyCode::P) {
            println!("best ai: {}", best_ai)
        }
        rng = StdRng::seed_from_u64(generation);
        let iterations = if training { 1 } else { 1 };
        for _ in 0..iterations {
            if step || !paused {
                step = false;
                generation += 1;
                if training {
                    best_ai = train(&best_ai, 1000, &mut rng);
                } else if tweaking {
                    test_ai = best_ai.create_layer_variant(rand::random::<f32>() * 0.5, &mut rng);
                } else {
                    test_ai = best_ai.create_variant(rand::random::<f32>() * 0.5, &mut rng);
                }
                let s = test(&test_ai, &mut rng, |_, _, _| ());
                let re_check = test(&best_ai, &mut rng, |_, _, _| ());
                // Constantly update best score based on new data
                score = (score * 15. + re_check) * 0.0625;
                if s <= score {
                    draw_text(&format!("Score was better: {}", s), 10., 260., 20., WHITE);
                    best_ai = test_ai.clone();
                    score = s;
                } else {
                    draw_text(&format!("Score was worse: {}", s), 10., 260., 20., WHITE);
                }
            }
        }
        if !quiet {
            if show_best {
                let r = test(&best_ai, &mut rng, |x, y, result| {
                    let colors = [GRAY, RED, PURPLE, YELLOW, GREEN];
                    draw_circle(x * 250., y * 250., 2., colors[result]);
                });
                draw_text(&format!("Best score: {}\n", r), 10., 275., 20., WHITE);
            } else {
                let r = test(&test_ai, &mut rng, |x, y, result| {
                    let colors = [GRAY, RED, PURPLE, YELLOW, GREEN];
                    draw_circle(x * 250., y * 250., 2., colors[result]);
                });
                draw_text(&format!("Test score: {}\n", r), 10., 275., 20., WHITE);
            }
            draw_text(&format!("Training: {}", training), 10., 300., 20.0, WHITE);
            let (x, y) = mouse_position();
            if x <= 250. && y <= 250. {
                let mouse_cost = test_point(&best_ai, x / 250.0, y / 250.0);
                draw_text(&format!("Mouse: {}", mouse_cost), 10., 325., 20.0, WHITE);
                let mouse_input = Vector3::new(1.0, x / 250.0, y / 250.0);
                draw_nn(&best_ai, &mouse_input, vec2(225.0, 0.), vec2(600., 600.));
            }
        }
        draw_text(
            &format!("Generation: {}", generation),
            10.,
            290.,
            20.0,
            WHITE,
        );
        if is_key_pressed(KeyCode::S) {
            match std::fs::File::create("./dotfield-save.vai") {
                Ok(mut file) => {
                    match best_ai.write(&mut file) {
                        Ok(_) => {
                            println!("Saved matrix");
                        }
                        Err(err) => {
                            println!("Save Error: {}", err)
                        }
                    };
                }
                Err(err) => {
                    println!("Save Error: {}", err)
                }
            };
        }
        if is_key_pressed(KeyCode::O) {
            match std::fs::File::open("./dotfield-save.vai") {
                Ok(file) => {
                    match vai::VAI::<3, 2, LAYER_SIZE, EXTRA_LAYERS>::read(
                        &mut std::io::BufReader::new(file).lines(),
                    ) {
                        Ok(result) => {
                            best_ai = result;
                            score = test(&best_ai, &mut rng, |_, _, _| ());
                            println!("Loaded matrix");
                        }
                        Err(err) => {
                            println!("Load Error: {}", err)
                        }
                    };
                }
                Err(err) => {
                    println!("Load Error: {}", err)
                }
            };
        }

        next_frame().await
    }
    println!("Final ai:\n{}", best_ai);
    println!("Final score: {}", score);
}
