use std::io::BufRead;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use rand::{self, Rng, SeedableRng};
use rand::rngs::StdRng;
use macroquad::{text::draw_text, shapes::draw_circle, window::next_frame};
use macroquad::prelude::{is_key_pressed, KeyCode, is_key_down, Conf};
use macroquad::prelude::{GRAY, RED, PURPLE, YELLOW, GREEN, WHITE};
use nalgebra as na;

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
    return  w1 * w2;
}

fn test<const I: usize, const C: usize, const E: usize>(
    ai: &vai::VAI<I, 1, C, E>,
    random: &mut crate::rand::rngs::StdRng,
    debug: impl Send + Fn(f32, f32, usize) -> ())
-> f32{
    let tests = 1000;
    let random_lock = Arc::new(Mutex::new(random));
    let debug_lock = Arc::new(Mutex::new(debug));
    let outer = Arc::new(Mutex::new(0.));
    let miss_outer = Arc::new(Mutex::new(0.));
    let inner = Arc::new(Mutex::new(0.));
    let miss_inner = Arc::new(Mutex::new(0.));
    (0..tests).into_par_iter().for_each(|_| {

        let x:f32;
        let y:f32;
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
        let out = ai.process(&input)[0];
        let actual = outside(x, y);
        let path:usize;
        if actual > 0. {
            *outer.lock().unwrap() += 1.;
            if !(out > 0.) {
                path = 1;
                *miss_outer.lock().unwrap() += 1.;
            } else {
                path = 2;
            }
        } else {
            *inner.lock().unwrap() += 1.;
            if out > 0. {
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
        outer_cost = *miss_outer.lock().unwrap()/ *outer.lock().unwrap();
        outer_cost *= outer_cost;
    }
    let mut inner_cost = 0.;
    if *inner.lock().unwrap() > 0. {
        inner_cost = *miss_inner.lock().unwrap() / *inner.lock().unwrap();
        inner_cost *= inner_cost;
    }
    return (inner_cost + outer_cost) * 0.5;
}

fn window_conf() -> Conf {
    Conf {
        window_title: "World's Worst AI".to_owned(),
        fullscreen: false,
        window_width: 250,
        window_height: 350,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut rng = StdRng::seed_from_u64(0);
    let mut best_ai = vai::VAI::<3, 1, 16, 1>::new();
    let mut score = test(&best_ai, &mut rng, |_,_,_| ());

    println!("Starting ai:\n{}", best_ai);
    println!("Starting score: {}", score);

    let mut test_ai = best_ai.clone();
    let mut tweaking = false;
    let mut generation = 0;
    let mut paused = true;
    let mut step = false;
    let mut quiet = false;
    let mut show_best = true;
    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
        tweaking ^= is_key_pressed(KeyCode::T);
        step ^= is_key_pressed(KeyCode::Enter);
        paused ^= is_key_pressed(KeyCode::Space);
        quiet ^= is_key_pressed(KeyCode::Q);
        show_best ^= is_key_pressed(KeyCode::B);
        if is_key_pressed(KeyCode::P) {
            println!("best ai: {}", best_ai)
        }
        rng = StdRng::seed_from_u64(generation);
        for _ in 0..16 {
            if step || !paused {
                step = false;
                generation += 1;
                // Some mutations will be big, some small
                if tweaking {
                    test_ai = best_ai.create_layer_variant(rand::random::<f32>());
                } else {
                    test_ai = best_ai.create_variant(rand::random::<f32>());
                }
                let s = test(&test_ai, &mut rng, |_,_,_| ());
                let re_check = test(&best_ai, &mut rng, |_,_,_| ());
                // Constantly update best score based on new data
                score = (score * 15. + re_check) * 0.0625;
                if s < score {
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
                let r = test(&best_ai, &mut rng,  |x, y, result| {
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
            draw_text(&format!("Tweaking: {}", tweaking), 10., 305., 20.0, WHITE);
        }
        draw_text(&format!("Generation: {}", generation), 10., 290., 20.0, WHITE);
        if is_key_pressed(KeyCode::S) {
            match std::fs::File::create("./save.vai") {
                Ok(mut file) => {
                    match best_ai.write(&mut file) {
                        Ok(_) => {
                            println!("Saved matrix");
                        },
                        Err(err) => {println!("Save Error: {}", err)},
                    };
                },
                Err(err) => {println!("Save Error: {}", err)},
            };
        }
        if is_key_pressed(KeyCode::O) {
            match std::fs::File::open("./save.vai") {
                Ok(file) => {
                    match vai::VAI::<3, 1, 16, 1>::read(
                        &mut std::io::BufReader::new(file).lines()) {
                        Ok(result) => {
                            best_ai = result;
                            score = test(&best_ai, &mut rng, |_,_,_| ());
                            println!("Loaded matrix");
                        },
                        Err(err) => {println!("Load Error: {}", err)},
                    };
                },
                Err(err) => {println!("Load Error: {}", err)},
            };
        }

        next_frame().await
    }
    println!("Final ai:\n{}", best_ai);
    println!("Final score: {}", score);
}
