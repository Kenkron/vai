use std::io::BufRead;

use rand::{self, Rng, SeedableRng};
use rand::rngs::StdRng;
use macroquad::{text::draw_text, shapes::draw_circle, window::next_frame};
use macroquad::prelude::{is_key_pressed, KeyCode, is_key_down, GRAY, RED, PURPLE, YELLOW, GREEN, WHITE};
use nalgebra as na;
mod vai;

fn relu(x: f32) -> f32 {
    return x.max(0.);
}

fn outside(x: f32, y: f32) -> f32 {
    // Overlap of two circles
    // x, y, r
    //let c1 = (0.2, 0.3, 0.1);
    let c2 = (0.6, 0.6, 0.2);
    // distsq
    //let d1 = (c1.0 - x) * (c1.0 - x) + (c1.1 - y) * (c1.1 - y);
    let d2 = (c2.0 - x) * (c2.0 - x) + (c2.1 - y) * (c2.1 - y);
    // weight
    //let w1 = relu(d1 - c1.2 * c1.2);
    let w2 = relu(d2 - c2.2 * c2.2);
    return  w2;
}

fn test<const I: usize, const C: usize, const E: usize>(
    ai: &vai::VAI<I, 1, C, E>,
    random: &mut crate::rand::rngs::StdRng,
    debug: impl Fn(f32, f32, usize) -> ())
-> f32{
    let mut outer = 0.0;
    let mut miss_outer = 0.0;
    let mut inner = 0.0;
    let mut miss_inner = 0.0;
    let tests = 1000;
    for _ in 0..tests {
        let x = random.gen();
        let y = random.gen();
        let mut input = na::SMatrix::<f32, I, 1>::zeros();
        input[0] = 1.0;
        input[1] = x;
        input[2] = y;
        //input[3] = ((x*PI).sin() + (y*PI).sin()) * 0.5;
        let out = ai.process(&input)[0];
        let actual = outside(x, y);
        let path:usize;
        if actual > 0. {
            outer += 1.;
            if !(out > 0.) {
                path = 1;
                miss_outer += 1.;
            } else {
                path = 2;
            }
        } else {
            inner += 1.;
            if out > 0. {
                path = 3;
                miss_inner += 1.;
            } else {
                path = 4;
            }
        }
        debug(x, y, path);
    }
    let mut outer_cost = 0.;
    if outer > 0. {
        outer_cost = miss_outer/outer;
        outer_cost *= outer_cost;
    }
    let mut inner_cost = 0.;
    if inner > 0. {
        inner_cost = miss_inner/inner;
        inner_cost *= inner_cost;
    }
    return (inner_cost + outer_cost) * 0.5;
}

#[macroquad::main("World's Worst AI")]
async fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let mut best_ai = vai::VAI::<3, 1, 8, 1>::new();
    //best_ai = best_ai.create_variant(1.0);
    let mut score = test(&best_ai, &mut rng, |_,_,_| ());

    println!("Starting ai:\n{}", best_ai);
    println!("Starting score: {}", score);

    let mut test_ai = best_ai.clone();
    let mut tweaking = true;
    let mut generation = 0;
    let mut paused = true;
    let mut step = false;
    let mut quiet = false;
    loop {
        if is_key_pressed(KeyCode::T) {
            tweaking = !tweaking;
        }
        if is_key_pressed(KeyCode::Enter) {
            step = true;
        }
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        rng = StdRng::seed_from_u64(generation);
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
            score = (score + re_check) * 0.5;
            if s < score {
                //draw_text(&format!("Score was better: {}", s), 10., 260., 20., WHITE);
                best_ai = test_ai.clone();
                score = s;
            } else {
                //draw_text(&format!("Score was worse: {}", s), 10., 260., 20., WHITE);
            }
        }
        if is_key_pressed(KeyCode::P) {
            println!("best ai: {}", best_ai)
        }
        if is_key_pressed(KeyCode::Q) {
            quiet = !quiet;
        }
        if !quiet {
            if is_key_down(KeyCode::B) {
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
        if is_key_down(KeyCode::Escape) {
            break;
        }
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
                    match vai::VAI::<3, 1, 8, 1>::read(
                        &mut std::io::BufReader::new(file).lines()) {
                        Ok(result) => {
                            best_ai = result;
                            println!("Loaded matrix");
                        },
                        Err(err) => {println!("Load Error: {}", err)},
                    };
                },
                Err(err) => {println!("Save Error: {}", err)},
            };
        }

        next_frame().await
    }
    println!("Final ai:\n{}", best_ai);
    println!("Final score: {}", score);
}
