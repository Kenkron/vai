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
    let c1 = (0.2, 0.3, 0.1);
    let c2 = (0.6, 0.6, 0.2);
    // distsq
    let d1 = (c1.0 - x) * (c1.0 - x) + (c1.1 - y) * (c1.1 - y);
    let d2 = (c2.0 - x) * (c2.0 - x) + (c2.1 - y) * (c2.1 - y);
    // weight
    let w1 = relu(d1 - c1.2 * c1.2);
    let w2 = relu(d2 - c2.2 * c2.2);
    return  w2;
}

fn test<const I: usize, const C: usize, const E: usize>(
    ai: &vai::VAI<I, 1, C, E>,
    debug: impl Fn(f32, f32, usize) -> ())
-> f32{
    let mut outer = 0.0;
    let mut miss_outer = 0.0;
    let mut inner = 0.0;
    let mut miss_inner = 0.0;
    let tests = 1000;
    for _ in 0..tests {
        let mut input = na::SMatrix::<f32, I, 1>::new_random();
        input[0] = 1.0;
        let x = input[1];
        let y = input[2];
        for i in 3..I {
            input[i] = 0.0;
        }
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

    let mut best_ai = vai::VAI::<3, 1, 8, 1>::new();
    //best_ai = best_ai.create_variant(1.0);
    let mut score = test(&best_ai, |_,_,_| ());

    println!("Starting ai:\n{}", best_ai);
    println!("Starting score: {}", score);

    let mut test_ai = best_ai.clone();
    let mut tweaking = false;
    let mut generation = 0;
    loop {
        generation += 1;
        if is_key_pressed(KeyCode::T) {
            tweaking = !tweaking;
        }
        if !is_key_down(KeyCode::Space) {
            // Some mutations will be big, some small
            if tweaking {
                test_ai = best_ai.create_layer_variant(rand::random::<f32>() * 0.25);
            } else {
                test_ai = best_ai.create_variant(rand::random::<f32>() * 0.25);
            }
            let s = test(&test_ai, |_,_,_| ());
            let re_check = test(&best_ai, |_,_,_| ());
            // Constantly update best score based on new data
            score = (score + re_check) * 0.5;
            if s < score {
                draw_text(&format!("Score was better: {}", s), 10., 260., 20., WHITE);
                //println!("Score was better: {}", s);
                best_ai = test_ai.clone();
                score = s;
            } else {
                draw_text(&format!("Score was worse: {}", s), 10., 260., 20., WHITE);
                //println!("Score was worse: {}", s);
            }
        }

        if is_key_pressed(KeyCode::B) {
            println!("best ai: {}", best_ai)
        }
        if is_key_down(KeyCode::B) {
            let r = test(&best_ai, |x, y, result| {
                let colors = [GRAY, RED, PURPLE, YELLOW, GREEN];
                draw_circle(x * 250., y * 250., 2., colors[result]);
            });
            draw_text(&format!("Best score: {}\n", r), 10., 275., 20., WHITE);
        } else {
            let r = test(&test_ai, |x, y, result| {
                let colors = [GRAY, RED, PURPLE, YELLOW, GREEN];
                draw_circle(x * 250., y * 250., 2., colors[result]);
            });
            draw_text(&format!("Test score: {}\n", r), 10., 275., 20., WHITE);
        }
        draw_text(&format!("Generation: {}", generation), 10., 290., 20.0, WHITE);
        draw_text(&format!("Tweaking: {}", tweaking), 10., 305., 20.0, WHITE);
        
        if is_key_down(KeyCode::Escape) {
            break;
        }

        next_frame().await
    }
    println!("Final ai:\n{}", best_ai);
    println!("Final score: {}", score);
}
