use macroquad::prelude::*;

use nalgebra as na;
mod vai;

fn relu(x: f32) -> f32 {
    return x.max(0.);
}

fn outside(x: f32, y: f32) -> f32 {
    // Overlap of two circles
    // x, y, r
    let c1 = (2., 3., 2.);
    let c2 = (1., 2., 2.);
    // distsq
    let d1 = (c1.0 - x) * (c1.0 - x) + (c1.1 - y) * (c1.1 - y);
    let d2 = (c2.0 - x) * (c2.0 - x) + (c2.1 - y) * (c2.1 - y);
    // weight
    let w1 = relu(d1 - c1.2 * c1.2);
    let w2 = relu(d2 - c2.2 * c2.2);
    return w1 + w2;
}

fn test<const C: usize>(
    ai: &vai::VAI<3,1,C>,
    debug: impl Fn(f32, f32, usize) -> ())
-> f32{
    let mut miss_outer = 0.0;
    let mut miss_inner = 0.0;
    for _ in 0..1000 {
        let mut input = na::Matrix3x1::<f32>::new_random() * 5.0;
        input[0] = 5.0;
        let out = ai.process(&input)[0];
        let actual = outside(input[1], input[2]);
        let path:usize;
        if actual > 0. {
            if !(out > 0.) {
                path = 1;
                miss_outer += 1.;
            } else {
                path = 2;
            }
        } else {
            if out > 0. {
                path = 3;
                miss_inner += 1.;
            } else {
                path = 4;
            }
        }
        debug(input[1], input[2], path);
    }
    return miss_outer * miss_outer + miss_inner * miss_inner;
}

#[macroquad::main("World's Worst AI")]
async fn main() {

    let mut best_ai = vai::VAI::<3, 1, 1>::new();
    best_ai = best_ai.create_variant();
    let mut score = test(&best_ai, |_,_,_| ());

    println!("Starting ai:\n{}", best_ai);
    println!("Starting score: {}", score);

    let mut test_ai = best_ai.clone();
    let mut tweaking = false;
    loop {
        if is_key_down(KeyCode::T) {
            tweaking = !tweaking;
        }
        if !is_key_down(KeyCode::Space) {
            // Some mutations will be big, some small
            if tweaking {
                test_ai = best_ai.create_layer_variant();
            } else {
                test_ai = best_ai.create_variant();
            }
            let s = test(&test_ai, |_,_,_| ());
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
                draw_circle(x * 50., y * 50., 2., colors[result]);
            });
            draw_text(&format!("Best score: {}\n", r), 10., 275., 20., WHITE);
        } else {
            let r = test(&test_ai, |x, y, result| {
                let colors = [GRAY, RED, PURPLE, YELLOW, GREEN];
                draw_circle(x * 50., y * 50., 2., colors[result]);
            });
            draw_text(&format!("Test score: {}\n", r), 10., 275., 20., WHITE);
        }
        draw_text(&format!("Tweaking: {}", tweaking), 10., 290., 20.0, WHITE);
        
        if is_key_down(KeyCode::Escape) {
            break;
        }

        next_frame().await
    }
    println!("Final ai:\n{}", best_ai);
    println!("Final score: {}", score);
}
