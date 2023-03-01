use macroquad::prelude::*;

use na::ComplexField;
//mod mesh;
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

fn test_one(input: &na::Matrix3x1<f32>, layer1: &na::Matrix3<f32>, layer2: &na::Matrix1x3<f32>) -> f32 {
    let mut out1 = layer1 * input;
    out1.apply(|x| *x = x.max(0.));
    let out2 = layer2 * out1;
    return out2[0];
}

fn test(layer1: &na::Matrix3<f32>, layer2: &na::Matrix1x3<f32>) -> f32{
    let mut miss_outer = 0.0;
    let mut miss_inner = 0.0;
    for _ in 0..10000 {
        let mut input = na::Matrix3x1::<f32>::new_random() * 5.0;
        input[0] = 5.0;
        let out = test_one(&input, layer1, layer2);
        let actual = outside(input[1], input[2]);
        if actual > 0. {
            if !(out > 0.) {
                miss_outer += 1.;
            }
        } else {
            if out > 0. {
                miss_inner += 1.;
            }
        }
    }
    return miss_outer * miss_outer + miss_inner * miss_inner;
}

fn test_render(layer1: &na::Matrix3<f32>, layer2: &na::Matrix1x3<f32>) -> f32{
    let mut miss_outer = 0.0;
    let mut miss_inner = 0.0;
    rand::srand((mouse_position().0 + mouse_position().1 * screen_width()) as u64);
    for _ in 0..1000 {
        //let input = na::Matrix2x1::<f32>::new_random() * 5.0;
        let x = rand::gen_range(0., 5.);
        let y = rand::gen_range(0., 5.);
        let input = na::Matrix3x1::new(5.0, x, y);
        let out = test_one(&input, layer1, layer2);
        let mut marker = GRAY;
        let actual = outside(input[1], input[2]);
        if actual > 0. {
            if out > 0. {
                marker = PURPLE;
            } else {
                marker = RED;
                miss_outer += 1.;
            }
        } else {
            if out > 0. {
                marker = YELLOW;
                miss_inner += 1.;
            } else {
                marker = GREEN;
            }
        }
        draw_circle(input[1] * 50., input[2] * 50., 2., marker);
    }
    return (miss_outer * miss_outer + miss_inner * miss_inner) * 10.;
}

#[macroquad::main("World's Worst AI")]
async fn main() {

    let mut layer1 = na::Matrix3::<f32>::new_random();
    layer1 = layer1.add_scalar(-0.5).scale(5.);
    let mut layer2 = na::Matrix1x3::<f32>::new_random();
    layer2 = layer2.add_scalar(-0.5).scale(5.);
    let mut score = test(&layer1, &layer2);

    println!("Starting matricies:\n{}\n{}", layer1, layer2);
    println!("Starting score: {}", score);

    let mut t1 = layer1.clone();
    let mut t2 = layer2.clone();
    let mut st = test(&t1, &t2);
    loop {
        if !is_key_down(KeyCode::Space) {
            // Some mutations will be big, some small
            let rand_scale = rand::gen_range(0.0, 2.) * rand::gen_range(0.0, 2.);
            let mut l1 = na::Matrix3::<f32>::new_random();
            l1 = l1.add_scalar(-0.5).scale(rand_scale);
            l1 += layer1;
            t1 = l1.clone();
            let mut l2 = na::Matrix1x3::<f32>::new_random();
            l2 = l2.add_scalar(-0.5).scale(rand_scale);
            l2 += layer2;
            t2 = l2.clone();
            let s = test(&l1, &l2);
            st = s;
            if s < score {
                draw_text(&format!("Score was better: {}", s), 10., 260., 20., WHITE);
                //println!("Score was better: {}", s);
                layer1 = l1;
                layer2 = l2;
                score = s;
            } else {
                draw_text(&format!("Score was worse: {}", s), 10., 260., 20., WHITE);
                //println!("Score was worse: {}", s);
            }
        }

        if is_key_down(KeyCode::B) {
            let r = test_render(&layer1, &layer2);
            draw_text(&format!("Best score: {}\n", r), 10., 275., 20., WHITE);
        } else {
            let r = test_render(&t1, &t2);
            draw_text(&format!("Test score: {}\n", r), 10., 275., 20., WHITE);
        }
        
        if is_key_down(KeyCode::Escape) {
            break;
        }

        next_frame().await
    }
    println!("Final matricies:\n{}\n{}", layer1, layer2);
    println!("Final score: {}", score);
}
