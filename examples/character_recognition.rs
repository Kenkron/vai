extern crate rand;
use rand::random;

use macroquad::{text::draw_text, window::next_frame};
use macroquad::prelude::*;
use nalgebra as na;

const IMAGE_SIZE: usize = 24;
const PIXEL_COUNT: usize = IMAGE_SIZE * IMAGE_SIZE;

fn create_random_render(render_target: RenderTarget) 
-> usize{
    // 0..32, 0..32 camera
    let mut cam = Camera2D::from_display_rect(Rect::new(0.,0.,IMAGE_SIZE as f32,-(IMAGE_SIZE as f32)));
    cam.render_target = Some(render_target);
    set_camera(&cam);
    clear_background(BLACK);
    let font_size = 16.0 + random::<f32>() * (IMAGE_SIZE as f32 - 16.0);
    let x =  random::<f32>() * (IMAGE_SIZE as f32 - font_size * 0.5);
    let y =  random::<f32>() * (font_size * 0.5 - IMAGE_SIZE as f32);
    let number = (10. * random::<f32>()).floor() as usize;
    let number_string = "0123456789";
    draw_text(&number_string[number..number+1], x, y, font_size, WHITE);
    set_default_camera();
    return number;
}
fn extract_pixels(texture: &Texture2D)
-> na::SMatrix<f32, PIXEL_COUNT, 1>{
    let pixels = texture.get_texture_data().bytes;
    let mut result = na::SMatrix::<f32, PIXEL_COUNT, 1>::zeros();
    for i in 0..PIXEL_COUNT {
        result[i] = pixels[i * 4] as f32 / 256.;
    }
    return result;
}

fn score(test_number: usize, actual: &na::SMatrix<f32, 10, 1>)
-> f32 {
    let mut expectation = na::SMatrix::<f32, 10, 1>::zeros();
    expectation[test_number] = 1.0;
    let difference = expectation - actual;
    return difference.dot(&difference);
}

fn window_conf() -> Conf {
    Conf {
        window_title: "World's Worst AI".to_owned(),
        fullscreen: false,
        window_width: (IMAGE_SIZE * 10) as i32,
        window_height: (IMAGE_SIZE * 10) as i32 + 100,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let render_target = render_target(IMAGE_SIZE as u32, IMAGE_SIZE as u32);
    render_target.texture.set_filter(FilterMode::Nearest);
    let mut generation = 0;
    let mut inputs = extract_pixels(&render_target.texture);
    let mut pause = false;
    let mut step = false;

    const H: usize = 1;
    const C: usize = 32;

    let mut best_ai = vai::VAI::<PIXEL_COUNT,10,C,H>::new();
    let mut test_ai = vai::VAI::<PIXEL_COUNT,10,C,H>::new();
    let mut test_number = create_random_render(render_target);
    let mut best_outputs = na::SMatrix::<f32, 10, 1>::zeros();
    let mut best_score = f32::INFINITY;
    let mut test_score = best_score;
    loop {
        pause ^= is_key_pressed(KeyCode::Space);
        step ^= is_key_pressed(KeyCode::Enter);
        if is_key_pressed(KeyCode::P) {
            println!("Best ai: {}", best_ai);
        }

        if !pause || step {
            generation += 1;
            step = false;
            test_ai = best_ai.create_variant(random());
            let tests_per_eval = 1;
            test_score = 0.;
            //let mut new_best_score = 0.;
            for _ in 0..tests_per_eval {
                test_number = create_random_render(render_target);
                inputs = extract_pixels(&render_target.texture);
                test_score += score(test_number, &test_ai.process(&inputs));
                best_outputs = best_ai.process(&inputs);
                //new_best_score += score(test_number, &best_outputs);
            }
            if test_score < best_score {
                best_score = test_score;
                best_ai = test_ai;
            }
        }

        clear_background(DARKGRAY);
        draw_texture_ex(
            render_target.texture,
            0.,
            0.,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2((IMAGE_SIZE * 10) as f32, (IMAGE_SIZE * 10) as f32)),
                ..Default::default()
            },
        );
        draw_text(format!("Generation: {}", generation).as_str(), 10., (IMAGE_SIZE * 10) as f32 + 20., 20., WHITE);
        draw_text(format!("Best Score: {}", best_score).as_str(), 10., (IMAGE_SIZE * 10) as f32 + 40., 20., WHITE);
        let mut best_readable = String::new();
        for i in 0..10 {
            best_readable = format!("{} {}", best_readable, best_outputs[i]);
        }
        draw_text(best_readable.as_str(), 10., (IMAGE_SIZE * 10) as f32 + 60., 20., WHITE);
        let mut best_choice = 0;
        for i in 0..10 {
            if best_outputs[i] > best_outputs[best_choice] {
                best_choice = i;
            }
        }
        draw_text(format!("Best Choice: {}", best_choice).as_str(), 10., (IMAGE_SIZE * 10) as f32 + 80., 20., WHITE);
        next_frame().await
    }
}