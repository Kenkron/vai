extern crate rand;

use std::io::BufRead;

use rand::random;
use rayon::prelude::*;

use macroquad::{text::draw_text, window::next_frame};
use macroquad::prelude::*;
use nalgebra as na;

const IMAGE_SIZE: usize = 12;
const PIXEL_COUNT: usize = IMAGE_SIZE * IMAGE_SIZE;
const INPUTS: usize = PIXEL_COUNT + 1;

fn create_random_render(render_target: RenderTarget, font: &Font) 
-> usize{
    // 0..32, 0..32 camera
    let mut cam = Camera2D::from_display_rect(Rect::new(0.,0.,IMAGE_SIZE as f32,-(IMAGE_SIZE as f32)));
    cam.render_target = Some(render_target);
    set_camera(&cam);
    clear_background(BLACK);
    let max_font_size = 10.0;
    let font_size = max_font_size + random::<f32>() * (IMAGE_SIZE as f32 - max_font_size);
    let x =  random::<f32>() * (IMAGE_SIZE as f32 - font_size);
    let y =  random::<f32>() * (font_size - IMAGE_SIZE as f32);
    let number = (10. * random::<f32>()).floor() as usize;
    let number_string = "0123456789";
    draw_text_ex(&number_string[number..number+1], x, y,
            TextParams {
            font_size: max_font_size.floor() as u16,
            font_scale: 1.25 * font_size / max_font_size.floor(),
            font: *font,
            ..Default::default()
        });
    set_default_camera();
    return number;
}

fn extract_pixels(texture: &Texture2D)
-> na::SMatrix<f32, INPUTS, 1>{
    let pixels = texture.get_texture_data().bytes;
    let mut result = na::SMatrix::<f32, INPUTS, 1>::zeros();
    for i in 0..PIXEL_COUNT {
        result[i] = pixels[i * 4] as f32 / 256.;
    }
    result[INPUTS - 1] = 1.0; // bias
    return result;
}

fn score(test_number: usize, actual: &na::SMatrix<f32, 10, 1>)
-> f32 {
    let number_val = actual[test_number];
    let mut nth = 0.0;
    for i in 0..10 {
        if i != test_number && actual[i] >= number_val {
            nth += 1.0;
        }
    }
    return nth;
}

fn window_conf() -> Conf {
    Conf {
        window_title: "World's Worst AI".to_owned(),
        fullscreen: false,
        window_width: (IMAGE_SIZE * 20) as i32,
        window_height: (IMAGE_SIZE * 20) as i32 + 250,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let font = load_ttf_font("./examples/OpenSans-Regular.ttf").await.unwrap();
    let render_target = render_target(IMAGE_SIZE as u32, IMAGE_SIZE as u32);
    render_target.texture.set_filter(FilterMode::Nearest);
    let mut generation = 0;
    let mut inputs = na::SMatrix::<f32, INPUTS, 1>::zeros();
    let mut pause = true;
    let mut step = false;

    const H: usize = 0;
    const C: usize = 32;

    let mut best_ais = [vai::VAI::<INPUTS,10,C,H>::new().create_variant(1.0); 5];
    let mut test_number = create_random_render(render_target, &font);
    let mut best_outputs = na::SMatrix::<f32, 10, 1>::zeros();
    let mut best_score = f32::INFINITY;
    loop {
        pause ^= is_key_pressed(KeyCode::Space);
        step ^= is_key_pressed(KeyCode::Enter);
        if is_key_pressed(KeyCode::P) {
            println!("Best ai: {}", best_ais[0]);
        }

        if !pause || step {
            generation += 1;
            step = false;
            // Tuple: (ai, score)
            let mut test_ais = vec![(vai::VAI::<INPUTS,10,C,H>::new(), 0.0 as f32); 100];
            for i in 0..test_ais.len() {
                if i < best_ais.len() {
                    test_ais[i].0 = best_ais[i];
                } else {
                    let intensity = random::<f32>() * (i / best_ais.len()) as f32;
                    test_ais[i].0 = best_ais[i % best_ais.len()].create_variant(intensity);
                }
            }
            let tests_per_generation = 1000;
            for _ in 0..tests_per_generation {
                test_number = create_random_render(render_target, &font);
                inputs = extract_pixels(&render_target.texture);
                test_ais.par_iter_mut().for_each(|x| {
                    x.1 += score(test_number, &x.0.process(&inputs));
                });
            }
            test_ais.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            best_score = test_ais[0].1;
            for i in 0..best_ais.len() {
                best_ais[i] = test_ais[i].0;
            }
            // recalculate for debugging
            best_outputs = best_ais[0].process(&inputs);
        }

        clear_background(DARKGRAY);
        draw_texture_ex(
            render_target.texture,
            0.,
            0.,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2((IMAGE_SIZE * 20) as f32, (IMAGE_SIZE * 20) as f32)),
                ..Default::default()
            },
        );
        draw_text(format!("Generation: {}", generation).as_str(), 10., (IMAGE_SIZE * 20) as f32 + 20., 20., WHITE);
        draw_text(format!("Best Score: {}", best_score).as_str(), 10., (IMAGE_SIZE * 20) as f32 + 40., 20., WHITE);
        //let mut best_readable = String::new();
        let mut best_tuple = Vec::<(usize, f32)>::new();
        for i in 0..10 {
            //best_readable = format!("{} {}", best_readable, best_outputs[i]);
            best_tuple.push((i, best_outputs[i]));
        }
        best_tuple.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        for i in 0..best_tuple.len() {
            let mut color = ORANGE;
            if best_tuple[i].0 == test_number {
                if i == 0 {
                    color = GREEN;
                } else {
                    color = RED;
                }
            }
            draw_text(
                format!("{}: {}", best_tuple[i].0, best_tuple[i].1).as_str(),
                10.0,
                (IMAGE_SIZE * 20 + 60 + i * 20) as f32,
                20.0,
                color);
        }
        if is_key_pressed(KeyCode::S) {
            match std::fs::File::create("./save.vai") {
                Ok(mut file) => {
                    match best_ais[0].write(&mut file) {
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
                    match vai::VAI::<INPUTS, 10, C, H>::read(
                        &mut std::io::BufReader::new(file).lines()) {
                        Ok(result) => {
                            best_ais[0] = result;
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
}