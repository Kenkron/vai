use std::f32::consts::{PI, E};
use std::io::BufRead;
use std::iter::zip;
use std::sync::{Arc, Mutex};

use macroquad::prelude::{is_key_down, is_key_pressed, Conf, KeyCode, draw_circle, draw_rectangle, draw_text};
use macroquad::prelude::{RED, WHITE, BLUE, BLACK};
use macroquad::window::next_frame;
use rand;
use rayon::prelude::*;
use vai;

const HIDDEN: usize = 1;
const LAYERS: usize = 0;

type AI = vai::VAI::<3, 1, HIDDEN, LAYERS>;

 struct State {
    ai: AI,
    inputs: [f32; 3],
    target_timer: f32,
    lifetime: f32,
    score: f32
}

impl State {
    fn new(ai: AI) -> Self {
        Self {
            ai,
            inputs: [0.,0.,1.],
            target_timer: 0.,
            lifetime: 0.,
            score: 0.
        }
    }
    fn update(&mut self, dt: f32) {
        self.lifetime += dt;
        self.target_timer += dt;

        let timer = self.target_timer;
        if self.ai.process_slice(&self.inputs)[0] > 0. {
            // The closer to the center, the better the score
            self.score += timer.sin().abs()/(0.1 + timer.cos().abs());
            self.target_timer = 0.;
        }

        // Let input 0 trigger when score would be low
        self.inputs[0] = if self.target_timer.sin().abs() < 0.25 { 1.0 } else { 0.0 };

        // This will make input 1 tend towards input 0 at a rate of 1/e^x
        self.inputs[1] += (self.inputs[0] - self.inputs[1]) * dt;
    }
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

fn draw_simulation(simulation: &State) {
    let width = 300.0;
    draw_rectangle(0., 0., width, 300., BLACK);
    let target_y = 150.;
    draw_circle((simulation.target_timer * width / PI) % width, target_y, 60., WHITE);
    draw_circle((simulation.target_timer * width / PI) % width, target_y, 40., RED);
    draw_circle((simulation.target_timer * width / PI) % width, target_y, 20., WHITE);
    if simulation.target_timer < 0.1 {
        draw_rectangle(width/2. - 2.0, target_y, 4.0, 300. - target_y, BLUE);
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut step = false;
    let mut paused = true;
    let mut quiet = false;
    let mut generation = 0;
    let generation_duration = 500;
    // box the states so they can be sorted quickly
    let mut simulations = Vec::<Box<State>>::new();
    for i in 0..200 {
        simulations.push(Box::new(State::new(AI::new_deterministic(i))));
    }
    let mut frame_count: usize = 0;
    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
        step ^= is_key_pressed(KeyCode::Enter);
        paused ^= is_key_pressed(KeyCode::Space);
        quiet ^= is_key_pressed(KeyCode::Q);
        if is_key_pressed(KeyCode::P) {
            println!("best ai: {}", simulations.last().unwrap().ai);
        }

        if frame_count >= generation_duration {
            // replace the worst simulations with the best simulations
            // and reset their scores
            simulations.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
            let mid = simulations.len()/2;
            let (worst, best) = simulations.split_at_mut(mid);
            for (good, bad) in zip(best, worst) {
                bad.ai = good.ai.create_variant(0.1);
                bad.score = 0.0;
                good.score = 0.0;
            }
            generation += 1;
            frame_count = 0;
        }

        draw_simulation(simulations.last().unwrap().as_ref());

        if !paused {
            simulations.par_iter_mut().for_each(|simulation| {
                simulation.update(1./120.);
            });
            frame_count += 1;
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
                    match simulations.last().unwrap().ai.write(&mut file) {
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
                    match AI::read(&mut std::io::BufReader::new(file).lines())
                    {
                        Ok(result) => {
                            for simulation in &mut simulations {
                                simulation.ai = result.clone();
                            }
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
    while frame_count < generation_duration {
        simulations.par_iter_mut().for_each(|simulation| {
            simulation.update(1./60.);
        });
        frame_count += 1;
    }
    simulations.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
    println!("Final ai:\n{}", simulations.last().unwrap().ai);
    println!("Final score: {}", simulations.last().unwrap().score);
}
