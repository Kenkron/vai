#![allow(clippy::needless_return)]

use std::io::{self, BufRead};
use std::sync::{Arc, Mutex};
use std::sync::mpsc;
use std::thread;

use nalgebra as na;
use rand::rngs::StdRng;
use rand::{self, Rng, SeedableRng};
use rayon::prelude::*;

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
    return w1 * w2;
}

fn test<const I: usize, const C: usize, const E: usize>(
    ai: &vai::VAI<I, 1, C, E>,
    random: &mut crate::rand::rngs::StdRng,
    debug: impl Send + Fn(f32, f32, usize),
) -> f32 {
    let tests = 1000;
    let random_lock = Arc::new(Mutex::new(random));
    let debug_lock = Arc::new(Mutex::new(debug));
    let outer = Arc::new(Mutex::new(0.));
    let miss_outer = Arc::new(Mutex::new(0.));
    let inner = Arc::new(Mutex::new(0.));
    let miss_inner = Arc::new(Mutex::new(0.));
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
        let out = ai.process(&input)[0];
        let actual = outside(x, y);
        let path: usize;
        if actual > 0. {
            *outer.lock().unwrap() += 1.;
            if out > 0. {
                path = 2;
            } else {
                path = 1;
                *miss_outer.lock().unwrap() += 1.;
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
        outer_cost = *miss_outer.lock().unwrap() / *outer.lock().unwrap();
        outer_cost *= outer_cost;
    }
    let mut inner_cost = 0.;
    if *inner.lock().unwrap() > 0. {
        inner_cost = *miss_inner.lock().unwrap() / *inner.lock().unwrap();
        inner_cost *= inner_cost;
    }
    return (inner_cost + outer_cost) * 0.5;
}

struct SimState {
	rng: StdRng,
    best_ai: vai::VAI::<3, 1, 16, 1>,
    score: f32,
    test_ai: vai::VAI::<3, 1, 16, 1>,
    tweaking: bool,
    generation: usize,
    paused: bool, 
    step:bool,
	show_best: bool
}

fn main() {

	// read stdin on a separate thread
	let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        loop {
            let mut buffer = String::new();
            io::stdin().read_line(&mut buffer).unwrap();
            tx.send(buffer).unwrap();
        }
    });

   	let mut rng = StdRng::seed_from_u64(0);
   	let mut best_ai = vai::VAI::<3, 1, 16, 1>::new_deterministic(0);
   	let mut score = test(&best_ai, &mut rng, |_, _, _| ());
   	let mut test_ai = best_ai.clone();
   	let mut tweaking = false;
   	let mut backpropogate = false;
   	let mut generation = 0;
   	let mut paused = true;
   	let mut step = false;
   	let mut quiet = false;
   	let mut show_best = true;
    
    println!("Starting score: {}", score);
    println!("Starting ai:\n{}", best_ai);

    loop {
        rng = StdRng::seed_from_u64(generation);
        let mut test_score = 0.0;
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
                test_score = test(&test_ai, &mut rng, |_, _, _| ());
                let re_check = test(&best_ai, &mut rng, |_, _, _| ());
                // Constantly update best score based on new data
                score = (score * 15. + re_check) * 0.0625;
                if test_score < score {
                    best_ai = test_ai.clone();
                    score = test_score;
                }
            }
        }
        println!("Generation: {}, Best: {}, Test: {}", generation, score, test_score);
    	if let Ok(raw_input) = rx.try_recv() {
    		let input = raw_input.to_lowercase();
	        if input.contains("quit") {
    	        break;
            }
	        tweaking ^= input.contains('t');
	        step ^= input.contains('n');
	        paused ^= input.contains(' ');
	        quiet ^= input.contains('q');
	        show_best ^= input.contains('b');
	        if input.contains('p') {
	            println!("best ai: {}", best_ai)
	        }
	        if input.contains('s') {
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
	        if input.contains('o') {
	            match std::fs::File::open("./dotfield-save.vai") {
	                Ok(file) => {
	                    match vai::VAI::<3, 1, 16, 1>::read(&mut std::io::BufReader::new(file).lines())
	                    {
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
        }
    }
    println!("Final ai:\n{}", best_ai);
    println!("Final score: {}", score);
}
