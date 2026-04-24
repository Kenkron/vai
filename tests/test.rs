#![allow(clippy::needless_return)]

use nalgebra as na;
use std::assert;

use rand::rngs::StdRng;
use rand::{self, SeedableRng};

fn linear_score<const C: usize, const H: usize>(ai: &vai::VAI<2, 1, C, H>) -> f32 {
    let m = -2.0;
    let b = 0.75;
    let mut score = 0.0;
    for i in 0..=10 {
        let x = 0.1 * i as f32;
        let target_y = m * x + b;
        let ai_y = ai.process(&na::SMatrix::<f32, 2, 1>::new(x, 1.0))[0];
        println!("target: {}, ai: {}", target_y, ai_y);
        score += (target_y - ai_y) * (target_y - ai_y);
    }
    return score;
}

fn non_linear_score<const C: usize, const H: usize>(ai: &vai::VAI<2, 1, C, H>) -> f32 {
    let mut score = 0.0;
    for i in 0..=10 {
        let x = 0.1 * i as f32;
        let target_y = 2. * (x - 0.5) * (x - 0.5);
        let ai_y = ai.process(&na::SMatrix::<f32, 2, 1>::new(x, 1.0))[0];
        score += (target_y - ai_y) * (target_y - ai_y);
    }
    return score;
}

#[test]
fn linear_test() {
	let mut rng = StdRng::seed_from_u64(0);
    let mut best_ai = vai::VAI::<2, 1, 4, 0>::new();
    let mut best_score = linear_score(&best_ai);
    let initial_score = best_score;
    println!("Initial Score: {}", initial_score);
    for i in 0..1000 {
        let test_ai = best_ai.create_variant(1.0, &mut rng);
        let test_score = linear_score(&test_ai);
        if test_score < best_score {
            best_ai = test_ai;
            best_score = test_score;
            println!("Best Score: {}: {}", i, best_score);
        }
    }
    println!("Best Score: {}", best_score);
    println!("Best AI: {}", best_ai);
    assert!(best_score < initial_score * 0.1);
}

#[test]
fn non_linear_test() {
	let mut rng = StdRng::seed_from_u64(0);
    let mut best_ai = vai::VAI::<2, 1, 4, 1>::new();
    let mut best_score = non_linear_score(&best_ai);
    let initial_score = best_score;
    println!("Initial Score: {}", initial_score);
    for i in 0..1000 {
        let test_ai = best_ai.create_variant(1.0, &mut rng);
        let test_score = non_linear_score(&test_ai);
        if test_score < best_score {
            best_ai = test_ai;
            best_score = test_score;
            println!("Best Score: {}: {}", i, best_score);
        }
    }
    println!("Best Score: {}", best_score);
    println!("Best AI: {}", best_ai);
    assert!(best_score < initial_score * 0.1);
}

#[test]
fn backpropogation() {
	let mut ai = vai::VAI::<1, 1, 1, 0>::new();
	ai.input_connections[(0, 0)] = 1.0;
	ai.output_connections[(0, 0)] = 1.0;
	println!("Initial NN: {}", ai);
	ai = ai.backpropogate(&na::vector![1.0], &na::vector![3.0]);
	println!("Backpropogated NN: {}", ai);
	assert!((ai.input_connections[(0, 0)] - 2.0).abs() < 0.1);
	assert!((ai.output_connections[(0, 0)] - 2.0).abs() < 0.1);
}
