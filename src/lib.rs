use rand;
pub mod vai;
pub mod vaid;
pub use vai::VAI;
pub use vaid::VAID;


/// Maps a 0-1 valud to +- infinity, with low weighted extremes
pub fn infinite_map(input: f32) -> f32 {
    if input <= 0. || input >= 1. {
        return 0.;
    }
    let x = input - 0.5;
    return 0.5 * x / (0.25 - x * x).sqrt();
}

/// Gets a random index less than the provided length
pub fn rand_index(len: usize) -> usize {
    (rand::random::<f32>() * (len + 1) as f32).floor() as usize
}