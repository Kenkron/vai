use std::fmt::Display;

use nalgebra as na;
use rand;

pub fn create_variant<const R: usize, const C: usize>(original: &na::SMatrix<f32, R, C>, intensity: f32)
-> na::SMatrix<f32, R, C> {
    // Adding two variances increases the odds of small changes,
    // but also makes larger infrequent changes possible
    let mut variance = na::SMatrix::<f32, R, C>::new_random();
    variance.apply(|x| *x = infinite_map(*x));
    return original + variance.scale(intensity);
}

/**Maps a 0-1 valud to +- infinity, with low weighted extremes
 */
pub fn infinite_map(input: f32) -> f32 {
    let x = input - 0.5;
    return x/(0.25 - x*x).sqrt();
}

fn rand_index(len: usize) -> usize {
    (rand::random::<f32>() * (len + 1) as f32).floor() as usize
}

#[derive(Clone)]
pub struct VAI<const I: usize, const O: usize, const HIDDEN_LAYERS: usize> {
    hidden_layers: [na::SMatrix::<f32, I, I>; HIDDEN_LAYERS],
    end_layer: na::SMatrix::<f32, O, I>
}

impl<const I: usize, const O: usize, const HIDDEN_LAYERS: usize> Display for VAI<I, O, HIDDEN_LAYERS> {
    fn fmt(&self,f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut result = String::new();
        for mat in &self.hidden_layers {
            result += mat.to_string().as_str();
        }
        result += self.end_layer.to_string().as_str();
        write!(f, "{}", result)
    }
}

impl<const I: usize, const O: usize, const HIDDEN_LAYERS: usize> VAI<I, O, HIDDEN_LAYERS> {
    pub fn new() -> Self {
        Self {
            hidden_layers: [na::SMatrix::<f32, I, I>::zeros(); HIDDEN_LAYERS],
            end_layer: na::SMatrix::<f32, O, I>::zeros()
        }
    }
    pub fn create_variant(&self, intensity: f32) -> Self {
        let mut result = self.clone();
        let s_intensity = intensity/(1.0 + result.hidden_layers.len() as f32);
        for mat in &mut result.hidden_layers {
            *mat = create_variant(&mat, s_intensity);
        }
        result.end_layer = create_variant(&result.end_layer, s_intensity);
        return result;
    }
    pub fn create_layer_variant(&self, intensity: f32) -> Self {
        let mut result = self.clone();
        let hidden_layers = &mut result.hidden_layers;
        let layer = rand_index(hidden_layers.len() + 1);
        if layer < hidden_layers.len() {
            hidden_layers[layer] = create_variant(&hidden_layers[layer], intensity);
        } else {
            result.end_layer = create_variant(&result.end_layer, intensity);
        }
        return result;
    }
    pub fn process(&self, inputs: &na::SMatrix<f32, I, 1>) -> na::SMatrix<f32, O, 1> {
        let mut intermediate = inputs.clone();
        for mat in &self.hidden_layers {
            intermediate = mat * intermediate;
            // Apply relu
            intermediate.apply(|x| *x = x.max(0.));
        }
        return self.end_layer * intermediate;
    }
}