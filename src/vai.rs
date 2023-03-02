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

/**Maps a 0-1 valud to +- infinity, with low weighted extremes*/
pub fn infinite_map(input: f32) -> f32 {
    if input <= 0. || input >= 1. {
        return 0.;
    }
    let x = input - 0.5;
    return x/(0.25 - x*x).sqrt();
}

fn rand_index(len: usize) -> usize {
    (rand::random::<f32>() * (len + 1) as f32).floor() as usize
}

/**A Very Artificial Intelligence.
 * I: Number of inputs. You should probably include a constant bias.
 * O: Number of outputs
 * C: Complexity of (number of nodes in) hidden layers
 * EXTRA_LAYERS: There is always at least one hidden layer. This number adds more.
 */
#[derive(Clone)]
pub struct
VAI<const I: usize, const O: usize, const C: usize, const EXTRA_LAYERS: usize> {
    input_layer: na::SMatrix::<f32, C, I>,
    extra_hidden_layers: [na::SMatrix::<f32, C, C>; EXTRA_LAYERS],
    output_layer: na::SMatrix::<f32, O, C>
}

impl<const I: usize, const O: usize, const HIDDEN_LAYERS: usize, const LAYER_SIZE: usize>
Display for VAI<I, O, HIDDEN_LAYERS, LAYER_SIZE> {
    fn fmt(&self,f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut result = String::new();
        result += self.input_layer.to_string().as_str();
        for mat in &self.extra_hidden_layers {
            result += mat.to_string().as_str();
        }
        result += self.output_layer.to_string().as_str();
        write!(f, "{}", result)
    }
}

impl<const I: usize, const O: usize, const C: usize, const EXTRA_LAYERS: usize>
VAI<I, O, C, EXTRA_LAYERS> {
    pub fn new() -> Self {
        Self {
            input_layer: na::SMatrix::<f32, C, I>::zeros(),
            extra_hidden_layers: [na::SMatrix::<f32, C, C>::zeros(); EXTRA_LAYERS],
            output_layer: na::SMatrix::<f32, O, C>::zeros()
        }
    }
    pub fn create_variant(&self, intensity: f32) -> Self {
        let mut result = self.clone();
        let s_intensity = intensity/(1.0 + result.extra_hidden_layers.len() as f32);
        result.input_layer = create_variant(&result.input_layer, intensity);
        for mat in &mut result.extra_hidden_layers {
            *mat = create_variant(&mat, s_intensity);
        }
        result.output_layer = create_variant(&result.output_layer, s_intensity);
        return result;
    }
    pub fn create_layer_variant(&self, intensity: f32) -> Self {
        let mut result = self.clone();
        let extra_hidden_layers = &mut result.extra_hidden_layers;
        let layer = rand_index(extra_hidden_layers.len() + 2);
        if layer < extra_hidden_layers.len() {
            extra_hidden_layers[layer] = create_variant(&extra_hidden_layers[layer], intensity);
        } else if layer == extra_hidden_layers.len() {
            result.input_layer = create_variant(&result.input_layer, intensity);
        } else {
            result.output_layer = create_variant(&result.output_layer, intensity);
        } 
        return result;
    }
    pub fn process(&self, inputs: &na::SMatrix<f32, I, 1>) -> na::SMatrix<f32, O, 1> {
        let mut intermediate = self.input_layer * inputs;
        // Apply relu
        intermediate.apply(|x| *x = x.max(0.));
        for mat in &self.extra_hidden_layers {
            intermediate = mat * intermediate;
            // Apply relu
            intermediate.apply(|x| *x = x.max(0.));
        }
        return self.output_layer * intermediate;
    }
}