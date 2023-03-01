use nalgebra as na;
use nalgebra::Dim;
use rand;

#[derive(Clone)]
pub struct VAI<const I: usize, const O: usize, const HIDDEN_LAYERS: usize> {
    hidden_layers: [na::SMatrix::<f32, I, I>; HIDDEN_LAYERS],
    end_layer: na::SMatrix::<f32, O, I>
}

pub fn create_variant<const R: usize, const C: usize>(original: &na::SMatrix<f32, R, C>)
-> na::SMatrix<f32, R, C> {
    let result = original.clone();
    return result;
}

fn rand_index(len: usize) -> usize {
    (rand::random::<f32>() * (len + 1) as f32).floor() as usize
}

impl<const I: usize, const O: usize, const HIDDEN_LAYERS: usize> VAI<I, O, HIDDEN_LAYERS> {
    pub fn new() -> Self {
        Self {
            hidden_layers: [na::SMatrix::<f32, I, I>::identity(); HIDDEN_LAYERS],
            end_layer: na::SMatrix::<f32, O, I>::identity()
        }
    }
    pub fn create_variant(original: &Self) -> Self {
        let mut result = original.clone();
        for mat in &mut result.hidden_layers {
            *mat = create_variant(&mat);
        }
        result.end_layer = create_variant(&result.end_layer);
        return result;
    }
    pub fn create_layer_variant(original: &Self) -> Self {
        let mut result = original.clone();
        let hidden_layers = &mut result.hidden_layers;
        let layer = rand_index(hidden_layers.len() + 1);
        if layer < hidden_layers.len() {
            hidden_layers[layer] = create_variant(&hidden_layers[layer]);
        } else {
            result.end_layer = create_variant(&result.end_layer);
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