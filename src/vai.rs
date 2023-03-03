use std::{fmt::Display, fs::File};
use std::io::{Write, self, Lines};

use na::SMatrix;
use nalgebra as na;
use rand;

/**Maps a 0-1 valud to +- infinity, with low weighted extremes*/
pub fn infinite_map(input: f32) -> f32 {
    if input <= 0. || input >= 1. {
        return 0.;
    }
    let x = input - 0.5;
    return 0.5 * x/(0.25 - x*x).sqrt();
}

/**Creates a random variation of a matrix*/
pub fn create_variant<const R: usize, const C: usize>(original: &na::SMatrix<f32, R, C>, intensity: f32)
-> na::SMatrix<f32, R, C> {
    // Adding two variances increases the odds of small changes,
    // but also makes larger infrequent changes possible
    let mut variance = na::SMatrix::<f32, R, C>::new_random();
    variance.apply(|x| *x = infinite_map(*x));
    return original + variance.scale(intensity);
}

fn rand_index(len: usize) -> usize {
    (rand::random::<f32>() * (len + 1) as f32).floor() as usize
}

pub fn write_matrix<const R: usize, const C: usize>(
    matrix: &SMatrix<f32, R, C>,
    file: &mut File)
-> std::io::Result<()> {
    for r in 0..R {
        let row = matrix.row(r);
        for c in 0..C {
            write!(file, "{} ", row.get(c).unwrap())?;
        }
        writeln!(file, "")?;
    }
    writeln!(file, "")?;
    return Ok(());
}

pub fn read_matrix<const R: usize, const C: usize>(
    lines: &mut Lines<std::io::BufReader<File>>)
-> std::io::Result<SMatrix<f32, R, C>> {
    let mut result = SMatrix::<f32, R, C>::zeros();
    let mut r = 0;
    for line in lines {
        let line = line?;
        let vals: Vec<&str> = line.split_whitespace().collect();
        if vals.len() == 0 {
            // skip empty lines
            continue;
        }
        if vals.len() != C {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Wrong number of columns"));
        }
        for c in 0..vals.len() {
            result.row_mut(r)[c] = match vals[c].parse() {
                Ok(x) => {x},
                Err(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Invalid number"));
                },
            };
        }
        r += 1;
        if r == R {
            break;
        }
    }
    if r != R {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Wrong number of rows"));
    }
    return Ok(result);
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
        let fields = I * C + C * C * EXTRA_LAYERS + C * O;
        let s_intensity = intensity/(1.0 + fields as f32);
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
            let fields = C * C;
            extra_hidden_layers[layer] = create_variant(&extra_hidden_layers[layer], intensity/fields as f32);
        } else if layer == extra_hidden_layers.len() {
            let fields = I * C;
            result.input_layer = create_variant(&result.input_layer, intensity/fields as f32);
        } else {
            let fields = C * O;
            result.output_layer = create_variant(&result.output_layer, intensity/fields as f32);
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
    pub fn read(lines: &mut Lines<std::io::BufReader<File>>) -> std::io::Result<Self> {
        let mut result = Self::new();
        result.input_layer = read_matrix(lines)?;
        for matrix in &mut result.extra_hidden_layers {
            *matrix = read_matrix(lines)?;
        }
        result.output_layer = read_matrix(lines)?;
        return Ok(result);
    }
    pub fn write(&self, file: &mut File) -> std::io::Result<()> {
        write_matrix(&self.input_layer, file)?;
        for matrix in &self.extra_hidden_layers {
            write_matrix(matrix, file)?;
        }
        write_matrix(&self.output_layer, file)?;
        return Ok(());
    }
}