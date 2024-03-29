#![allow(clippy::needless_return)]

use std::cmp::Ordering;
use std::io::{self, Lines, Write};
use std::{fmt::Display, fs::File};

extern crate nalgebra as na;
use na::SMatrix;
extern crate rand;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use crate::{infinite_map, rand_index};

/// Creates a random variation of a matrix
/// * original - The matrix that will be varied
/// * intensity - The severity to which the matrix will be randomized.
///
/// Intensity affects the random distribution to favor smaller values, but the
/// resulting matrix can still be changed by an arbitrary amount.
pub fn create_variant<const R: usize, const C: usize>(
    original: &na::SMatrix<f32, R, C>,
    intensity: f32,
) -> na::SMatrix<f32, R, C> {
    // Adding two variances increases the odds of small changes,
    // but also makes larger infrequent changes possible
    let mut variance = na::SMatrix::<f32, R, C>::new_random();
    variance.apply(|x| *x = infinite_map(*x));
    return original + variance.scale(intensity);
}

/// Creates a random variation of a matrix using a provided StdRng
/// * original - The matrix that will be varied
/// * intensity - The severity to which the matrix will be randomized.
///
/// Intensity affects the random distribution to favor smaller values, but the
/// resulting matrix can still be changed by an arbitrary amount.
pub fn create_variant_stdrng<const R: usize, const C: usize>(
    rng: &mut StdRng,
    original: &na::SMatrix<f32, R, C>,
    intensity: f32,
) -> na::SMatrix<f32, R, C> {
    let mut result = original.clone_owned();
    result.apply(|x| *x += intensity * infinite_map(rng.gen::<f32>()));
    return result;
}

/// Writes a matrix to a file with space-delimited columns,
/// newline delimited rows, and a trailing newline.
/// * matrix - The matrix to write
/// * file - The file to write to
pub fn write_matrix<const R: usize, const C: usize>(
    matrix: &SMatrix<f32, R, C>,
    file: &mut File,
) -> std::io::Result<()> {
    for r in 0..R {
        let row = matrix.row(r);
        for c in 0..C {
            write!(file, "{} ", row.get(c).unwrap())?;
        }
        writeln!(file)?;
    }
    writeln!(file)?;
    return Ok(());
}

/// Reads a matrix from lines of a file with space-delimited columns,
/// and newline delimited rows. Empty (whitespace) lines are ignored.
/// * lines - A line iterator from which to read the matrix
/// (generally provided by BufReader::new(file).lines())
pub fn read_matrix<const R: usize, const C: usize>(
    lines: &mut Lines<std::io::BufReader<File>>,
) -> std::io::Result<SMatrix<f32, R, C>> {
    let mut result = SMatrix::<f32, R, C>::zeros();
    let mut r = 0;
    for line in lines {
        let line = line?;
        let vals: Vec<&str> = line.split_whitespace().collect();
        if vals.is_empty() {
            continue; // Skip empty lines
        }
        if vals.len() != C {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Wrong number of columns",
            ));
        }
        for (c, val) in vals.iter().enumerate() {
            result.row_mut(r)[c] = match val.parse() {
                Ok(x) => x,
                Err(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "Invalid number",
                    ));
                }
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
            "Wrong number of rows",
        ));
    }
    return Ok(result);
}

/// Very Artificial Intelligence.
/// * I: Number of inputs. You should probably include a constant bias.
/// * O: Number of outputs
/// * C: Complexity of (number of nodes in) hidden layers
/// * EXTRA_LAYERS: There is always at least one hidden layer. This number adds more.
#[derive(Clone)]
pub struct VAI<const I: usize, const O: usize, const C: usize, const EXTRA_LAYERS: usize> {
    pub rng: StdRng,
    pub input_connections: na::SMatrix<f32, C, I>,
    pub hidden_connections: [na::SMatrix<f32, C, C>; EXTRA_LAYERS],
    pub output_connections: na::SMatrix<f32, O, C>,
}

impl<const I: usize, const O: usize, const HIDDEN_LAYERS: usize, const LAYER_SIZE: usize> Display
    for VAI<I, O, HIDDEN_LAYERS, LAYER_SIZE>
{
    /// Concatenates the string representations of the input,
    /// hidden layer, and output matricies.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.input_connections)?;
        for mat in &self.hidden_connections {
            write!(f, "{}", mat)?;
        }
        write!(f, "{}", self.output_connections)
    }
}

impl<const I: usize, const O: usize, const C: usize, const EXTRA_LAYERS: usize> Default
    for VAI<I, O, C, EXTRA_LAYERS>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const I: usize, const O: usize, const C: usize, const EXTRA_LAYERS: usize>
    VAI<I, O, C, EXTRA_LAYERS>
{
    /// Creates a VAI with zeros for all connection weights
    pub fn new() -> Self {
        Self::new_deterministic(rand::random())
    }

    /// Creates a VAI with zeros for all connection weights,
    /// using a specific seed for random number generatoin.
    pub fn new_deterministic(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            input_connections: na::SMatrix::<f32, C, I>::zeros(),
            hidden_connections: [na::SMatrix::<f32, C, C>::zeros(); EXTRA_LAYERS],
            output_connections: na::SMatrix::<f32, O, C>::zeros(),
        }
    }

    /// Creates a random variant of this VAI
    /// * intensity - Scaler for the added randomness
    ///
    /// Intensity affects the random distribution to favor low magnitude
    /// values, but the result can still be changed by an arbitrary amount.
    /// Randomness is applied to each weight of each connection.
    ///
    /// In order to keep variation fairly consistent on neural networks
    /// of various sizes, the intensity is scaled down by the number of
    /// connections in the network before being applied.
    ///
    /// see also:
    ///  * [`create_variant_stdrng`]
    ///  * [`VAI::create_layer_variant`]
    pub fn create_variant(&mut self, intensity: f32) -> Self {
        let mut result = self.clone();
        let fields = I * C + C * C * EXTRA_LAYERS + C * O;
        let s_intensity = intensity / (1.0 + fields as f32);
        result.input_connections =
            create_variant_stdrng(&mut self.rng, &result.input_connections, intensity);
        for mat in &mut result.hidden_connections {
            *mat = create_variant_stdrng(&mut self.rng, mat, s_intensity);
        }
        result.output_connections =
            create_variant_stdrng(&mut self.rng, &result.output_connections, s_intensity);
        return result;
    }

    /// Creates a random variant of this VAI that only changes one layer
    /// * intensity - Scaler for the added randomness
    ///
    /// Intensity affects the random distribution to favor low magnitude
    /// values, but the result can still be changed by an arbitrary amount.
    /// Randomness is applied to each weight of each connection on a randomly
    /// chosen layer.
    ///
    /// In order to keep variation fairly consistent on neural networks
    /// of various sizes, the intensity is scaled down by the number of
    /// connections in the chosen layer before being applied.
    ///
    /// see also:
    ///  * [`create_variant_stdrng`]
    pub fn create_layer_variant(&mut self, intensity: f32) -> Self {
        let mut result = self.clone();
        let hidden_connections = &mut result.hidden_connections;
        let layer = rand_index(hidden_connections.len() + 2);
        let intensity = intensity / (C * C + 1) as f32;
        match layer.cmp(&hidden_connections.len()) {
            Ordering::Less => {
                let original = &hidden_connections[layer];
                hidden_connections[layer] =
                    create_variant_stdrng(&mut self.rng, original, intensity);
            }
            Ordering::Equal => {
                let original = &result.input_connections;
                result.input_connections =
                    create_variant_stdrng(&mut self.rng, original, intensity);
            }
            Ordering::Greater => {
                let original = &result.output_connections;
                result.output_connections =
                    create_variant_stdrng(&mut self.rng, original, intensity);
            }
        }
        return result;
    }

    /// Runs an input matrix through the neural network to get an output
    /// * inputs - The inputs. One of them should be a constant for a bias.
    ///
    /// see also:
    ///  * [`VAI::process_slice`]
    ///  * [`VAI::process_transparent`]
    pub fn process(&self, inputs: &na::SMatrix<f32, I, 1>) -> na::SMatrix<f32, O, 1> {
        let mut intermediate = self.input_connections * inputs;
        // Apply relu
        intermediate.apply(|x| *x = x.max(0.));
        for mat in &self.hidden_connections {
            intermediate = mat * intermediate;
            // Apply relu
            intermediate.apply(|x| *x = x.max(0.));
        }
        return self.output_connections * intermediate;
    }

    /// Runs an input slice through the neural network to get an output
    /// * inputs - The inputs. One of them should be a constant for a bias.
    ///
    /// see also:
    ///  * [`VAI::process`]
    pub fn process_slice(&self, inputs: &[f32]) -> Vec<f32> {
        let matrix_inputs = na::SMatrix::<f32, I, 1>::from_column_slice(inputs);
        let output = self.process(&matrix_inputs);
        output.iter().map(|x| x.to_owned()).collect()
    }

    /// Runs an input matrix through the neural network to get an output
    /// returning the value of all the nodes: input, hidden, and output.
    ///
    /// Note: the value of hidden nodes is supplie *before*
    /// relu to preserve information
    /// * inputs - The inputs. One of them should be a constant for a bias.
    ///
    /// see also:
    ///  * [`VAI::process_slice_transparent`]
    pub fn process_transparent(&self, inputs: &na::SMatrix<f32, I, 1>) -> Vec<Vec<f32>> {
        let mut output: Vec<Vec<f32>> = vec![inputs.iter().map(|x| x.to_owned()).collect()];
        let mut intermediate = self.input_connections * inputs;
        output.push(intermediate.iter().map(|x| x.to_owned()).collect());
        // Apply relu
        intermediate.apply(|x| *x = x.max(0.));
        for mat in &self.hidden_connections {
            intermediate = mat * intermediate;
            output.push(intermediate.iter().map(|x| x.to_owned()).collect());
            // Apply relu
            intermediate.apply(|x| *x = x.max(0.));
        }
        let out = self.output_connections * intermediate;
        output.push(out.iter().map(|x| x.to_owned()).collect());
        return output;
    }

    /// Runs an input slice through the neural network to get an output
    /// returning the value of all the nodes: input, hidden, and output.
    /// * inputs - The inputs. One of them should be a constant for a bias.
    ///
    /// see also:
    ///  * [`VAI::process_transparent`]
    pub fn process_slice_transparent(&self, inputs: &[f32]) -> Vec<Vec<f32>> {
        let matrix_inputs = na::SMatrix::<f32, I, 1>::from_column_slice(inputs);
        return self.process_transparent(&matrix_inputs);
    }

    /// Writes a vai to a file, writing its input, hidden, and output
    /// connections in order, as written by [`write_matrix`]
    /// * lines - A line iterator from which to read the vai
    /// (generally provided by BufReader::new(file).lines())
    pub fn write(&self, file: &mut File) -> std::io::Result<()> {
        write_matrix(&self.input_connections, file)?;
        for matrix in &self.hidden_connections {
            write_matrix(matrix, file)?;
        }
        write_matrix(&self.output_connections, file)?;
        return Ok(());
    }

    /// Reads a matrix from lines of a file containing its input, hidden, and
    /// output connections in order, as read by [`read_matrix`]
    /// * lines - A line iterator from which to read the vai
    /// (generally provided by BufReader::new(file).lines())
    pub fn read(lines: &mut Lines<std::io::BufReader<File>>) -> std::io::Result<Self> {
        let mut result = Self::new();
        result.input_connections = read_matrix(lines)?;
        for matrix in &mut result.hidden_connections {
            *matrix = read_matrix(lines)?;
        }
        result.output_connections = read_matrix(lines)?;
        return Ok(result);
    }
}
