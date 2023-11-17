#![allow(clippy::needless_return)]

use crate::{infinite_map, rand_index};

use std::io::{Lines, Write};
use std::{fmt::Display, fs::File};

extern crate nalgebra as na;
use na::{DMatrix, RowDVector};
extern crate rand;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Creates a random variation of a matrix
/// * original - The matrix that will be varied
/// * intensity - The severity to which the matrix will be randomized.
///
/// Intensity affects the random distribution to favor smaller values, but the
/// resulting matrix can still be changed by an arbitrary amount.
pub fn create_variant(
    original: &na::DMatrix<f32>,
    intensity: f32
) -> na::DMatrix<f32> {
    // Adding two variances increases the odds of small changes,
    // but also makes larger infrequent changes possible
    let mut variance = DMatrix::<f32>::new_random(original.shape().0, original.shape().1);
    variance.apply(|x| *x = infinite_map(*x));
    return original + variance.scale(intensity);
}

/// Creates a random variation of a matrix using a provided StdRng
/// * original - The matrix that will be varied
/// * intensity - The severity to which the matrix will be randomized.
///
/// Intensity affects the random distribution to favor smaller values, but the
/// resulting matrix can still be changed by an arbitrary amount.
pub fn create_variant_stdrng(
    rng: &mut StdRng,
    original: &na::DMatrix<f32>,
    intensity: f32,
) -> na::DMatrix<f32> {
    let mut result = original.clone_owned();
    result.apply(|x| *x += intensity * infinite_map(rng.gen::<f32>()));
    return result;
}

/// Writes a matrix to a file with space-delimited columns,
/// newline delimited rows, and a trailing newline.
/// * matrix - The matrix to write
/// * file - The file to write to
pub fn write_matrix(
    matrix: &DMatrix<f32>,
    file: &mut File,
) -> std::io::Result<()> {
    writeln!(file, "{}", matrix.shape().0)?;
    for row in matrix.row_iter() {
        for val in &row {
            write!(file, "{} ", val)?;
        }
        writeln!(file)?;
    }
    return Ok(());
}

/// Reads a matrix from lines of a file with space-delimited columns,
/// and newline delimited rows. Empty (whitespace) lines are ignored.
/// * lines - A line iterator from which to read the matrix
/// (generally provided by BufReader::new(file).lines())
pub fn read_matrix(
    lines: &mut Lines<std::io::BufReader<File>>,
) -> std::io::Result<DMatrix<f32>> {
    use std::io::{Error, Result, ErrorKind};
    let row_line_error = || Error::new(ErrorKind::Other, "Bad Row Count");
    let parse_float_error = || Error::new(ErrorKind::Other, "Error parsing float");
    let row_line_error_result = || Result::Err(row_line_error());

    let first_line = lines.next().unwrap_or(row_line_error_result())?;
    let num_rows: usize = first_line.parse().map_err(|_| row_line_error())?;
    let mut row_count = 0;
    let mut rows = Vec::<RowDVector<f32>>::new();
    for line in lines {
        let line = line?;
        let vals: Vec<f32> = line
            .split_whitespace()
            .map(|val| {
                val.parse::<f32>().map_err(|_| parse_float_error())
            })
            .collect::<Result<Vec<f32>>>()?;
        let row = RowDVector::from_row_slice(&vals);

        //skip empty rows
        if !row.is_empty() {
            rows.push(row);
            row_count += 1;
        }

        if row_count == num_rows {
            break;
        }
    }
    return Ok(DMatrix::<f32>::from_rows(&rows));
}

/// Very Artificial Intelligence Dynamic
///
/// Much like VAI, but with dynamically allocated layers
#[derive(Clone, PartialEq)]
pub struct VAID {
    pub rng: StdRng,
    pub connections: Vec<DMatrix<f32>>
}

impl Display for VAID
{
    /// Concatenates the string representations of the input,
    /// hidden layer, and output matricies.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for mat in &self.connections {
            write!(f, "{}", mat)?;
        }
        write!(f, "")
    }
}

impl VAID {
    /// Creates a VAID with zeros for all connection weights,
    /// using a random seed for random number generatoin.
    ///
    /// * layers - The number of neurons in each layer, starting with
    /// the number of input nodes, and ending with the number of output nodes.
    /// If this does not have at least two layers, the inputs will
    /// be mapped directly to outputs.
    pub fn new(layers: &[usize]) -> Self {
        Self::new_deterministic(rand::random(), layers)
    }

    /// Creates a VAID with zeros for all connection weights,
    /// using a specific seed for random number generatoin.
    ///
    /// * layers - The number of neurons in each layer, starting with
    /// the number of input nodes, and ending with the number of output nodes.
    /// If this does not have at least two layers, the inputs will
    /// be mapped directly to outputs.
    pub fn new_deterministic(seed: u64, layers: &[usize]) -> Self {
        let mut connections = Vec::<DMatrix<f32>>::new();
        for i in 0..layers.len()-1 {
            connections.push(DMatrix::<f32>::zeros(layers[i + 1], layers[i]));
        }
        return Self {rng: StdRng::seed_from_u64(seed), connections}
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
    ///  * [`VAID::create_layer_variant`]
    pub fn create_variant(&mut self, intensity: f32) -> Self {
        let mut result = self.clone();
        let fields: usize = self.connections.iter().map(|x| x.len()).sum();
        let s_intensity = intensity / (1.0 + fields as f32);
        for mat in &mut result.connections {
            *mat = create_variant_stdrng(&mut self.rng, mat, s_intensity);
        }
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
        let layer = rand_index(result.connections.len());
        let original = &self.connections[layer];
        let intensity = intensity / (original.len() + 1) as f32;
        result.connections[layer] =
            create_variant_stdrng(&mut self.rng, original, intensity);
        return result;
    }

    /// Runs an input matrix through the neural network to get an output
    /// * inputs - The inputs. One of them should be a constant for a bias.
    ///
    /// see also:
    ///  * [`VAIN::process_slice`]
    ///  * [`VAIN::process_transparent`]
    pub fn process(&self, inputs: &na::DMatrix<f32>) -> na::DMatrix<f32> {
        let mut intermediate = inputs.clone();
        if let Some((last, first)) = self.connections.split_last() {
            for mat in first {
                intermediate = mat * intermediate;
                // Apply relu
                intermediate.apply(|x| *x = x.max(0.));
            }
            return last * intermediate;
        }
        return inputs.clone();
    }

    /// Runs an input slice through the neural network to get an output
    /// * inputs - The inputs. One of them should be a constant for a bias.
    ///
    /// see also:
    ///  * [`VAIN::process`]
    pub fn process_slice(&self, inputs: &[f32]) -> Vec<f32> {
        let matrix_inputs = na::DMatrix::<f32>::from_columns(&[
            na::DVector::<f32>::from_column_slice(inputs)]);
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
    ///  * [`VAIN::process_slice_transparent`]
    pub fn process_transparent(&self, inputs: &na::DMatrix<f32>) -> Vec<Vec<f32>> {
        let mut output: Vec<Vec<f32>> = vec![inputs.iter().map(|x| x.to_owned()).collect()];
        let mut intermediate = inputs.clone();
        if let Some((last, first)) = self.connections.split_last() {
            for mat in first {
                intermediate = mat * intermediate;
                output.push(intermediate.iter().map(|x| x.to_owned()).collect());
                // Apply relu
                intermediate.apply(|x| *x = x.max(0.));
            }
            let out = last * intermediate;
            output.push(out.iter().map(|x| x.to_owned()).collect());
        }
        return output;
    }

    /// Runs an input slice through the neural network to get an output
    /// returning the value of all the nodes: input, hidden, and output.
    /// * inputs - The inputs. One of them should be a constant for a bias.
    ///
    /// see also:
    ///  * [`VAIN::process_transparent`]
    pub fn process_slice_transparent(&self, inputs: &[f32]) -> Vec<Vec<f32>> {
        let matrix_inputs = na::DMatrix::<f32>::from_columns(&[
            na::DVector::<f32>::from_column_slice(inputs)]);
        return self.process_transparent(&matrix_inputs);
    }

    /// Writes a vai to a file, writing its input, hidden, and output
    /// connections in order, as written by [`write_matrix`]
    /// * lines - A line iterator from which to read the vai
    /// (generally provided by BufReader::new(file).lines())
    pub fn write(&self, file: &mut File) -> std::io::Result<()> {
        writeln!(file, "{}", self.connections.len())?;
        for matrix in &self.connections {
            write_matrix(matrix, file)?;
        }
        return Ok(());
    }

    /// Reads a matrix from lines of a file containing its input, hidden, and
    /// output connections in order, as read by [`read_matrix`]
    /// * lines - A line iterator from which to read the vai
    /// (generally provided by BufReader::new(file).lines())
    pub fn read(lines: &mut Lines<std::io::BufReader<File>>) -> std::io::Result<Self> {
        use std::io::{Error, Result, ErrorKind};
        let row_line_error = || Error::new(ErrorKind::Other, "Bad Row Count");
        let row_line_error_result = || Result::Err(row_line_error());
        let mut connections = Vec::<DMatrix<f32>>::new();

        let first_line = lines.next().unwrap_or(row_line_error_result())?;
        let num_matrices: usize = first_line.parse().map_err(|_| row_line_error())?;
        for _ in 0..num_matrices {
            connections.push(read_matrix(lines)?);
        }
        return Ok(Self {rng: StdRng::seed_from_u64(rand::random()), connections});
    }
}