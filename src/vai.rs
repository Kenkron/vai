#![allow(clippy::needless_return)]

use std::cmp::Ordering;
use std::io::{self, Lines, Write};
use std::ops::{Add, Mul, Sub};
use std::{fmt::Display, fs::File};

extern crate nalgebra as na;
use na::SMatrix;
use nalgebra::SVector;
extern crate rand;
use crate::{infinite_map, rand_index};
use rand::rngs::StdRng;
use rand::Rng;

/// Used for classification wherein the largest value is the chosen
/// category. Equivalent to the log(probability) for each value...
/// Sort of.
fn softmax_slice(values: &[f32]) -> Vec<f32> {
    let max_value = values.iter().fold(-f32::NEG_INFINITY, |acc, x| acc.max(*x));
    let softmax_values: Vec<f32> = values.iter().map(|val| (val - max_value).exp()).collect();
    let softmax_total = softmax_values.iter().fold(0.0, |acc, x| acc + x);
    softmax_values.iter().map(|x| x / softmax_total).collect()
}

/// Used for classification wherein the largest value is the chosen
/// category. Equivalent to the log(probability) for each value...
/// Sort of.
fn softmax<const O: usize>(values: SVector<f32, O>) -> SVector<f32, O> {
    let max_value = values.max();
    values.apply(|val| (val - max_value).exp()).collect();
    let softmax_total = values.sum();
    values.apply(|x| x / softmax_total);
    values
}

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

pub fn backpropogate<const I: usize, const O: usize>(
    activations: &na::SVector<f32, I>,
    weights: &na::SMatrix<f32, O, I>,
    costs: &na::SVector<f32, O>,
    relu: bool,
) -> na::SMatrix<f32, O, I> {
    let mut weight_cost = SMatrix::<f32, O, I>::zeros();
    for i in 0..I {
        if activations[i] == 0.0 {
            // if this had no activation, it had no effect
            continue;
        }
        for o in 0..O {
            let influence = activations[i] * weights.row(o).get(i).unwrap();
            if relu && influence < 0.0 {
                // If a relu was applied, negative influence would be zero influence
                continue;
            }
            weight_cost[(o, i)] = influence * costs[o];
        }
    }
    // if you have multiple inputs, they should share the cost
    return weight_cost / I as f32;
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
/// * I: Number of inputs. The first element will always be set to 1 as a bias.
/// * O: Number of outputs
/// * C: Complexity of (number of nodes in) hidden layers. The first one will always be set to 1 as a bias.
/// * EXTRA_LAYERS: There is always at least one hidden layer. This number adds more.
#[derive(Clone)]
pub struct VAI<const I: usize, const O: usize, const C: usize, const EXTRA_LAYERS: usize> {
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
            input_connections: na::SMatrix::<f32, C, I>::zeros(),
            hidden_connections: [na::SMatrix::<f32, C, C>::zeros(); EXTRA_LAYERS],
            output_connections: na::SMatrix::<f32, O, C>::zeros(),
        }
    }

    // Direction towards the expected output from the expected input
    pub fn backpropogate(
        &self,
        inputs: &na::SVector<f32, I>,
        expected_outputs: &na::SVector<f32, O>,
        softmax: bool,
    ) -> Self {
        let mut difference = Self {
            input_connections: na::SMatrix::<f32, C, I>::zeros(),
            hidden_connections: [na::SMatrix::<f32, C, C>::zeros(); EXTRA_LAYERS],
            output_connections: na::SMatrix::<f32, O, C>::zeros(),
        };

        let mut hidden_layers: Vec<SVector<f32, C>> = vec![];
        let mut intermediate_layer = self.input_connections * inputs;
        intermediate_layer.apply(|x| *x = x.max(0.));
        hidden_layers.push(intermediate_layer);
        for connection in self.hidden_connections {
            intermediate_layer = connection * intermediate_layer;
            intermediate_layer.apply(|x| *x = x.max(0.));
            hidden_layers.push(intermediate_layer);
        }
        let outputs = self.output_connections * intermediate_layer;
        let mut cost = expected_outputs - outputs;
        if softmax {
            cost = softmax(cost);
        }

        let weight_cost = backpropogate(
            &hidden_layers[hidden_layers.len() - 1],
            &self.output_connections,
            &cost,
            false,
        );
        let mut layer_cost = weight_cost.row_sum_tr();
        difference.output_connections = weight_cost / (EXTRA_LAYERS + 2) as f32;
        for i in 0..self.hidden_connections.len() {
            let weight_cost = backpropogate(
                &hidden_layers[hidden_layers.len() - i - 1],
                &self.hidden_connections[self.hidden_connections.len() - i - 1],
                &layer_cost,
                true,
            );
            layer_cost = weight_cost.row_sum_tr();
            difference.hidden_connections[self.hidden_connections.len() - i - 1] = weight_cost;
        }

        let weight_cost = backpropogate(&inputs, &self.input_connections, &layer_cost, true);
        difference.input_connections = weight_cost / (EXTRA_LAYERS + 2) as f32;

        return difference;
    }

    pub fn train<A>(&self, data: A, scale: f32) -> Self
    where
        A: IntoIterator<Item = (SVector<f32, I>, SVector<f32, O>)>,
    {
        let mut error = Self::new();
        let mut count = 0_usize;
        for (input, expected_output) in data {
            error = &error + &self.backpropogate(&input, &expected_output);
            count += 1;
        }
        if count == 0 {
            return self.clone();
        }
        let scaled_error = &error * (scale / count as f32);
        println!("calculated error: {}", scaled_error);
        self - &scaled_error
    }

    pub fn train_categorizer<A>(&self, data: A, scale: f32) -> Self
    where
        A: IntoIterator<Item = (SVector<f32, I>, SVector<f32, O>)>,
    {
        let mut error = Self::new();
        let mut count = 0_usize;
        for (input, expected_output) in data {
            error = &error + &self.backpropogate(&input, &expected_output);
            count += 1;
        }
        if count == 0 {
            return self.clone();
        }
        let scaled_error = &error * (scale / count as f32);
        println!("calculated error: {}", scaled_error);
        self - &scaled_error
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
    pub fn create_variant(&mut self, intensity: f32, rng: &mut StdRng) -> Self {
        let mut result = self.clone();
        let fields = I * C + C * C * EXTRA_LAYERS + C * O;
        let s_intensity = intensity / (1.0 + fields as f32);
        result.input_connections = create_variant_stdrng(rng, &result.input_connections, intensity);
        for mat in &mut result.hidden_connections {
            *mat = create_variant_stdrng(rng, mat, s_intensity);
        }
        result.output_connections =
            create_variant_stdrng(rng, &result.output_connections, s_intensity);
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
    pub fn create_layer_variant(&mut self, intensity: f32, rng: &mut StdRng) -> Self {
        let mut result = self.clone();
        let hidden_connections = &mut result.hidden_connections;
        let layer = rand_index(hidden_connections.len() + 2);
        let intensity = intensity / (C * C + 1) as f32;
        match layer.cmp(&hidden_connections.len()) {
            Ordering::Less => {
                let original = &hidden_connections[layer];
                hidden_connections[layer] = create_variant_stdrng(rng, original, intensity);
            }
            Ordering::Equal => {
                let original = &result.input_connections;
                result.input_connections = create_variant_stdrng(rng, original, intensity);
            }
            Ordering::Greater => {
                let original = &result.output_connections;
                result.output_connections = create_variant_stdrng(rng, original, intensity);
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
        // Apply bias
        if I > 0 {
            inputs[0] = 1;
        }
        let mut intermediate = self.input_connections * inputs;
        // Apply relu
        intermediate.apply(|x| *x = x.max(0.));
        for mat in &self.hidden_connections {
            // Apply bias
            if intermediate.len() > 0 {
                intermediate[0] = 1;
            }
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
    pub fn process_transparent(
        &self,
        inputs: &na::SVector<f32, I>,
    ) -> (Vec<SVector<f32, C>>, SVector<f32, O>) {
        let mut hidden_nodes: Vec<SVector<f32, C>> = vec![];

        // clear input 0 for bias
        if I > 0 {
            inputs[0] = 0.0;
        }
        // Compute nodes
        let mut intermediate = self.input_connections * inputs;
        // Apply relu
        intermediate.apply(|x| *x = x.max(0.));
        // Add the bias
        intermediate += self.input_connections.column(0);
        // Clear node 0 for bias
        intermediate[0] = 0.0;
        // Add to hidden nodes
        hidden_nodes.push(intermediate.clone());

        for hidden_connection in &self.hidden_connections {
            // Compute nodes
            let mut intermediate = hidden_connection * intermediate;
            // Apply relu
            intermediate.apply(|x| *x = x.max(0.));
            // Add the bias
            intermediate += hidden_connection.column(0);
            // Clear node 0 for bias
            intermediate[0] = 0.0;
            // Add to hidden nodes
            hidden_nodes.push(intermediate.clone());
        }
        let output = self.output_connections * intermediate;
        return (hidden_nodes, output);
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

impl<'a, const I: usize, const O: usize, const C: usize, const EXTRA_LAYERS: usize> Add
    for &'a VAI<I, O, C, EXTRA_LAYERS>
{
    type Output = VAI<I, O, C, EXTRA_LAYERS>;

    fn add(self, other: Self) -> Self::Output {
        let mut hidden_connections = [SMatrix::<f32, C, C>::zeros(); EXTRA_LAYERS];
        for i in 0..EXTRA_LAYERS {
            hidden_connections[i] = self.hidden_connections[i] + other.hidden_connections[i];
        }
        VAI {
            input_connections: self.input_connections + other.input_connections,
            output_connections: self.output_connections + other.output_connections,
            hidden_connections,
        }
    }
}

impl<'a, const I: usize, const O: usize, const C: usize, const EXTRA_LAYERS: usize> Sub
    for &'a VAI<I, O, C, EXTRA_LAYERS>
{
    type Output = VAI<I, O, C, EXTRA_LAYERS>;

    fn sub(self, other: Self) -> Self::Output {
        let mut hidden_connections = [SMatrix::<f32, C, C>::zeros(); EXTRA_LAYERS];
        for i in 0..EXTRA_LAYERS {
            hidden_connections[i] = self.hidden_connections[i] - other.hidden_connections[i];
        }
        VAI {
            input_connections: self.input_connections - other.input_connections,
            output_connections: self.output_connections - other.output_connections,
            hidden_connections,
        }
    }
}

impl<'a, const I: usize, const O: usize, const C: usize, const EXTRA_LAYERS: usize> Mul<f32>
    for &'a VAI<I, O, C, EXTRA_LAYERS>
{
    type Output = VAI<I, O, C, EXTRA_LAYERS>;

    fn mul(self, other: f32) -> Self::Output {
        let mut hidden_connections = [SMatrix::<f32, C, C>::zeros(); EXTRA_LAYERS];
        for i in 0..EXTRA_LAYERS {
            hidden_connections[i] = self.hidden_connections[i] * other;
        }
        VAI {
            input_connections: self.input_connections * other,
            output_connections: self.output_connections * other,
            hidden_connections,
        }
    }
}
