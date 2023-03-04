Very Artificial Intelligence
============================

A primitive implementation of a neural network.

> VAI is a simple implementation of a Deep Learning Neural Network
> that supports an arbitrary number of input nodes, output nodes, hidden layers
> and hidden neurons. It provides randomization functions for neuroevolution
> with a distribution that favors modest changes, but will occasionally produce
> very large changes, and scales based on the number of connections in the
> network.

 * Written in rust
 * Statically defined, arbitrary number of:
   * Input nodes (make sure one of them is a constant for bias)
   * Output nodes
   * Hidden layers (minimum of 1)
   * Hidden layer size (all hidden layers are the same size)
 * Uses ReLu for non-linear behaviour
 * Writing / Reading convinience functions

Dotfield example
----------------

The dotfield example uses vai to train a neural network to outline two circles.
It is visalized with macroquad, so requires it for execution.
The program will usually require several hundred thousand generations to
outline the circles spearately, so the release flag is recommended.

It can be run with:

    cargo run --release --features="macroquad" --example dotfield

The example uses 3 inputs (x, y, and a bias), two hidden layers of 16 nodes
(ie. one hidden 16x16 layer of connections), and a single output indicating
whether the ai thinks the point is inside one of the circles.
Test samples will be shown as color coded dots.

* Red: Outside the circles. The AI got it wrong.
* Yellow: Inside the circles. The AI got it wrong.
* Purple: Outside the circles. The AI got it right.
* Green: Inside the circles. The AI got it right.

There are a number of controls:

* Escape: exit
* Space: start/pause the training
* Enter: when paused, run a single training step
* Q: turn off/on the visualization
* B: toggle between showing the best ai, and the most recently tested ai
* P: print out the current ai
* S: save the current ai to save.vai
* O: load the current ai from save.vai
