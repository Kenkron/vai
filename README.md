Very Artificial Intelligence
============================

![Tests](https://github.com/Kenkron/vai/actions/workflows/Tests.yml/badge.svg)

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

examples
--------

 * **dotfield:** Trains a Vai to outline two circles
 * **character_recognition:** Trains a Vai to recognize characters

Further information in [examples/README.md]

disclaimer
----------

I made this library as a bit of a joke. A bunch of youtube videos were appearing
in my feed that featured "evolutionary" creatures powered by neural networks.
The premise was that each new generation would introduce a random variation to
the best of the previous generation, until the creature became really good at
its task.

These videos were entertaining and, as far as I could figure, avoided most
of the "hard part" of neural networks. Backpropogation? Who needs it?
Just roll the dice with a couple of matrices and a ReLu function!
