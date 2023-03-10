Dotfield
--------

The dotfield example trains a neural network to outline two circles.
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

Character Recognition
---------------------

The character recognition example trains a neural network to recognize the
numbers 0-9. The characters are 12-16 pixels tall, and are drawn on a 16x16
texture. The scale and position are randomized for each test.

It can be run with:

    cargo run --release --features="macroquad" --example character_recognition

The network provides an output for each character. For scoring, these outputs
are ordered, and the further the correct character is from the top of the list,
the worse the score.

The network generates 100 test networks, runs 500 randomly generated characters
therough them, keeps the best 5, and bases the next generation of test networks
on those 5 best.

There are a number of controls:

 * Escape: exit
 * Space: start/pause the training
 * Enter: when paused, run a single training step
 * S: save the current ai to save.vai
 * O: load the current ai from save.vai
 * P: print out the current best-scoring ai
