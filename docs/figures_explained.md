# Figure Explanations — Filter Learning Framework Paper

This document explains the key figures from the paper
“Learning Laws for Deep CNNs with Guaranteed Convergence”
in structured text form for programmatic and engineering usage.

------------------------------------------------------------
FIGURE 1 — Mathematical Representation of CNN Layers
------------------------------------------------------------

## What the figure shows
The figure illustrates how an input image is processed through:

1. Convolution
2. Activation (Leaky/Adjustable ReLU)
3. Pooling
4. Flattening
5. Fully Connected Classification

It also shows the mathematical abstraction of these operations
into matrix multiplications.

## Mathematical meaning
Each convolution layer is represented as:

X_fj * Vj

Where

X_fj
    Filter-dependent input matrix
    Constructed by sliding-window extraction
    (equivalent to im2col representation)

Vj
    Filter matrix formed by stacking flattened filters

Activation:

Rj ( X_fj Vj )

Pooling:

Gamma_j * Rj( X_fj Vj )

Flatten:

sigma_j = F( Gamma_j Rj ( X_fj Vj ) )

Final classification:

y = phi( W sigma_j )

## Why this figure matters
This representation is the foundation that allows:

- Derivation of update laws
- Lyapunov stability analysis
- Matrix-form convergence proof

Without converting CNN operations into matrix algebra,
the convergence analysis would be impossible.

## Code Mapping Insight
Typical PyTorch equivalents:

Conv2D → implicit X_fj construction
ReLU → Rj
AvgPool/MaxPool → Gamma_j
Flatten → F
Linear → W

------------------------------------------------------------
FIGURE 2 — Pooling Matrix Interpretation
------------------------------------------------------------

## What the figure shows
A small example demonstrating:

- How convolution outputs form vectors
- How pooling selects or averages elements
- How pooling is expressed as matrix multiplication

## Mathematical meaning

Average pooling:

Gamma_avg b =
(1/p^2) * sum(b_i)

Gamma matrix rows contain uniform weights.

Max pooling:

Gamma_max b =
max(b_i)

Gamma matrix rows contain one-hot selectors.

## Why this figure matters
Key insight of the paper:

Pooling is treated as a linear operator

This allows pooling to be embedded inside the
Lyapunov convergence derivation.

Standard deep learning treats pooling as procedural,
but this framework treats it algebraically.

## Engineering implication
During early training:

Average pooling preferred
(stable linear behavior)

Later training:

Switch to Max pooling
(increased nonlinearity capacity)

------------------------------------------------------------
FIGURE 3 — Filter Learning Framework Architecture
------------------------------------------------------------

## What the figure shows
The most important diagram in the paper.

It illustrates:

A single CNN being trained by
multiple auxiliary filter learning systems.

Each system contains:

Input → Conv_j → Activation → Pool → Flatten → FC_j

And each system updates:

- Convolution weights of layer j
- Auxiliary classifier weights W_j

## Core concept
Instead of training the CNN as one monolithic system:

The network is decomposed into n subsystems

Each subsystem:

Observes intermediate features
Produces prediction y_j
Generates training error
Updates layer-specific weights

All subsystems update simultaneously.

## Why this is revolutionary
This architecture enables:

1. End-to-end concurrent updates
2. Stability-controlled learning
3. Convergence guarantees
4. Parallelizable training

It replaces traditional backprop:

Global gradient flow

with

Structured local learning laws

## Snapshot Mechanism Illustrated
The diagram implies:

Filters updated using
previous batch classifier weights

Classifiers updated using
previous batch filter weights

This decoupling is essential for stability proof.

## Code Mapping Insight

Each system corresponds to:

One conv layer
One auxiliary linear layer

Implementation pattern:

systems[j] = {
    conv_layer[j],
    pooling[j],
    activation[j],
    auxiliary_fc[j]
}

------------------------------------------------------------
END OF DOCUMENT
------------------------------------------------------------