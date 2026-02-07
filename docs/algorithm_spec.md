# Algorithm Spec — Learning Laws for Deep CNNs with Guaranteed Convergence
Paper: "Learning Laws for Deep Convolutional Neural Networks with Guaranteed Convergence"

> Purpose: Convert the paper into an implementable, testable algorithm specification that an IDE agent (Cursor) can consume.
> Scope: Defines the training framework (filter learning framework), variables, update laws, control flow (sample/batch), and the operational “learning laws” (gain scheduling, adjustable ReLU schedule, pooling switch, and condition checks).

---

## 0. One-line Summary
Train a deep CNN by decomposing it into multiple **filter learning systems** (one per conv layer), each with an auxiliary classifier. In each mini-batch, **freeze classifier weights from the previous batch to update filters** and **freeze filter weights from the previous batch to update classifiers**, while enforcing **gain-matrix conditions** that guarantee bounded convergence of training error.

---

## 1. Problem Definition
### 1.1 Task
Supervised classification. Given input samples x(k) and labels y(k), learn CNN parameters so that predictions y_hat(k) approximate y(k).

### 1.2 Model Family
A deep convolutional neural network consisting of:
- Multiple convolutional layers (with weight sharing)
- Nonlinearities (paper uses Adjustable ReLU)
- Pooling operators (average pooling early; optional switch to max pooling later)
- A final classification objective

### 1.3 Key Difficulty Addressed
Standard end-to-end BP convergence proofs do not directly transfer to CNNs due to parameter sharing and coupled dynamics. Paper re-frames training as a stable learning dynamical system with explicitly designed update laws.

---

## 2. Core Framework: Filter Learning Systems
### 2.1 Definition
For each convolutional layer j, define one **filter learning system** S_j that includes:
- Layer-j convolution filters (trainable)
- Layer-j activation and pooling (deterministic operators)
- A layer-j auxiliary classifier (FC) that maps layer-j features to output y_hat_j
- Its own error signals and learning gains

Thus for a CNN with J convolutional layers:
- There are J filter learning systems {S_1, S_2, ..., S_J}
- They can be trained in parallel conceptually (paper mentions parallelism is possible)

### 2.2 Why This Helps
This creates a set of locally analyzable learning dynamics:
- Each system S_j has an explicit “filter learning error” and a “classifier error”
- Update laws are designed so a Lyapunov function decreases under stated conditions

---

## 3. Notation and Data Flow (Implementation-Ready)
### 3.1 Indices
- b: mini-batch index
- k: sample index within a batch (k = 1..B)
- j: convolutional layer index (j = 1..J)
- i: filter index in layer j

### 3.2 Inputs/Outputs
- Input sample: x(k)
- Target label: y(k) (one-hot or real vector)
- System-j prediction: y_hat_j(k)
- Optional “global” prediction: depends on how you aggregate (paper evaluates each system; implementation may use final layer system or ensemble)

### 3.3 Learnable Parameters
Per layer j:
- Filters: v_{j,i}  (for each filter i in layer j; collectively V_j)
- Auxiliary classifier: W_j (FC weights mapping features of layer j to output)

### 3.4 Frozen (“bar”) Parameters Across Batches
To enable the convergence conditions, define batch-frozen snapshots:
- \bar{W}_j^{[b-1]} = classifier weights from previous batch
- \bar{V}_j^{[b-1]} = filter weights from previous batch

Within batch b:
- When updating filters V_j, use \bar{W}_j^{[b-1]} (frozen classifier)
- When updating classifiers W_j, use \bar{V}_j^{[b-1]} (frozen filters)

End of batch b:
- Refresh snapshots: \bar{W}_j^{[b]} := W_j,  \bar{V}_j^{[b]} := V_j

---

## 4. Operators in Each Filter Learning System
### 4.1 Convolution with Weight Sharing
- Standard convolution operation with shared filter parameters.
- In code: conv_j forward uses v_{j,i} for all spatial locations (normal CNN conv).

### 4.2 Pooling Operator Γ
- Paper models pooling as a linear operator Γ in the learning law.
- Implementation uses standard pooling layers:
  - Default early: AveragePooling
  - Optional later: switch to MaxPooling (triggered by training rule below)

### 4.3 Nonlinearity: Adjustable ReLU
Activation: Adjustable ReLU with parameter a (batch-wise constant)
- a is updated across training according to an error-based schedule.
- Initialization: a = 1
- Decrease a as training error drops, with lower bound 0.01
- Once a reaches 0.01 (or near), switch pooling from average to max.

Practical implementation:
- Use: y = ReLU(x) but with adjustable slope/behavior per the paper’s definition.
- If your code uses a simplified version: keep it consistent with the paper schedule (a changes between batches, not within).

---

## 5. Error Signals
Each system S_j maintains two errors:

### 5.1 Classifier error (for updating W_j)
- e^c_j(k) = y(k) - y_hat_j(k)

### 5.2 Filter learning error (for updating V_j)
- e^f_j(k): derived error that drives the filter update.
- In the paper, e^f is tied to classifier error and the frozen classifier weights (through the designed learning law).
- Implementation requirement:
  - e^f_j must be computed per sample using forward quantities of S_j and \bar{W}_j^{[b-1]}.

---

## 6. Update Laws (Algorithmic “Learning Laws”)
### 6.1 Filter Update (Paper Eq.15 — abstracted)
For each layer j and filter i, per sample k:
- v_{j,i}(k+1) = v_{j,i}(k) + α^f_j * D_{j,i}(k) * L^f_j(k) * e^f_j(k)

Where:
- α^f_j: scalar learning rate for filters in layer j
- L^f_j(k): filter learning gain matrix (must satisfy stability condition)
- D_{j,i}(k): regressor-like term built from:
  - input/features to layer j
  - activation/pooling operators
  - frozen classifier weights \bar{W}_j^{[b-1]}

Implementation requirement:
- You must implement D_{j,i}(k) exactly as in the code/paper mapping.
- If the original repo provides D computation, keep it and document it.

### 6.2 Classifier Update (Paper Eq.19 — abstracted)
Per layer j, per sample k:
- W_j(k+1) = W_j(k) + α^c_j * L^c_j(k) * e^c_j(k) * (σ_j^{[b-1]}(k))^T

Where:
- α^c_j: scalar learning rate for classifier j
- L^c_j(k): classifier gain matrix (must satisfy stability condition)
- σ_j^{[b-1]}(k): feature vector for layer j computed using frozen filters \bar{V}_j^{[b-1]}

Implementation requirement:
- Features used for classifier update must be consistent with “frozen filter within batch” design.

---

## 7. Gain Matrices and Condition Checks (Guarantee Mechanism)
### 7.1 Gain Matrices
For each layer j:
- L^f_j(k): filter gain matrix
- L^c_j(k): classifier gain matrix

They are not arbitrary learning rates; they are selected/adjusted to satisfy sufficient conditions for bounded convergence.

### 7.2 Condition Checks
The paper references conditions (e.g., condition (20) and condition (37)) that constrain L matrices and related bounds.
Implementation spec:
- At a chosen cadence (typically once per batch, or periodically), evaluate whether the conditions hold.
- If a condition is violated, adjust L^f_j and/or L^c_j (e.g., shrink/scale) to restore feasibility.
- Document the exact rule used in the original code (often a scale-down or projection).

Cursor-facing requirement:
- Put the condition-check logic in a dedicated module/function, e.g.:
  - check_conditions_and_update_gains(...)
- Ensure it has logs/diagnostics outputs.

---

## 8. Training Loop (Exact Control Flow)
### 8.1 Initialization
- Initialize CNN filter parameters V_j (standard init)
- Initialize auxiliary classifiers W_j
- Initialize adjustable ReLU parameter: a = 1
- Initialize pooling: average pooling
- Initialize snapshots:
  - \bar{V}_j^{[0]} := V_j
  - \bar{W}_j^{[0]} := W_j

### 8.2 For each mini-batch b = 1..:
Within batch b, run per-sample updates (online-in-batch):
1) FOR each sample k in batch:
   a) Forward pass through each system S_j
      - Compute layer-j features and y_hat_j(k)
   b) Compute classifier error e^c_j(k)
   c) Compute filter learning error e^f_j(k)
   d) Filter update:
      - Update V_j using frozen \bar{W}_j^{[b-1]}
   e) Classifier update:
      - Update W_j using frozen \bar{V}_j^{[b-1]}

2) End of batch:
   a) Condition check & gain adjustment:
      - Update L^f_j, L^c_j if needed
   b) Adjustable ReLU schedule:
      - Update a using norm of training error:
        a = (||e|| - 0.01) / ||e||
        with lower bound a >= 0.01
   c) Pooling switch rule:
      - When a reaches 0.01 (or within tolerance), switch average pooling -> max pooling
   d) Refresh snapshots:
      - \bar{V}_j^{[b]} := V_j
      - \bar{W}_j^{[b]} := W_j

### 8.3 Batch Size and Sampling
- Paper’s case study uses batch size B = 16 as an example.
- Updates are per-sample within batch (not standard batch-gradient accumulation).

---

## 9. Outputs, Metrics, and Expected Behaviors
### 9.1 Convergence Meaning
- Training error is guaranteed to be bounded and convergent to a neighborhood determined by approximation error (not necessarily exactly 0).
- Practically: stability/robustness over long training without “late-stage divergence” seen in some optimizers.

### 9.2 Metrics
- Track per-system accuracy (each S_j can be evaluated)
- Track final chosen system (e.g., last conv layer system) test accuracy
- Track training loss/error norm ||e||
- Track a schedule value and pooling mode
- Track condition-check pass/fail counts and gain scaling events

---

## 10. Implementation Requirements for a Cursor-Friendly Repo
To make Cursor reliably reason about your implementation, the repo MUST contain:
1) This file (algorithm_spec.md)
2) A symbol mapping table (symbol_mapping.md) — separate step
3) A single authoritative training loop file with clear sections:
   - forward_all_systems
   - compute_errors
   - update_filters_eq15
   - update_classifiers_eq19
   - check_conditions_update_gains
   - update_adjustable_relu_and_pooling
4) Logging:
   - batch index, sample index
   - ||e||, a, pooling type
   - gains L^f, L^c (or scale factors)
   - test accuracy

---

## 11. Assumptions and Non-Negotiables (for correctness)
- MUST keep the “frozen across batch” snapshot design (\bar{V}, \bar{W}) exactly.
- MUST do per-sample updates (online-in-batch), not batch gradient accumulation.
- MUST include gain condition checks or replicate the original code’s substitute.
- Adjustable ReLU parameter a is batch-wise (do not change a per sample).
- Pooling switch is triggered by the a schedule rule (or original code’s equivalent).

---

## 12. Open Items to Resolve with the Original Code
When integrating the original repo, explicitly answer:
- Exact form of D_{j,i}(k) and how it is computed in tensors
- Exact definition of e^f_j(k) used in code
- Exact condition-check formulas (paper conditions vs code implementation)
- Exact adjustable ReLU function definition (paper vs code)
- Which system’s output is used for final evaluation (last layer vs ensemble)

Add these answers as a final section called:
"Resolved Implementation Decisions"
once verified from code.

---
End of Spec