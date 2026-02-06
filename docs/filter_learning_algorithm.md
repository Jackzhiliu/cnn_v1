FILTER LEARNING FRAMEWORK — TEXT VERSION
Source: Learning Laws for Deep CNNs with Guaranteed Convergence
(Structured for programmatic parsing / Cursor ingestion)

============================================================
SECTION 1 — MODEL REPRESENTATION
============================================================

Deep CNN output is modeled as:

y =
phi(
    W *
    F{
        Gamma_n R_n(
            ...
            Gamma_2 R_2(
                Gamma_1 R_1( X_f1 * V1 )
            ) * V2 ...
        ) * Vn
    }
)

Where:

X_fj      = filter-dependent input matrix
Vj        = convolution filter matrix of layer j
Gamma_j   = pooling matrix
Rj        = activation matrix (Leaky ReLU)
F(.)      = flatten operator
W         = fully connected weight matrix
phi(.)    = classification activation

This corresponds to Eq.(1). 

------------------------------------------------------------

Pooling matrices:

Average pooling:

Gamma =
[1/p^2 ... 1/p^2 0 ... 0
 ...
 0 ... 0 1/p^2]

Max pooling:

Gamma_ij =
1 if element is max
0 otherwise

Corresponds to Eq.(2)-(3). 

------------------------------------------------------------

Activation matrix:

R_j,i = diag(r_i1, r_i2, ..., r_in)

r_iq(k) =
    1      if x_fjq * v_ji >= 0
    0.01   otherwise

Eq.(8) 

============================================================
SECTION 2 — FILTER LEARNING SYSTEM STRUCTURE
============================================================

For each CNN layer j:

sigma_j = F( Gamma_j R_j ( X_fj * V_j ) )

y_CNN = phi( W_j * sigma_j )

Eq.(6)-(7) 

Expanded:

y_CNN =
phi(
  sum_{i=1..nfj}
     W_j,i * Gamma_j,i * R_j,i * X_fj * v_j,i
)

Eq.(9)

============================================================
SECTION 3 — MINI-BATCH SNAPSHOT TRAINING
============================================================

Initialize:

V_bar_j[0]
W_bar_j[0]

For batch b:

Update filters using W_bar[b-1]
Update classifiers using V_bar[b-1]

After batch:

V_bar[b] = V_hat
W_bar[b] = W_hat

Eq.(10) description 

============================================================
SECTION 4 — FILTER UPDATE LAW
============================================================

Estimated output:

y_hat_fj(k) =
phi(
  W_bar[b-1]_j *
  sigma_hat_j(k)
)

Eq.(11)

Error:

e_fj(k) = y(k) - y_hat_fj(k)

Eq.(14)

Update rule:

v_hat_j,i(k+1) =
v_hat_j,i(k)
+
alpha_f_j *
D_j,i(k) *
L_f_j(k) *
e_fj(k)

Eq.(15)

Where:

D_j,i(k) =
X_fj^T(k)
R_hat_j,i(k)
Gamma_hat_j,i^T(k)
W_bar_j,i^T[b-1]

L_f_j = positive diagonal gain matrix

============================================================
SECTION 5 — CLASSIFIER UPDATE LAW
============================================================

Estimated output:

y_hat_cj(k) =
phi(
    W_hat_j(k) *
    sigma_bar_j[b-1](k)
)

Eq.(16)

Error:

e_cj(k) = y(k) - y_hat_cj(k)

Eq.(18)

Update:

W_hat_j(k+1) =
W_hat_j(k)
+
alpha_c_j *
L_c_j(k) *
e_cj(k) *
sigma_bar_j^T[b-1](k)

Eq.(19)

============================================================
SECTION 6 — CONVERGENCE CONDITIONS
============================================================

Learning gains must satisfy:

2 L_fjm phi_jM − d_fjM L_fjM^2 > 0
2 L_cjm phi_jM − d_cjM L_cjM^2 > 0

Eq.(20)

Then training errors converge to bounds dependent on
approximation errors epsilon_f and epsilon_c.

============================================================
SECTION 7 — LYAPUNOV FUNCTION
============================================================

Define objective:

V(k) =
sum_j 1/alpha_fj sum_i ||Delta v_j,i||^2
+
sum_j 1/alpha_cj Tr(Delta W_j^T Delta W_j)

Eq.(21)

Change:

Delta V(k) = V(k+1) − V(k)

Expanded expression Eq.(22)

If conditions hold:

Delta V(k) <= 0

Therefore errors converge.

============================================================
SECTION 8 — ADJUSTABLE RELU SCHEDULE
============================================================

Activation slope parameter a:

Initial:
a = 1

Update between batches:

a = (||e|| − 0.01) / ||e||

Stop when:
a >= 0.01

Switch pooling:
Average → Max

Case study description 

============================================================
SECTION 9 — FULL ALGORITHM (EXECUTABLE FORM)
============================================================

Initialize V_bar, W_bar

FOR each batch b:

    FOR each sample k:

        Compute sigma_hat_j
        Compute e_fj
        Update v_j via Eq.(15)

        Compute sigma_bar_j
        Compute e_cj
        Update W_j via Eq.(19)

    END

    Update snapshots:
        V_bar <- V_hat
        W_bar <- W_hat

    Adjust a
    Adjust L matrices

END

============================================================
END OF FILE
============================================================