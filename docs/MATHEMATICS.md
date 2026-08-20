# Mathematical Formulation

## Forward model

A simplified advection/dispersion model is used for pretraining:

\[
C(x,y) \propto \frac{Q}{u\sigma(x)}
\exp\left(-\frac{y^2}{2\sigma(x)^2}\right)
\]

with zero/upweighted concentration only downwind of the source.

The full atmospheric problem is related to:

\[
\frac{\partial C}{\partial t}+\mathbf{u}\cdot\nabla C
=
D\nabla^2C+S-\lambda C.
\]

MethaneTwin-R does not claim its pretraining simulator is a CFD replacement; it is a controlled physics prior.

## Inverse problem

Given observed enhancement \(C_{\rm obs}\), estimate source position \(s\) and emission rate \(Q\):

\[
(s^*,Q^*)=
\arg\min_{s,Q}
\|C_{\rm obs}-\mathcal T(s,Q,\mathbf u)\|_2^2.
\]

A classical grid-search/least-squares inverse is included as a baseline.

## Multi-task model

\[
h=f_\theta(X,\mathbf u)
\]

with heads:

\[
p_{\rm plume},\quad \hat M,\quad \hat s,\quad
\log \hat Q,\quad \log \hat\sigma_Q^2.
\]

## Heteroscedastic emission objective

For \(y=\log Q\):

\[
\mathcal L_Q=
\frac12\left[
e^{-\ell}(y-\hat y)^2+\ell
\right],
\]

where \(\ell=\log \sigma^2\).

## Physics consistency

Predicted source/emission are passed through a differentiable plume operator:

\[
\mathcal L_{\rm phys}
=
\|\tilde C_{\rm obs}-
\tilde{\mathcal T}(\hat s,\hat Q,\mathbf u)\|_2^2.
\]

## Conformal interval

Using a dedicated calibration set:

\[
r_i=|\log Q_i-\log \hat Q_i|
\]

and calibrated quantile \(q_{1-\alpha}\),

\[
[\hat Q e^{-q},\hat Q e^q]
\]

gives a multiplicative emission interval.

## Persistence

For \(d\) detections in \(n\) valid observation opportunities:

\[
p\sim{\rm Beta}(\alpha_0+d,\beta_0+n-d).
\]
