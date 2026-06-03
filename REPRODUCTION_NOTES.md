# Reproduction Notes

## Scope

This repository reproduces the VTCNN2 and CV-VTCNN2 comparison from Tu et al. 2020.
It does not add data augmentation, learning-rate schedules, ensembling, multi-seed
selection, or architecture changes intended to improve accuracy.

## VTCNN2 Structure

The official RadioML notebook is the source for concrete layer parameters:

| Stage | Parameters | Output shape |
| --- | --- | --- |
| Input | I/Q waveform | `2 x 128` |
| Reshape | add one image channel | `1 x 2 x 128` |
| Zero padding | time axis: 2 on each side | `1 x 2 x 132` |
| Conv 1 | 256 filters, kernel `(1, 3)`, valid, ReLU | `256 x 2 x 130` |
| Dropout | `0.5` | `256 x 2 x 130` |
| Zero padding | time axis: 2 on each side | `256 x 2 x 134` |
| Conv 2 | 80 filters, kernel `(2, 3)`, valid, ReLU | `80 x 1 x 132` |
| Dropout | `0.5` | `80 x 1 x 132` |
| Flatten | direct flatten | `10560` |
| Dense 1 | 256 units, ReLU | `256` |
| Dropout | `0.5` | `256` |
| Dense 2 | 11 units | `11` |

The notebook applies Softmax in the model. This PyTorch implementation returns logits
and uses `CrossEntropyLoss`, which is the numerically stable equivalent.

## Data Split

Tu et al. state that RadioML2016.10A contains 1000 samples per modulation class per
SNR, and that 800 randomly selected samples are used for training while the remaining
200 are used for test. The paper also states that classes are balanced.

The default `clean` protocol first creates that 800/200 split independently for every
modulation/SNR group, then takes 80 samples from the 800-sample training pool for
validation. This yields 720 train, 80 validation, and 200 final test samples per group.

The official notebook instead performs a global 50/50 random split and uses the test
set as validation. That behavior is not used by default. The explicit `paper_like`
validation mode makes validation identical to test and is provided only for diagnostic
comparison; it can cause test leakage.

## CV-VTCNN2 Representation And Layers

Tu et al. represent complex tensors with the first half of feature maps as real and
the second half as imaginary. This repository uses that channel-stacked representation
everywhere.

For one complex input channel, the input tensor is `[I, Q]`. Complex convolution and
complex dense layers implement:

```text
(A + iB)(x + iy) = (Ax - By) + i(Bx + Ay)
```

The paper states that CV-VTCNN2 has the same architecture as VTCNN2 with convolution
and dense layers replaced by complex-valued versions. Therefore the default
`paper_faithful` variant keeps complex widths `256 -> 80 -> 256 -> 11`. The
`same_width` name is an alias for this configuration.

The paper also states that complex networks use half the real-valued parameter amount
for fair representation capacity, but it does not provide the exact reduced VTCNN2
widths. The optional `param_matched` variant uses widths `181 -> 57 -> 181 -> 11`,
approximately scaling hidden widths by `1/sqrt(2)`. It is not the default result.

## Complex Activation, Dropout, And Logits

The paper says VTCNN2 uses ReLU except for the output Softmax, but it does not define a
complex ReLU for this model. This implementation applies ReLU independently to real
and imaginary components, a conservative component-wise choice.

Dropout uses one shared mask for the real and imaginary components of each complex
activation so that dropping a feature removes the whole complex value.

The paper does not state how complex class scores become real Softmax inputs. This
implementation converts the final complex score to a non-negative real logit using
magnitude squared, `real^2 + imag^2`, then passes those logits to `CrossEntropyLoss`.
This is a PyTorch implementation choice and can affect numerical agreement.

## Complex Batch Normalization

`ComplexBatchNorm1d` implements the 2x2 real/imaginary covariance whitening described
by Tu et al. and Trabelsi et al., including learnable symmetric scaling and complex
shift. It is implemented for completeness but is not inserted into default CV-VTCNN2:
the official VTCNN2 has no batch normalization, and the paper only explicitly says to
replace convolution and dense layers.

## Initialization

Tu et al. follow Trabelsi-style complex initialization: sample weight magnitude from a
Rayleigh distribution and phase independently. This implementation uses Rayleigh
inverse-CDF sampling with uniform phase. The scale is `1/sqrt(fan_in + fan_out)` for
Glorot-style layers and `1/sqrt(fan_in)` for He-style dense layers. Exact framework
defaults may differ from the original Keras/TensorFlow implementation.

The real VTCNN2 follows the notebook's intent: Glorot initialization for convolution
and He initialization for dense layers.

## Training

- Optimizer: Adam
- Learning rate: `1e-4`, from Tu et al.
- Early stopping patience: `20`, from Tu et al.
- Maximum epochs: `500`, intentionally high so early stopping controls termination
- Batch size: `1024`, from the official notebook
- Seed: `2016`, matching the notebook's split seed
- Model selection: best validation accuracy checkpoint

The paper does not fully specify batch size, validation policy, random seed, complex
output conversion, or exact reduced widths. These are likely sources of discrepancy.

## Expected Sources Of Numerical Difference

- Clean validation versus test-as-validation
- Per-group 800/200 split versus the notebook's global 50/50 split
- PyTorch versus Keras/TensorFlow convolution and initialization details
- Complex activation, dropout, and final magnitude-squared logits
- Exact sample ordering and random-number generators
- Early stopping criterion and selected epoch
- Whether the paper's plotted CV-VTCNN2 used same-width or parameter-matched layers

Results that differ from the paper should be analyzed through these documented choices
rather than by adding unrelated training tricks.
