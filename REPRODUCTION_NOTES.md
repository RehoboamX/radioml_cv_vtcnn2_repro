# Reproduction Notes

## Scope

This repository implements the VTCNN2 and CV-VTCNN2 comparison discussed by Tu et al.
2020. It follows the paper and official RadioML example for the model family and key
layer dimensions, while using a documented clean fixed-budget training protocol for an
auditable comparison.

It does not add data augmentation, learning-rate schedules, ensembling, multi-seed
selection, or architecture changes intended only to improve accuracy.

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
SNR, with 800 randomly selected samples used for training and the remaining 200 used
for test. This repository creates that split independently for every modulation/SNR
group with seed `2016`.

There is no validation set. The test set is not used for early stopping, checkpoint
selection, or hyperparameter selection. This avoids the test-as-validation leakage in
the official VTCNN2 notebook.

## CV-VTCNN2 Representation And Layers

Tu et al. represent complex tensors with the first half of feature maps as real and the
second half as imaginary. This repository uses that channel-stacked representation
everywhere. For one complex input channel, the input tensor is `[I, Q]`.

Complex convolution and complex dense layers implement:

```text
(A + iB)(x + iy) = (Ax - By) + i(Bx + Ay)
```

The `same_width` variant keeps hidden complex widths `256 -> 80 -> 256`. The
`param_matched` variant uses widths `181 -> 57 -> 181`, approximately scaling hidden
widths by `1/sqrt(2)` to keep the number of real trainable scalars close to VTCNN2.

The trainable parameter counts before training are approximately:

| Model | Trainable parameters | Relative to VTCNN2 |
| --- | ---: | ---: |
| VTCNN2 | 2,830,427 | 1.000 |
| CV-VTCNN2 `param_matched` | 2,791,507 | 0.986 |
| CV-VTCNN2 `same_width` | 5,537,963 | 1.956 |

The same-width model is a structural replacement experiment. The param-matched model
is the fairer comparison when attributing differences to complex-valued modeling rather
than parameter count.

## Complex Activation, Dropout, And Classification Head

The paper does not define the exact complex activation, dropout mask, or conversion
from complex class features to real Softmax inputs for CV-VTCNN2.

This implementation applies ReLU independently to real and imaginary components.
Dropout uses one shared mask for the real and imaginary components of each complex
feature. After the final complex hidden layer, real and imaginary features are
concatenated and passed to one real-valued linear classifier that emits 11 signed real
logits for `CrossEntropyLoss`.

The real-valued classification head is a PyTorch/AMC implementation choice. It uses
both real and imaginary features, but it means the final classification layer is not a
complex dense layer. This choice can affect numerical agreement with Tu et al.

## Complex Batch Normalization

`ComplexBatchNorm1d` implements 2x2 real/imaginary covariance whitening with learnable
symmetric scaling and complex shift. It is implemented for completeness but is not
inserted into CV-VTCNN2 because the official VTCNN2 has no batch normalization and the
paper does not clearly specify adding it to this architecture.

## Initialization

Tu et al. follow Trabelsi-style complex initialization: sample weight magnitude from a
Rayleigh distribution and phase independently. This implementation uses Rayleigh
inverse-CDF sampling with uniform phase. The scale is `1/sqrt(fan_in + fan_out)` for
Glorot-style layers and `1/sqrt(fan_in)` for He-style dense layers. Exact framework
defaults may differ from the original Keras/TensorFlow implementation.

The real VTCNN2 follows the notebook's intent: Glorot initialization for convolution
and He initialization for dense layers.

## Fixed-Budget Training Protocol

- Optimizer: Adam
- Learning rate: `1e-3`
- Batch size: `1024`
- Epochs: `100`
- Seed: `2016`
- Validation set: none
- Early stopping: none
- Checkpoint selection: final epoch only
- Test evaluation: once after training

Tu et al. report Adam with learning rate `1e-4` and early-stopping patience `20`, but do
not document the validation data used for early stopping. The official notebook uses a
global 50/50 split and reuses the test set as validation. This repository intentionally
does not copy that leakage-prone behavior.

The fixed-budget protocol is not a literal reproduction of every Tu et al. training
hyperparameter. It is a clean AMC baseline designed to compare aligned training loss
and accuracy curves under the same number of epochs and training samples.

## Committed Single-Seed Results

All models use the same 800/200 split, seed, batch size, learning rate, and 100-epoch
training budget. The final-epoch checkpoints give:

| Model | Final train accuracy | Test accuracy |
| --- | ---: | ---: |
| VTCNN2 | 0.5431 | 0.5383 |
| CV-VTCNN2 `param_matched` | 0.6116 | 0.5356 |
| CV-VTCNN2 `same_width` | 0.6776 | 0.5313 |

The complex models reach substantially higher training accuracy but do not outperform
VTCNN2 on the final test set in this single-seed fixed-budget run. This suggests a
generalization gap under the chosen real-valued classification head and training
protocol. The result should be reported as-is rather than used to tune the epoch budget
or learning rate against the test set.
## Expected Sources Of Numerical Difference

- Fixed-budget training versus Tu et al.'s incompletely documented early stopping
- Per-group 800/200 split versus the notebook's global 50/50 split
- PyTorch versus Keras/TensorFlow convolution and initialization details
- Complex activation, shared-mask dropout, and the real-valued classification head
- Exact sample ordering and random-number generators
- Same-width versus parameter-matched CV-VTCNN2 capacity

Results that differ from the paper should be analyzed through these documented choices
rather than by adding unrelated training tricks.