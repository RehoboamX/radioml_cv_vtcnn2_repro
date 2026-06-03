# RadioML CV-VTCNN2 Reproduction

PyTorch implementation of real-valued VTCNN2 and complex-valued CV-VTCNN2 for
RadioML2016.10A, based on Tu et al. 2020 and the official RadioML VTCNN2 example.

This repository uses a clean fixed-budget AMC protocol: every modulation/SNR group is
split into 800 training samples and 200 test samples, all models train for the same 100
epochs, and the test set is evaluated only after training. There is no validation set,
early stopping, or checkpoint selection.

## References

- Tu et al., "Complex-Valued Networks for Automatic Modulation Classification":
  https://www.eng.auburn.edu/~szm0001/papers/TuTVT2020.pdf
- Official RadioML VTCNN2 example notebook:
  https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb

## Dataset

Download RadioML2016.10A separately and place the pickle file at:

```text
data/RML2016.10a_dict.dat
```

The file is intentionally ignored by git. It must be a dictionary keyed by
`(modulation, snr)` with arrays shaped `(1000, 2, 128)`. The dataset contains 11
modulations and SNR values from -20 dB to 18 dB in 2 dB steps.

## Environment

```bash
python -m pip install -r requirements.txt
```

The scripts use CUDA automatically when available.

## Fixed-Budget Protocol

- Split: 800 train / 200 test per modulation and SNR
- Validation set: none
- Optimizer: Adam
- Learning rate: `1e-3`
- Batch size: `1024`
- Epochs: `100`
- Seed: `2016`
- Checkpoint: final epoch only
- Test evaluation: overall accuracy and per-SNR accuracy after training

The fixed learning rate and epoch budget are shared by all models so their training
loss and training accuracy curves can be compared directly.

## Run

Run commands from this directory:

```bash
python prepare_data.py
python sanity_checks.py --dataset data/RML2016.10a_dict.dat

python train_vtcnn2.py
python train_cv_vtcnn2.py
python train_cv_vtcnn2.py \
  --variant param_matched \
  --checkpoint results/cv_vtcnn2_param_matched_final.pt \
  --history results/cv_vtcnn2_param_matched_history.json

python evaluate.py \
  --checkpoint results/vtcnn2_final.pt \
  --output results/vtcnn2_test_metrics.json
python evaluate.py \
  --checkpoint results/cv_vtcnn2_same_width_final.pt \
  --output results/cv_vtcnn2_same_width_test_metrics.json
python evaluate.py \
  --checkpoint results/cv_vtcnn2_param_matched_final.pt \
  --output results/cv_vtcnn2_param_matched_test_metrics.json

python plot_results.py
```

## Models

- `VTCNN2`: official real-valued VTCNN2 structure.
- `CV-VTCNN2 same_width`: complex convolution and hidden dense layers with the same
  widths as VTCNN2.
- `CV-VTCNN2 param_matched`: reduced complex widths for a trainable-parameter-count
  comparison with VTCNN2.

CV-VTCNN2 concatenates the real and imaginary parts of its final complex hidden
features, then uses one real-valued linear classifier to produce 11 signed logits for
`CrossEntropyLoss`.

## Outputs

- `results/split/split_indices.npz`: deterministic train/test indices
- `results/split/split_metadata.json`: split counts and metadata
- `results/*_final.pt`: final-epoch checkpoints
- `results/*_history.json`: fixed-budget training histories and parameter counts
- `results/*_test_metrics.json`: overall and per-SNR test accuracy
- `figures/train_loss_comparison.png`: aligned training loss curves
- `figures/train_accuracy_comparison.png`: aligned training accuracy curves
- `figures/per_snr_test_accuracy.png`: final per-SNR test comparison

## Tests

```bash
python -m pytest tests -q
python sanity_checks.py
```

Passing `--dataset` to `sanity_checks.py` validates all per-SNR/per-class split counts
against the real dataset.