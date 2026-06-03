# RadioML CV-VTCNN2 Reproduction

PyTorch reproduction of the real-valued VTCNN2 baseline and the complex-valued
CV-VTCNN2 baseline discussed in Tu et al. 2020, using RadioML2016.10A.

The repository prioritizes a faithful, documented, auditable baseline over accuracy
tuning. The default CV-VTCNN2 is the paper-structure same-width model. The
`param_matched` variant is the appropriate comparison when controlling trainable
parameter count. See `REPRODUCTION_NOTES.md` before comparing numerical results.

## References

- Tu et al., "Complex-Valued Networks for Automatic Modulation Classification":
  https://www.eng.auburn.edu/~szm0001/papers/TuTVT2020.pdf
- Official RadioML VTCNN2 example notebook:
  https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb
- Optional fixed RadioML examples:
  https://github.com/sofwerx/radioML_examples

## Dataset

Download RadioML2016.10A separately. Place the Python pickle file at:

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

## Run

Run commands from this directory:

```bash
python prepare_data.py
python sanity_checks.py --dataset data/RML2016.10a_dict.dat
python train_vtcnn2.py
python train_cv_vtcnn2.py
python evaluate.py --checkpoint results/vtcnn2_best.pt --output results/vtcnn2_test_metrics.json
python evaluate.py --checkpoint results/cv_vtcnn2_best.pt --output results/cv_vtcnn2_test_metrics.json
python plot_results.py
```

The default split is balanced independently for every modulation and SNR:

```text
720 train + 80 validation + 200 test
```

The 720/80 subsets come from the paper's 800-sample training pool. The 200 test
samples are used only for final evaluation.

## Parameter-Matched Comparison

Train and evaluate the approximate parameter-matched complex model:

```bash
python train_cv_vtcnn2.py \
  --variant param_matched \
  --checkpoint results/cv_vtcnn2_param_matched_best.pt \
  --history results/cv_vtcnn2_param_matched_history.json
python evaluate.py \
  --checkpoint results/cv_vtcnn2_param_matched_best.pt \
  --output results/cv_vtcnn2_param_matched_test_metrics.json
python plot_results.py \
  --cv-history results/cv_vtcnn2_param_matched_history.json \
  --cv-metrics results/cv_vtcnn2_param_matched_test_metrics.json \
  --figures-dir figures/param_matched
```

The committed single-seed clean-validation results are:

| Model | Trainable parameters | Test accuracy |
| --- | ---: | ---: |
| VTCNN2 | 2,830,427 | 0.5348 |
| CV-VTCNN2 `param_matched` | 2,791,518 | 0.5386 |
| CV-VTCNN2 same-width | 5,537,974 | 0.5474 |

The parameter-matched model is about 1.4% smaller than VTCNN2 and is the fairer
comparison for attributing gains to complex-valued modeling. The same-width result is
retained as a structural replacement experiment, not a parameter-count-controlled
comparison.

## Optional Modes

Create the notebook-like test-as-validation split:

```bash
python prepare_data.py --validation-mode paper_like --output-dir results/split_paper_like
```

`paper_like` can leak test information and is not the default.

## Outputs

- `results/split/split_indices.npz`: deterministic split indices
- `results/split/split_metadata.json`: split counts and metadata
- `results/*_best.pt`: best validation checkpoints
- `results/*_history.json`: training logs and parameter counts
- `results/*_test_metrics.json`: overall and per-SNR test accuracy
- `figures/*_train_loss.png`: training loss curves
- `figures/*_val_accuracy.png`: validation accuracy curves
- `figures/per_snr_test_accuracy.png`: same-width final per-SNR comparison
- `figures/param_matched/per_snr_test_accuracy.png`: parameter-matched fair comparison

## Tests

```bash
python -m pytest tests -q
python sanity_checks.py
```

Passing `--dataset` to `sanity_checks.py` also validates all per-SNR/per-class split
counts against the real dataset.
