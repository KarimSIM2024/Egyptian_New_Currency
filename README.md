# Egyptian_New_Currency
This project builds a complete image classification system for the new 2023 Egyptian currency using a custom Convolutional Neural Network. The model is implemented in Python with Keras inside a Jupyter Notebook. The system identifies different banknote denominations without using transfer learning or any pre-trained networks.

## Project Plan (for team discussion)

Objectives
- Train a from-scratch CNN (no transfer learning) to classify Egyptian banknotes.
- Target accuracy: >= 93% on validation/test with reproducible runs.

Data (Squad A)
- Audit dataset: class list, counts, resolution, corrupt files.
- Decide input size (e.g., 224x224) and color mode (RGB vs grayscale).
- Define splits (train/val/test) with fixed seed; use existing splits if trusted.
- Preprocess: rescale to [0,1]; consider per-channel standardization.
- Augment: flips, small rotation/zoom/translation; maybe brightness/contrast jitter.
- Pipeline: prefer tf.data with cache/prefetch; pick batch sizes (32/64) and num_parallel_calls.

Model (Squad B)
- Base block: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout; repeat with filters 32/64/128 (tune depth).
- Head: Flatten or GlobalAvgPool -> Dense(256) + dropout -> Dense(128) + dropout -> Softmax.
- Regularization: tune dropout (0.2-0.5); optional L2 on convs.
- Optimizer/loss: Adam or SGD+momentum; loss = sparse categorical crossentropy; metric = accuracy.
- Hyperparameters to search: LR (1e-4 to 3e-3), epochs (30-80), scheduler (ReduceLROnPlateau or cosine), batch (32/64), augmentation strength.

Training Workflow (Squad B)
- Fit with callbacks: EarlyStopping (val_loss), ModelCheckpoint (best val_acc), ReduceLROnPlateau.
- Log artifacts: best model file, training history (JSON/CSV), optional TensorBoard.
- Use mixed precision if GPU allows; monitor throughput.

Evaluation (Squad C)
- Load best model; run on held-out test (no shuffle).
- Metrics: accuracy, per-class precision/recall/F1, macro/micro averages.
- Confusion matrix plot; inspect common confusions between denominations.
- Error analysis: sample misclassifications with images; note lighting/pose issues.
- Optional: calibration (reliability), threshold tuning.

Notebook (Squad C with A/B inputs)
- Sections: problem statement, dataset exploration visuals, preprocessing summary, model summary, training curves, evaluation metrics + confusion matrix, sample predictions, takeaways/future work.
- Keep end-to-end runnable; clear cell order and minimal manual edits.

Validation & Reproducibility
- Set random seeds; document hardware (GPU type).
- Success criteria: >= 93% accuracy, stable learning curves, confusion matrix without dominant confusions.

Risks & Mitigations
- Class imbalance: monitor per-class F1; consider class weights/oversampling.
- Domain shift (lighting/background): stronger augmentation/normalization.
- Overfitting: early stopping, dropout/L2, more data if available.

Next Steps
- Squad A: finalize input size, augmentation plan, and confirm splits.
- Squad B: prototype baseline CNN and run a short sanity train.
- Squad C: prepare evaluation notebook section and plotting utilities.
