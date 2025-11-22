# Residual CNN-LSTM Autoencoder for Network Intrusion Detection (UNSW-NB15)

This project implements a **high-accuracy deep learning anomaly detector** for the [UNSW-NB15](https://www.kaggle.com/) network intrusion dataset using a **Residual CNN-LSTM Autoencoder** and a **hybrid reconstruction error** (MSE + cosine distance).

It is designed to work in environments like **Kaggle Notebooks**, but can easily be adapted to run locally.

---

## 🔍 Problem Statement

Traditional IDS systems often struggle with:

* Highly **imbalanced** traffic (few attacks vs many normal flows or vice versa).
* **Sequential** nature of network traffic (packets/flows over time).
* **Skewed** feature distributions (durations, bytes, loads, etc.).

This project addresses these with:

* **Log-transformed numeric features**
* **Top-K categorical encoding**
* **Sliding-window sequence modeling**
* **Residual CNN + Bi-LSTM Autoencoder**
* **Hybrid anomaly score** tuned via **precision-recall F1** on an imbalanced test set

---

## ✨ Key Features

* ✅ **Advanced preprocessing**

  * Log1p transform for skewed numeric features:

    * `dur`, `sbytes`, `dbytes`, `sload`, `dload`, `spkts`, `dpkts`
  * Categorical top-K handling for:

    * `proto`, `service`, `state`
      (keep top 6 most frequent; others → `"other"`)
  * One-hot encoding of categorical columns
  * Min-Max scaling of all features

* ✅ **Sequence modeling via sliding window**

  * Sequence length: **16**
  * Stride: **2** (heavily overlapping windows → more training data)
  * Sequence label = **attack** if > **30%** of frames in the window are attacks

* ✅ **Residual CNN-LSTM Autoencoder**

  * 1D convolutions + **residual blocks** to preserve information
  * **Bi-LSTM bottleneck** to capture temporal dynamics
  * Symmetric upsampling decoder with TimeDistributed output layer

* ✅ **Hybrid anomaly score**

  * **MSE** (magnitude error) +
  * **Cosine distance** (directional mismatch)
  * Combined score:
    `total_error = 0.7 * mse_scaled + 0.3 * cosine_distance`

* ✅ **Threshold selection via F1 on Precision-Recall curve**

  * Best for **imbalanced** attack/normal distributions

---

## 📊 Results (on UNSW-NB15 Test Set)

Using the auto-selected optimal threshold from the precision-recall F1 curve:

```text
==================================================
FINAL HIGH-ACCURACY RESULTS
==================================================
              precision    recall  f1-score   support

      Normal       0.88      0.86      0.87     24365
      Attack       0.95      0.96      0.95     63298

    accuracy                           0.93     87663
   macro avg       0.91      0.91      0.91     87663
weighted avg       0.93      0.93      0.93     87663

ROC-AUC Score: 0.9679
Optimal Threshold (hybrid error): ~0.107
New Feature Count after encoding: 62
Training on normal sequences: 14,788
```

---

## 🧱 Model Architecture

**Input:** sequence of length `SEQ_LEN = 16` with `n_features` features per timestep.

### Encoder

1. `Conv1D(64, kernel=3, relu)`
2. **Residual Block (64 filters)**
3. `MaxPooling1D(2)` → length 16 → 8
4. **Residual Block (128 filters)**
5. `MaxPooling1D(2)` → length 8 → 4
6. `Bidirectional(LSTM(100, return_sequences=False))`
7. `Dense(latent_dim=64, relu)` → **latent_vector**

### Decoder

1. `RepeatVector(4)`  (to match downsampled length)
2. `Bidirectional(LSTM(100, return_sequences=True))`
3. `UpSampling1D(2)` → length 4 → 8
4. **Residual Block (128 filters)**
5. `UpSampling1D(2)` → length 8 → 16
6. **Residual Block (64 filters)**
7. `TimeDistributed(Dense(n_features))` → reconstruction

---

## 🧪 Data & Labeling Strategy

### Dataset

* **UNSW-NB15** training and testing splits:

  * `UNSW_NB15_training-set.csv`
  * `UNSW_NB15_testing-set.csv`
* These are expected under:

```text
/kaggle/input/unsw-nb15/
    ├── UNSW_NB15_training-set.csv
    └── UNSW_NB15_testing-set.csv
```

> If you run locally, just update the paths in `load_and_preprocess_high_acc(...)`.

### Labels

* Original binary label column: `label`

  * `0` → Normal
  * `1` → Attack
* For sequences:

  * Look at 16 consecutive frames (with stride 2).
  * If **more than 30%** of the frames in that sequence are labeled `1`, the whole sequence is labeled **Attack (1)**; otherwise **Normal (0)**.

### Training & Test Split

* **Training**:

  * Use **only normal sequences** (`y_train_seq == 0`) to train the autoencoder.
  * 80/20 split of these normal sequences → train/validation.
* **Testing**:

  * Use **all** sequences from the test set (normal + attack) for evaluation.

---

## ⚙️ Project Structure (Logical)

All logic is contained in a single script/notebook cell, with the following main components:

* **Preprocessing & Encoding**

  * `handle_categorical_top_k(df, col, k=5)`
  * `load_and_preprocess_high_acc(train_path, test_path)`

* **Sequence Creation**

  * `create_sequences(data, labels, sequence_length=16, stride=2)`

* **Model Definition**

  * `residual_block(x, filters, kernel_size=3)`
  * `build_high_acc_model(seq_len, n_features, latent_dim=64)`

* **Hybrid Error Calculation**

  * `calculate_hybrid_error(model, sequences)`

* **Pipeline Entry Point**

  * `main()`

    * Loads data
    * Preprocesses features
    * Creates sequences
    * Builds & trains the model
    * Computes hybrid errors
    * Selects optimal threshold via precision-recall F1
    * Prints classification metrics & ROC-AUC
    * Plots loss history & error distributions

If you turn this into a Python script, you’d typically save it as e.g. `main.py` and keep this structure.

---

## 📦 Requirements

Core dependencies used:

```text
Python 3.10+
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow >= 2.18.0
```

In `pip` form:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

---

## 🚀 How to Run

### On Kaggle

1. Create a new **Notebook**.
2. Add the **UNSW-NB15** dataset to the notebook (as shown in the metadata).
3. Paste the code into a cell (or import from a `.py` file).
4. Run the cell; `main()` is executed at the bottom:

```python
if __name__ == "__main__":
    main()
```

You should see:

* TensorFlow version printout
* Preprocessing logs
* Training progress for up to 40 epochs (with early stopping / LR reduction)
* Final evaluation metrics and threshold
* Two plots:

  * Loss history (train vs validation)
  * Error distribution (Normal vs Attack) with threshold line

### Locally

1. Download the UNSW-NB15 training & testing CSVs.
2. Update the paths in:

```python
train_df, test_df = load_and_preprocess_high_acc(
    '/path/to/UNSW_NB15_training-set.csv',
    '/path/to/UNSW_NB15_testing-set.csv'
)
```

3. Save the script as `main.py` and run:

```bash
python main.py
```

---

## 🔧 Hyperparameters

You can tweak these in `main()`:

```python
SEQ_LEN   = 16   # sequence length
STRIDE    = 2    # stride for sliding window (smaller -> more sequences)
BATCH_SIZE = 64
EPOCHS     = 40  # early stopping will usually stop earlier
```

And in the model:

```python
latent_dim = 64  # size of latent vector
```

Adjusting these can trade off:

* Training time vs accuracy
* Reconstruction fidelity vs generalization
* Sensitivity of anomaly detection

---

## 💡 Ideas for Extensions

* Multi-class attack classification using `attack_cat` instead of binary `label`.
* Per-flow or per-host aggregation of sequence scores.
* Experiment with:

  * Different sequence lengths and strides
  * GRU instead of LSTM
  * Different mixing weights for MSE vs cosine similarity.
* Add model saving/loading (`model.save(...)`) and error threshold persistence.

---

If you want, I can also:

* Turn this into a clean **GitHub-ready repository layout** (with `src/`, `requirements.txt`, etc.).
* Generate a **shorter Kaggle description** version of this README.
