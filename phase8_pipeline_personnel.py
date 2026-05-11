import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow import keras
from tensorflow.keras import layers

# ── 1. CHARGEMENT ─────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target

print("=" * 70)
print("PHASE 8 : PIPELINE COMPLET — BREAST CANCER")
print("=" * 70)
print(f"Samples   : {X.shape[0]} | Features : {X.shape[1]}")
print(f"Classes   : malignant={( y==0).sum()} | benign={(y==1).sum()}")
print("=" * 70)

# ── 2. SPLIT + NORMALISATION ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit sur train UNIQUEMENT
X_test_s  = scaler.transform(X_test)

print(f"Train : {X_train_s.shape} | Test : {X_test_s.shape}")
print("=" * 70)

# ── 3. NUMPY SCRATCH : 30-64-32-1 ─────────────────────────────────────
print("\n[1/2] NUMPY SCRATCH — Architecture 30-64-32-1")
print("-" * 70)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x): return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(float)

def bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

np.random.seed(42)
W1 = np.random.randn(30, 64) * np.sqrt(2 / 30)
b1 = np.zeros(64)
W2 = np.random.randn(64, 32) * np.sqrt(2 / 64)
b2 = np.zeros(32)
W3 = np.random.randn(32, 1)  * np.sqrt(2 / 32)
b3 = np.zeros(1)

lr = 0.01
n_epochs = 1000
numpy_losses = []

for epoch in range(n_epochs):
    z1 = X_train_s @ W1 + b1;  a1 = relu(z1)
    z2 = a1 @ W2 + b2;         a2 = relu(z2)
    z3 = a2 @ W3 + b3;         y_pred = sigmoid(z3).flatten()

    loss = bce(y_train, y_pred)
    numpy_losses.append(loss)

    e3 = y_pred - y_train
    dW3 = a2.T @ e3.reshape(-1, 1) / len(y_train)
    db3 = np.mean(e3)

    e2 = e3.reshape(-1, 1) @ W3.T * relu_grad(z2)
    dW2 = a1.T @ e2 / len(y_train)
    db2_g = np.mean(e2, axis=0)

    e1 = e2 @ W2.T * relu_grad(z1)
    dW1 = X_train_s.T @ e1 / len(y_train)
    db1_g = np.mean(e1, axis=0)

    W3 -= lr * dW3;  b3 -= lr * db3
    W2 -= lr * dW2;  b2 -= lr * db2_g
    W1 -= lr * dW1;  b1 -= lr * db1_g

    if epoch % 200 == 0:
        acc = np.mean((y_pred > 0.5) == y_train)
        print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | Train acc: {acc:.2%}")

z1t = X_test_s @ W1 + b1;  a1t = relu(z1t)
z2t = a1t @ W2 + b2;       a2t = relu(z2t)
yp_test_np = sigmoid(a2t @ W3 + b3).flatten()
numpy_acc  = np.mean((yp_test_np > 0.5) == y_test)

print(f"\n  → Test accuracy numpy : {numpy_acc:.4f} ({numpy_acc:.2%})")

# ── 4. KERAS : 30-64-32-1 ─────────────────────────────────────────────
print("\n[2/2] KERAS — Architecture 30-64-32-1")
print("-" * 70)

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(30,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1,  activation='sigmoid'),
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_s, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

_, keras_acc = model.evaluate(X_test_s, y_test, verbose=0)
print(f"\n  → Test accuracy Keras : {keras_acc:.4f} ({keras_acc:.2%})")

# ── 5. CLASSIFICATION REPORT KERAS ────────────────────────────────────
yp_keras = (model.predict(X_test_s, verbose=0).flatten() > 0.5).astype(int)
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT — KERAS")
print("=" * 70)
print(classification_report(y_test, yp_keras, target_names=data.target_names))

# ── 6. COMPARAISON FINALE ─────────────────────────────────────────────
print("=" * 70)
print("COMPARAISON NUMPY vs KERAS")
print("=" * 70)
print(f"{'':25} {'Numpy scratch':>18} {'Keras':>15}")
print("-" * 70)
print(f"{'Architecture':25} {'30-64-32-1':>18} {'30-64-32-1':>15}")
print(f"{'Epochs':25} {'1 000':>18} {'50':>15}")
print(f"{'Optimizer':25} {'SGD manuel':>18} {'Adam':>15}")
print(f"{'Test accuracy':25} {f'{numpy_acc:.2%}':>18} {f'{keras_acc:.2%}':>15}")
print(f"{'Data leakage évité':25} {'Oui':>18} {'Oui':>15}")
print("=" * 70)

# ── 7. VISUALISATION ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(numpy_losses, linewidth=2, color='#e74c3c', label='numpy')
axes[0].set_title("Loss Numpy — 1000 epochs")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
axes[0].grid(True, alpha=0.3); axes[0].legend()

axes[1].plot(history.history['accuracy'],     label='train', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='val',   linewidth=2)
axes[1].set_title("Accuracy Keras — 50 epochs")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

cm = confusion_matrix(y_test, yp_keras)
im = axes[2].imshow(cm, cmap='Blues')
axes[2].set_xticks([0, 1]); axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(data.target_names); axes[2].set_yticklabels(data.target_names)
axes[2].set_xlabel("Prédit"); axes[2].set_ylabel("Réel")
axes[2].set_title("Matrice de confusion — Keras")
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, cm[i, j], ha='center', va='center', fontsize=14,
                     color='white' if cm[i, j] > cm.max()/2 else 'black')

plt.suptitle(f"Phase 8 — Breast Cancer | Numpy: {numpy_acc:.2%} | Keras: {keras_acc:.2%}", fontsize=13)
plt.tight_layout()
plt.savefig("phase8_pipeline_personnel.png", dpi=100, bbox_inches='tight')

# ======================================================================
# CLASSIFICATION REPORT — KERAS
# ======================================================================
#               precision    recall  f1-score   support

#    malignant       0.93      0.98      0.95        42
#       benign       0.99      0.96      0.97        72

#     accuracy                           0.96       114
#    macro avg       0.96      0.97      0.96       114
# weighted avg       0.97      0.96      0.97       114

# ======================================================================
# COMPARAISON NUMPY vs KERAS
# ======================================================================
#                                Numpy scratch           Keras
# ----------------------------------------------------------------------
# Architecture                      30-64-32-1      30-64-32-1
# Epochs                                 1 000              50
# Optimizer                         SGD manuel            Adam
# Test accuracy                         95.61%          96.49%
# Data leakage évité                       Oui             Oui
# ======================================================================
