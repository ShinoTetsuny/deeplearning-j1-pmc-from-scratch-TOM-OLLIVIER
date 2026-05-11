import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 784).astype("float32") / 255.0

learning_rates = [1e-7, 1e-3, 1.0]
labels         = ['1e-7 (trop petit)', '1e-3 (sweet spot)', '1.0  (trop grand)']
colors         = ['#e74c3c', '#2ecc71', '#f39c12']
results        = {}

print("=" * 70)
print("PHASE 7 : VARIATION DU LEARNING RATE SUR MNIST")
print("=" * 70)
print(f"Architecture : 784 → Dense(128,ReLU) → Dense(64,ReLU) → Dense(10,Softmax)")
print(f"Optimizer : Adam | Epochs : 10 | Batch size : 64")
print("=" * 70)

for lr, label in zip(learning_rates, labels):
    print(f"\n--- LR = {lr:.0e} ---")

    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64,  activation='relu'),
        layers.Dense(10,  activation='softmax'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    results[lr] = {'history': history, 'test_acc': test_acc, 'label': label}

# Tableau récap
print("\n" + "=" * 70)
print("RÉSULTATS COMPARATIFS")
print("=" * 70)
print(f"{'Learning rate':20} | {'Val acc finale':>15} | {'Test acc':>10} | {'Diagnostic':>20}")
print("-" * 70)
diagnostics = {1e-7: 'pas de convergence', 1e-3: 'sweet spot ✓', 1.0: 'explosion / chaos'}
for lr, res in results.items():
    val_acc = res['history'].history['val_accuracy'][-1]
    print(f"{str(lr):20} | {val_acc:>15.4f} | {res['test_acc']:>10.4f} | {diagnostics[lr]:>20}")
print("=" * 70)

# Courbes superposées
epochs = range(1, 11)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for lr, label, color in zip(learning_rates, labels, colors):
    h = results[lr]['history'].history
    axes[0].plot(epochs, h['val_accuracy'], label=label, linewidth=2, color=color)
    axes[1].plot(epochs, h['val_loss'],     label=label, linewidth=2, color=color)

axes[0].set_title("Val Accuracy par learning rate")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val Accuracy")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(list(epochs))

axes[1].set_title("Val Loss par learning rate")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Loss")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(list(epochs))

plt.suptitle("Phase 7 — Variation du learning rate sur MNIST", fontsize=13)
plt.tight_layout()
plt.savefig("phase7_learning_rate.png", dpi=100, bbox_inches='tight')

# ======================================================================
# RÉSULTATS COMPARATIFS
# ======================================================================
# Learning rate        |  Val acc finale |   Test acc |           Diagnostic
# ----------------------------------------------------------------------
# 1e-07                |          0.1577 |     0.1599 |   pas de convergence
# 0.001                |          0.9788 |     0.9754 |         sweet spot ✓
# 1.0                  |          0.0960 |     0.1010 |    explosion / chaos
# ======================================================================