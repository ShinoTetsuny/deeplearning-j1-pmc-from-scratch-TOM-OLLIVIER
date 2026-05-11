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

activations = ['sigmoid', 'tanh', 'relu']
results = {}

print("=" * 70)
print("PHASE 6 : COMPARAISON ACTIVATIONS SUR MNIST")
print("=" * 70)
print(f"Architecture : 784 → Dense(128) → Dense(64) → Dense(10,Softmax)")
print(f"Epochs : 10 | Batch size : 64")
print("=" * 70)

for act in activations:
    print(f"\n--- {act.upper()} ---")

    model = keras.Sequential([
        layers.Dense(128, activation=act, input_shape=(784,)),
        layers.Dense(64,  activation=act),
        layers.Dense(10,  activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
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
    results[act] = {'history': history, 'test_acc': test_acc}

# Tableau récap
print("\n" + "=" * 70)
print("RÉSULTATS COMPARATIFS")
print("=" * 70)
print(f"{'Activation':12} | {'Val acc finale':>15} | {'Test acc':>10} | {'Convergence':>12}")
print("-" * 70)
for act, res in results.items():
    val_accs = res['history'].history['val_accuracy']
    epoch_90 = next((i+1 for i, v in enumerate(val_accs) if v >= 0.90), None)
    conv = f"epoch {epoch_90}" if epoch_90 else "> 10 epochs"
    print(f"{act:12} | {val_accs[-1]:>15.4f} | {res['test_acc']:>10.4f} | {conv:>12}")
print("=" * 70)

# Courbes val_accuracy superposées
colors = {'sigmoid': '#e74c3c', 'tanh': '#f39c12', 'relu': '#2ecc71'}
epochs = range(1, 11)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for act, res in results.items():
    h = res['history'].history
    axes[0].plot(epochs, h['val_accuracy'], label=act, linewidth=2, color=colors[act])
    axes[1].plot(epochs, h['val_loss'],     label=act, linewidth=2, color=colors[act])

axes[0].set_title("Val Accuracy — sigmoid / tanh / relu")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val Accuracy")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(list(epochs))

axes[1].set_title("Val Loss — sigmoid / tanh / relu")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Loss")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(list(epochs))

plt.suptitle("Phase 6 — Comparaison des fonctions d'activation sur MNIST", fontsize=13)
plt.tight_layout()
plt.savefig("phase6_activations.png", dpi=100, bbox_inches='tight')

# ======================================================================
# RÉSULTATS COMPARATIFS
# ======================================================================
# Activation   |  Val acc finale |   Test acc |  Convergence
# ----------------------------------------------------------------------
# sigmoid      |          0.9773 |     0.9727 |      epoch 1
# tanh         |          0.9802 |     0.9750 |      epoch 1
# relu         |          0.9777 |     0.9767 |      epoch 1
# ======================================================================

