import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

# Chargement MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 784).astype("float32") / 255.0

print("=" * 70)
print("PHASE 5 : PMC KERAS SUR MNIST")
print("=" * 70)
print(f"Train : {X_train.shape} | Test : {X_test.shape}")
print(f"Architecture : 784 → Dense(128,ReLU) → Dense(64,ReLU) → Dense(10,Softmax)")
print("=" * 70)

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64,  activation='relu'),
    layers.Dense(10,  activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print("=" * 70)

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

print("=" * 70)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy : {test_acc:.4f} ({test_acc:.2%})")
print(f"Test loss     : {test_loss:.4f}")
val_acc_finale = history.history['val_accuracy'][-1]
print(f"Val accuracy  : {val_acc_finale:.4f} ({val_acc_finale:.2%})")
objectif = "✓ ATTEINT" if val_acc_finale >= 0.97 else "✗ NON ATTEINT"
print(f"Objectif 97%  : {objectif}")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],     label='train', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='val',   linewidth=2)
axes[0].set_title("Accuracy par epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='train', linewidth=2)
axes[1].plot(history.history['val_loss'], label='val',   linewidth=2)
axes[1].set_title("Loss par epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f"MNIST — val_accuracy finale : {val_acc_finale:.2%}", fontsize=13)
plt.tight_layout()
plt.savefig("phase5_keras_mnist.png", dpi=100, bbox_inches='tight')
print("\nCourbe sauvegardée : phase5_keras_mnist.png")

# Comparaison numpy vs Keras
print("\n" + "=" * 70)
print("NUMPY vs KERAS — COMPARAISON")
print("=" * 70)
print(f"{'':30} {'Numpy (scratch)':>20} {'Keras':>15}")
print("-" * 70)
print(f"{'Dataset':30} {'XOR / Spirale':>20} {'MNIST':>15}")
print(f"{'Params manuels':30} {'Oui':>20} {'Non':>15}")
print(f"{'Backprop':30} {'Manuelle':>20} {'Auto (autodiff)':>15}")
print(f"{'Epochs':30} {'10 000':>20} {'5':>15}")
print(f"{'Accuracy':30} {'92.75%':>20} {f'{test_acc:.2%}':>15}")
print(f"{'Lignes de code':30} {'~80':>20} {'~15':>15}")
print("=" * 70)

# Conclusion
# ======================================================================
# Epoch 1/5
# 844/844 ━━━━━━━━━━━━━━━━━━━━ 5s 4ms/step - accuracy: 0.9151 - loss: 0.2907 - val_accuracy: 0.9605 - val_loss: 0.1437
# Epoch 2/5
# 844/844 ━━━━━━━━━━━━━━━━━━━━ 5s 6ms/step - accuracy: 0.9635 - loss: 0.1215 - val_accuracy: 0.9715 - val_loss: 0.0995
# Epoch 3/5
# 844/844 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9745 - loss: 0.0845 - val_accuracy: 0.9765 - val_loss: 0.0812
# Epoch 4/5
# 844/844 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - accuracy: 0.9803 - loss: 0.0636 - val_accuracy: 0.9762 - val_loss: 0.0768
# Epoch 5/5
# 844/844 ━━━━━━━━━━━━━━━━━━━━ 5s 6ms/step - accuracy: 0.9848 - loss: 0.0493 - val_accuracy: 0.9763 - val_loss: 0.0823
# ======================================================================
# Test accuracy : 0.9746 (97.46%)
# Test loss     : 0.0852
# Val accuracy  : 0.9763 (97.63%)
# Objectif 97%  : ✓ ATTEINT
# ======================================================================

# Courbe sauvegardée : phase5_keras_mnist.png

# ======================================================================
# NUMPY vs KERAS — COMPARAISON
# ======================================================================
#                                     Numpy (scratch)           Keras
# ----------------------------------------------------------------------
# Dataset                               XOR / Spirale           MNIST
# Params manuels                                  Oui             Non
# Backprop                                   Manuelle Auto (autodiff)
# Epochs                                       10 000               5
# Accuracy                                     92.75%          97.46%
# Lignes de code                                  ~80             ~15
# ======================================================================