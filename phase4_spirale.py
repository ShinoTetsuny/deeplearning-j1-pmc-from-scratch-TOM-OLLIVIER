import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
n = 200
t = np.linspace(0, 4*np.pi, n)
X0 = np.c_[t*np.cos(t), t*np.sin(t)] + np.random.randn(n, 2) * 0.5
X1 = np.c_[t*np.cos(t+np.pi), t*np.sin(t+np.pi)] + np.random.randn(n, 2) * 0.5
X = np.vstack([X0, X1])
y = np.array([0]*n + [1]*n)
X = (X - X.mean(axis=0)) / X.std(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def compute_loss_bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Initialisation He : std = sqrt(2 / n_entrées)
np.random.seed(42)
W1 = np.random.randn(2, 64) * np.sqrt(2 / 2)
b1 = np.zeros(64)
W2 = np.random.randn(64, 64) * np.sqrt(2 / 64)
b2 = np.zeros(64)
W3 = np.random.randn(64, 1) * np.sqrt(2 / 64)
b3 = np.zeros(1)

learning_rate = 0.05
n_epochs = 10000
losses = []

print("=" * 70)
print("PHASE 4 : SPIRALE 2D — RÉSEAU 2-64-64-1")
print("=" * 70)
print(f"Samples : {len(y)} | Architecture : 2 → 64 ReLU → 64 ReLU → 1 Sigmoid")
print(f"Learning rate : {learning_rate} | Initialisation : He")
print("=" * 70)

for epoch in range(n_epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1;   a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2;  a2 = relu(z2)
    z3 = np.dot(a2, W3) + b3;  y_pred = sigmoid(z3).flatten()

    loss = compute_loss_bce(y, y_pred)
    losses.append(loss)

    # Backprop couche 3
    e3 = y_pred - y
    dW3 = np.dot(a2.T, e3.reshape(-1, 1)) / len(y)
    db3 = np.mean(e3)

    # Backprop couche 2
    e2 = np.dot(e3.reshape(-1, 1), W3.T) * relu_grad(z2)
    dW2 = np.dot(a1.T, e2) / len(y)
    db2_grad = np.mean(e2, axis=0)

    # Backprop couche 1
    e1 = np.dot(e2, W2.T) * relu_grad(z1)
    dW1 = np.dot(X.T, e1) / len(y)
    db1_grad = np.mean(e1, axis=0)

    W3 -= learning_rate * dW3;  b3 -= learning_rate * db3
    W2 -= learning_rate * dW2;  b2 -= learning_rate * db2_grad
    W1 -= learning_rate * dW1;  b1 -= learning_rate * db1_grad

    if epoch % 2000 == 0:
        acc = np.mean((y_pred > 0.5) == y)
        print(f"Epoch {epoch:5d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

print("=" * 70)
acc = np.mean((y_pred > 0.5) == y)
print(f"Loss finale : {losses[-1]:.4f}")
print(f"Accuracy finale : {acc:.2%}")
print("=" * 70)

# Frontière de décision
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 300),
                     np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
a1g = relu(np.dot(grid, W1) + b1)
a2g = relu(np.dot(a1g, W2) + b2)
zg  = sigmoid(np.dot(a2g, W3) + b3).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].contourf(xx, yy, zg, alpha=0.4, cmap='RdBu')
axes[0].scatter(X[:,0], X[:,1], c=y, s=10, cmap='RdBu', edgecolors='none', alpha=0.8)
axes[0].set_title(f"Frontière de décision — Accuracy: {acc:.2%}")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")

axes[1].plot(losses, linewidth=2, color='#1f77b4')
axes[1].set_title("Convergence — Spirale 2D")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss BCE")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("phase4_spirale.png", dpi=100, bbox_inches='tight')
