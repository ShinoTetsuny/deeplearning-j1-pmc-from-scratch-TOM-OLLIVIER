import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X = np.array([[0.2, 0.1], [0.8, 0.9], [0.3, 0.7], [0.9, 0.2]])
y = np.array([0, 1, 1, 0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

np.random.seed(42)
w = np.random.randn(2) * 0.01
b = 0.0

learning_rate = 0.1
n_epochs = 50
losses = []

print("=" * 70)
print("PHASE 2 : DESCENTE DE GRADIENT MANUELLE")
print("=" * 70)
print(f"Learning rate : {learning_rate}")
print(f"Poids initiaux : {w.round(4)}, biais : {b:.4f}")
print("=" * 70)

for epoch in range(n_epochs):
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    loss = compute_loss(y, y_pred)
    losses.append(loss)
    
    error = y_pred - y
    dw = np.dot(X.T, error) / len(y)
    db = np.mean(error)
    
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | w: {w.round(3)} | b: {b:.3f}")

print("=" * 70)
print(f"Loss initiale : {losses[0]:.4f}")
print(f"Loss finale : {losses[-1]:.4f}")
print(f"Amélioration : {losses[0] - losses[-1]:.4f}")
print(f"Poids finaux : {w.round(4)}, biais : {b:.4f}")
print("=" * 70)

plt.figure(figsize=(8, 4))
plt.plot(losses, linewidth=2, color='#1f77b4')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss BCE", fontsize=12)
plt.title("Convergence du neurone unique - Descente de gradient", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("phase2_loss_curve.png", dpi=100, bbox_inches='tight')
print(f"\nCourbe sauvegardée : phase2_loss_curve.png")

print("\n" + "=" * 70)
print("PRÉDICTIONS FINALES")
print("=" * 70)
z_final = np.dot(X, w) + b
y_pred_final = sigmoid(z_final)

for i, (x, y_true, y_p) in enumerate(zip(X, y, y_pred_final)):
    print(f"Exemple {i}: input={x}, label={y_true}, pred={y_p:.4f}")