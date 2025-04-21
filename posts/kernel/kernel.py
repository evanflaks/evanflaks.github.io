import torch

class KernelLogisticRegression:
    def __init__(self, kernel_fn, lam=0.01, gamma=None):
        self.kernel_fn = kernel_fn
        self.lam = lam
        self.gamma = gamma
        self.X_train = None
        self.a = None

    def _compute_kernel(self, X1, X2):
        return self.kernel_fn(X1, X2, self.gamma) if self.gamma else self.kernel_fn(X1, X2)

    def fit(self, X, y, lr=0.1, epochs=1000):
        self.X_train = X
        y = y.float()
        n = X.shape[0]
        self.a = torch.zeros(n, dtype=torch.float32)
        K = self._compute_kernel(X, X)

        for epoch in range(epochs):
            s = K @ self.a
            probs = torch.sigmoid(s)
            error = probs - y
            grad = (K.T @ error) / n

            # Proximal step (soft-thresholding)
            a_temp = self.a - lr * grad
            self.a = torch.sign(a_temp) * torch.clamp(torch.abs(a_temp) - lr * self.lam, min=0.0)

    def score(self, X):
        K = self._compute_kernel(X, self.X_train)
        return K @ self.a

    def predict(self, X):
        return (torch.sigmoid(self.score(X)) > 0.5).float()
