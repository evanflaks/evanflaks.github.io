import torch

class KernelLogisticRegression:
    def __init__(self, kernel_fn, lam=0.01, gamma=None):
        """
        Initialize the model.

        Arguments:
            kernel_fn: callable, kernel function taking X1, X2, (and optionally gamma) → Tensor
            lam: float, L1 regularization strength
            gamma: optional hyperparameter passed to kernel function (e.g. for RBF)
        """
        self.kernel_fn = kernel_fn
        self.lam = lam
        self.gamma = gamma
        self.X_train = None
        self.a = None  # coefficient vector in dual space

    def _compute_kernel(self, X1, X2):
        if self.gamma is not None:
            return self.kernel_fn(X1, X2, self.gamma)
        else:
            return self.kernel_fn(X1, X2)

    def score(self, X):
        """
        Compute scores: s = K(X, X_train)^T @ a

        Arguments:
            X: torch.Tensor of shape (m, p) — test features

        Returns:
            scores: torch.Tensor of shape (m,)
        """
        K = self._compute_kernel(X, self.X_train)  # shape (m, n)
        return K @ self.a  # shape (m,)

    def predict(self, X):
        """
        Predict binary labels using sigmoid threshold at 0.5.

        Arguments:
            X: torch.Tensor of shape (m, p)

        Returns:
            y_hat: torch.Tensor of shape (m,) with values in {0.0, 1.0}
        """
        return (torch.sigmoid(self.score(X)) > 0.5).float()

    def loss(self, K, y, a):
        """
        Logistic loss with L1 regularization.

        Arguments:
            K: torch.Tensor of shape (m, n), kernel matrix K(X, X_train)
            y: torch.Tensor of shape (m,)
            a: torch.Tensor of shape (n,)

        Returns:
            scalar loss value
        """
        s = K @ a  # scores
        probs = torch.sigmoid(s)
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)

        # Binary cross-entropy loss
        log_loss = -y * torch.log(probs) - (1 - y) * torch.log(1 - probs)
        loss = log_loss.mean()

        # L1 regularization
        reg = self.lam * torch.norm(a, p=1)

        return loss + reg

    def grad(self, K, y, a):
        """
        Compute gradient of logistic + L1 loss.

        Arguments:
            K: kernel matrix of shape (m, n)
            y: labels of shape (m,)
            a: weights of shape (n,)

        Returns:
            gradient: tensor of shape (n,)
        """
        s = K @ a
        probs = torch.sigmoid(s)
        error = probs - y  # shape (m,)

        grad_log_loss = (K.T @ error) / K.size(0)  # shape (n,)
        grad_l1 = self.lam * torch.sign(a)  # subgradient of L1 norm

        return grad_log_loss + grad_l1

    def fit(self, X, y, lr=0.1, epochs=1000):
        """
        Fit the model using gradient descent.

        Arguments:
            X: training data of shape (n, p)
            y: binary labels of shape (n,)
            lr: learning rate
            epochs: number of iterations
        """
        self.X_train = X
        n = X.shape[0]
        self.a = torch.zeros(n, requires_grad=False)

        K = self._compute_kernel(X, X)  # shape (n, n)

        for epoch in range(epochs):
            grad = self.grad(K, y, self.a)
            self.a = self.a - lr * grad

            # Optional: Print loss every 100 iterations
            if epoch % 100 == 0:
                current_loss = self.loss(K, y, self.a).item()
                print(f"Epoch {epoch}: Loss = {current_loss:.4f}")