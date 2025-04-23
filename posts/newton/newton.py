
import torch

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has valuex None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        # your computation here: compute the vector of scores s
        return X @ self.w  # Matrix-vector multiplication for scores

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        return (self.score(X) > 0).float() 
    

class LogisticRegression(LinearModel):
    def loss(self, X, y):
        """
        Compute the empirical risk L(w) using the logistic loss function.
        
        L(w) = (1/n) * sum_{i=1}^n [ -y_i * log(σ(s_i)) - (1-y_i) * log(1-σ(s_i)) ]
        where s_i = <w, x_i> and σ(s_i) = 1 / (1 + exp(-s_i)).
        
        ARGUMENTS:
            X, torch.Tensor: Feature matrix with shape (n, p).
            y, torch.Tensor: Target vector with shape (n,).
        
        RETURNS:
            loss, torch.Tensor: The scalar loss value.
        """
        # Compute scores s = X @ self.w. This also initializes self.w if necessary.
        s = self.score(X)
        
        # Compute the probabilities using the sigmoid function.
        probs = torch.sigmoid(s)
        probs = torch.clamp(probs, min=1e-7, max=1-1e-7)  # <- Add this line

        
        # Compute logistic loss for each data point.
        loss_vals = -y * torch.log(probs) - (1 - y) * torch.log(1 - probs)
        
        # Return the average loss.
        return loss_vals.mean()
    
    def hessian(self, X, y):
        """
        Compute the Hessian matrix H(w) for logistic regression.
    
        H(w) = Xᵀ D X
        where D is a diagonal matrix with entries D_kk = σ(s_k)(1 - σ(s_k)),
        and s = X @ w

        ARGUMENTS:
            X, torch.Tensor: Feature matrix of shape (n, p)
            y, torch.Tensor: Target labels (not used in Hessian but included for consistency)

        RETURNS:
            H, torch.Tensor: Hessian matrix of shape (p, p)
        """
        s     = X @ self.w                   # (n,)
        sigma = torch.sigmoid(s)             # (n,)
        d     = sigma * (1 - sigma)          # (n,)
        Xd    = X * d.unsqueeze(1)           # (n, p) * (n,1) → (n, p)
        return (Xd.T @ X) / X.size(0)        # (p, p)      # Compute Hessian using matrix multiplication

        return H


    def grad(self, X, y):
        """
        Compute the gradient of the logistic loss function.
        
        Gradient = (1/n) * sum_{i=1}^n (σ(s_i) - y_i) * x_i
        where s_i = <w, x_i> and σ(s_i) = 1 / (1 + exp(-s_i)).
        
        ARGUMENTS:
            X, torch.Tensor: Feature matrix with shape (n, p).
            y, torch.Tensor: Target vector with shape (n,).
        
        RETURNS:
            grad, torch.Tensor: The gradient vector with shape (p,).
        """
        # Compute scores s = X @ self.w.
        s = self.score(X)
        
        # Compute probabilities using the sigmoid function.
        probs = torch.sigmoid(s)
        
        # Compute the error term (predicted probability minus actual label).
        error = probs - y
        
        # Compute the gradient: (1/n) * X.T @ error.
        n = X.size(0)
        gradient = (X.T @ error) / n
        
        return gradient


class GradientDescentOptimizer:
    def __init__(self, model):
        """
        Initialize the optimizer with the given model.
        The optimizer stores a reference to the model and keeps track of the previous weight vector.
        
        ARGUMENTS:
            model: An instance of a model (e.g., LogisticRegression) that has:
                   - a weight vector attribute `w`
                   - a method `grad(X, y)` to compute the gradient.
        """
        self.model = model
        self.prev_w = None  # Will hold the previous weight vector (w_{k-1}).

    def step(self, X, y, alpha, beta):
        """
        Perform one step of gradient descent with momentum.
        
        The update rule is:
            w_{k+1} = w_k - alpha * grad(L(w_k)) + beta * (w_k - w_{k-1})
            
        This method updates the model's weight vector `w` and also updates the stored previous weight.
        
        ARGUMENTS:
            X, torch.Tensor: Feature matrix.
            y, torch.Tensor: Target vector.
            alpha, float: Learning rate.
            beta, float: Momentum parameter.
        """
        if self.model.w is None:
            _ = self.model.score(X)  # Triggers lazy initialization

        # Get the current weight vector from the model.
        current_w = self.model.w

        # Compute the gradient of the loss at the current weight vector.
        grad_loss = self.model.grad(X, y)

        # For the first step, there is no previous weight so we initialize it.
        if self.prev_w is None:
            self.prev_w = current_w.clone()

        # Compute the momentum term: beta * (w_k - w_{k-1}).
        momentum_term = beta * (current_w - self.prev_w)

        # Update the weight vector using the gradient descent with momentum rule.
        new_w = current_w - alpha * grad_loss + momentum_term

        # Update the model's weights.
        self.model.w = new_w

        # Update the stored previous weight vector for the next iteration.
        self.prev_w = current_w.clone()

class NewtonOptimizer:
    def __init__(self, model):
        """
        Initialize the Newton optimizer.

        ARGUMENTS:
            model: An instance of LogisticRegression, with defined methods:
                   - model.w: current weights
                   - model.grad(X, y): returns ∇L(w)
                   - model.hessian(X, y): returns H(w)
        """
        self.model = model

    def step(self, X, y, alpha=1.0):
        """
        Perform one Newton update step:
            w <- w - α * H(w)^(-1) ∇L(w)

        ARGUMENTS:
            X, torch.Tensor: Feature matrix (n, p)
            y, torch.Tensor: Label vector (n,)
            alpha, float: Learning rate (default is 1.0)
        """
        grad = self.model.grad(X, y)             # ∇L(w)
        hess = self.model.hessian(X, y)          # H(w)

        # Add small value to the diagonal for numerical stability 
        eps = 1e-5
        hess_reg = hess + eps * torch.eye(hess.size(0))

        # Compute Newton step: H⁻¹ ∇L(w)
        step = torch.linalg.solve(hess_reg, grad)

        # Update weights
        self.model.w = self.model.w - alpha * step

class AdamOptimizer:
    def __init__(self, model, batch_size=32, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, w_0=None):
        """
        Initialize the Adam optimizer.

        ARGUMENTS:
            model: The logistic regression model to be optimized.
            batch_size: Size of the mini-batch for gradient updates.
            alpha: Learning rate.
            beta1: Decay rate for first moment estimate.
            beta2: Decay rate for second moment estimate.
            epsilon: Small constant to avoid division by zero.
            w_0: Optional initial weight vector.
        """
        self.model = model
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        if w_0 is not None:
            self.model.w = w_0.clone()
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0     # Timestep

    def step(self, X, y):
        """
        Perform one epoch of Adam optimization using mini-batch updates.

        ARGUMENTS:
            X: Feature matrix (torch.Tensor).
            y: Target vector (torch.Tensor).
        """
        if self.model.w is None:
            _ = self.model.score(X)  # Initialize weights

        n = X.size(0)
        indices = torch.randperm(n)  # Shuffle the data

        for i in range(0, n, self.batch_size):
            self.t += 1
            batch_idx = indices[i:i + self.batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            grad = self.model.grad(X_batch, y_batch)

            if self.m is None:
                self.m = torch.zeros_like(self.model.w)
            if self.v is None:
                self.v = torch.zeros_like(self.model.w)

            # Update moment estimates
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)

            # Parameter update
            self.model.w = self.model.w - self.alpha * m_hat / (torch.sqrt(v_hat) + self.epsilon)
