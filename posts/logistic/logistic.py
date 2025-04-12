
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
