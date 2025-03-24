import torch

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

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

class Perceptron(LinearModel):

    def loss(self, X, y):
        """
        Compute the misclassification rate. A point i is classified correctly if it holds that s_i*y_i_ > 0, where y_i_ is the *modified label* that has values in {-1, 1} (rather than {0, 1}). 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        
        HINT: In order to use the math formulas in the lecture, you are going to need to construct a modified set of targets and predictions that have entries in {-1, 1} -- otherwise none of the formulas will work right! An easy to to make this conversion is: 
        
        y_ = 2*y - 1
        """

        # replace with your implementation
        y_ = 2 * y - 1  # Convert labels from {0,1} to {-1,1}
        return (1.0 * (self.score(X) * y_ <= 0)).mean()  # Compute misclassification rate

    def grad(self, X, y):
        """
        Compute the perceptron gradient update for a single data point X and label y.
    
        ARGUMENTS:
        X, torch.Tensor: single feature vector, shape (p,)
        y, torch.Tensor: single label, scalar in {0.0, 1.0}
    
        RETURNS:
        update, torch.Tensor: vector of same shape as X
        """
        X = X.flatten()               # Ensure it's a 1D tensor
        y_ = 2 * y - 1                  # Convert label to {-1, 1}
        s = torch.dot(self.w, X)       # Compute score = <w, x>
        if s * y_ < 0:
            return -y_ * X             # Return update if misclassified
        else:
            return torch.zeros_like(X)  # Otherwise, return zero vector


class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """
        self.model.loss(X, y)                      # Call loss (could be used for logging/debugging)
        self.model.w = self.model.w - self.model.grad(X, y)  # Perform perceptron update