import numpy as np

class Linear_regression():
    
    def __init__(self, learning_rate, epochs = 100000):
        self.learningRate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []
        pass
        

    def _compute_loss(self,y,y_pred): 
        return np.mean((y-y_pred)**2)
    
    def mean_square_erroer(true_values, predictions):
        return np.mean((true_values - predictions) ** 2)

    def _compute_gradients(self,x,y,y_pred):
        n = x.shape[0]
        error = y - y_pred
        grad_w = -(2/n) * np.dot(x.T, error)
        grad_b = -(2/n) * np.sum(error)
        return grad_w, grad_b
    
    def update_parameters(self, grad_w, grad_b): 
        self.weights -= self.learningRate * grad_w
        self.bias -= self.learningRate * grad_b

    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.epochs): 
            lin_model = np.matmul(self.weights, X.T) + self.bias
            y_pred = lin_model
            grad_w, grad_b = self._compute_gradients(X,y,y_pred)
            self.update_parameters(grad_w, grad_b)
            loss= self._compute_loss(y,y_pred)
            self.losses.append(loss)


        print(f"Predicted function: y = {self.weights[0]:.4f} * x + {self.bias:.4f}")


        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        lin_model = np.matmul(X, self.weights) + self.bias
        return lin_model