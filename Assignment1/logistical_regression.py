import numpy as np

class Logistical_regression():
    
    def __init__(self, learning_rate, epochs = 300):
        self.learningRate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []
        pass
        

    def sigmoid_function(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _compute_loss(self,y,y_pred): 
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean((y*np.log(y_pred) + (1-y) * np.log(1-y_pred) ))
    
    def accuracy(self, true_values, predictions):
        return np.mean(true_values == predictions)
        

    def _compute_gradients(self,x,y,y_pred):
        n = x.shape[0]
        error = y_pred - y
        grad_w = (1/n) * np.dot(x.T, error)
        grad_b = (1/n) * np.sum(error)
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

            y_pred = self.sigmoid_function(lin_model)
            grad_w, grad_b = self._compute_gradients(X,y,y_pred)
            self.update_parameters(grad_w, grad_b)

            loss= self._compute_loss(y,y_pred)
            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
            self.train_accuracies.append(self.accuracy(y, pred_to_class))
            self.losses.append(loss)

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
        X = np.asarray(X)
        lin_model = np.matmul(X, self.weights) + self.bias
        y_pred = self.sigmoid_function(lin_model)
        return [1 if _y > 0.5 else 0 for _y in y_pred]


    def predict_proba(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        X = np.asarray(X)
        lin_model = np.matmul(X, self.weights) + self.bias
        y_pred = self.sigmoid_function(lin_model)
        return y_pred