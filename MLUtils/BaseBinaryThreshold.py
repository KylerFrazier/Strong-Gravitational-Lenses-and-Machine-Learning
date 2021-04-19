from abc import ABC

class BaseBinaryThreshold(ABC):
    
    def __init__(self, threshold = 0.5, *args, **kwargs):
        self.threshold = threshold
        super().__init__(*args, **kwargs)
    
    def predict(self, X):
        # Predicts class for X with an offset threshold
        return (self.predict_proba(X)[:,1] >= self.threshold).astype(int)
