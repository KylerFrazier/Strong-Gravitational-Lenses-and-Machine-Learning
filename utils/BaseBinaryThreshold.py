from abc import ABC
import inspect
import numpy as np

class BaseBinaryThreshold(ABC):
    
    def __init__(self, threshold = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
    
    def predict(self, X):
        # Predicts class for X with an offset threshold
        return (self.predict_proba(X)[:,1] >= self.threshold).astype(int)
    
    # def predict_proba(self, X):
    #     # Scales the probabilities so that threshold -> 0.5
    #     y = super().predict_proba(X)
    #     lt = y < self.threshold
    #     gt = y >= self.threshold
    #     y[lt] /= 2 * self.threshold
    #     if self.threshold != 1:
    #         y[gt] -= self.threshold
    #         y[gt] /= 2 * (1 - self.threshold)
    #         y[gt] += 0.5
    #     return y

    def error(self, x, y, soft = False, beta = 0.01):
        return 1 - self.fscore(x, y, soft, beta)

    def fscore(self, x, y, soft = False, beta = 1.0):
        p = self.precision(x, y, soft)
        r = self.recall(x, y, soft)
        b = beta**2
        return (1 + b) * (p * r) / (b * p + r) if b*p+r != 0 else 0

    def precision(self, x, y, soft = False):
        y_p = self.predict_proba(x).T[1] if soft else self.predict(x)
        fp = np.mean(y_p[y == 0])
        tp = np.mean(y_p[y == 1])
        return tp / (tp + fp) if tp+fp != 0 else 0
        
    def recall(self, x, y, soft = False):
        y_n = self.predict_proba(x).T[0] if soft else 1-self.predict(x)
        y_p = self.predict_proba(x).T[1] if soft else self.predict(x)
        fn = np.mean(y_n[y == 1])
        tp = np.mean(y_p[y == 1])
        return tp / (tp + fn) if tp+fn != 0 else 0
