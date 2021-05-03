from .BaseBinaryThreshold import BaseBinaryThreshold
from sklearn.ensemble import StackingClassifier
from sklearn.utils.validation import _deprecate_positional_args

class EnsembleBT(BaseBinaryThreshold, StackingClassifier):
    
    @_deprecate_positional_args
    def __init__(self, estimators, final_estimator=None, *, cv=None,
                 stack_method='auto', n_jobs=None, passthrough=False,
                 verbose=0, threshold=0.5):
        super().__init__(estimators=estimators,
                         final_estimator=final_estimator,
                         cv=cv,
                         stack_method=stack_method,
                         n_jobs=n_jobs,
                         passthrough=passthrough,
                         verbose=verbose)
        self.threshold = threshold