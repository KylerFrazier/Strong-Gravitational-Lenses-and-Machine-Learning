from .BaseBinaryThreshold import BaseBinaryThreshold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.validation import _deprecate_positional_args

class GradientBoostingBT(BaseBinaryThreshold, GradientBoostingClassifier):
    
    @_deprecate_positional_args
    def __init__(self, *, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0., min_impurity_split=None,
                 init=None, random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False, validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0, threshold=0.5):
        super().__init__(loss=loss, learning_rate=learning_rate,
                         n_estimators=n_estimators, subsample=subsample, 
                         criterion=criterion, 
                         min_samples_split=min_samples_split, 
                         min_samples_leaf=min_samples_leaf, 
                         min_weight_fraction_leaf=min_weight_fraction_leaf, 
                         max_depth=max_depth, 
                         min_impurity_decrease=min_impurity_decrease, 
                         min_impurity_split=min_impurity_split, init=init, 
                         random_state=random_state, max_features=max_features, 
                         verbose=verbose, max_leaf_nodes=max_leaf_nodes, 
                         warm_start=warm_start, 
                         validation_fraction=validation_fraction, 
                         n_iter_no_change=n_iter_no_change, tol=tol, 
                         ccp_alpha=ccp_alpha, threshold=threshold)
