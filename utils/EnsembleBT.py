from .BaseBinaryThreshold import BaseBinaryThreshold
from sklearn.ensemble import StackingClassifier
from sklearn.utils.validation import _deprecate_positional_args

# imports for the new fit function
from copy import deepcopy
import numpy as np
from joblib import Parallel
import scipy.sparse as sparse
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import column_or_1d
from sklearn.utils.fixes import delayed



class EnsembleBT(BaseBinaryThreshold, StackingClassifier):
    
    @_deprecate_positional_args
    def __init__(self, estimators, final_estimator=None, *, cv=None,
                 stack_method='auto', n_jobs=None, passthrough=False,
                 verbose=0, threshold=0.5):
        super().__init__(estimators=estimators, final_estimator=final_estimator,
                         cv=cv, stack_method=stack_method, n_jobs=n_jobs,
                         passthrough=passthrough, verbose=verbose, 
                         threshold=threshold)
    


    def fit(self, X, y, sample_weight=None, sample_weight_meta=None):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,) or default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

            .. versionchanged:: 0.23
               when not None, `sample_weight` is passed to all underlying
               estimators

        Returns
        -------
        self : object
        """

        check_classification_targets(y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_

        # all_estimators contains all estimators, the one to be fitted and the
        # 'drop' string.
        names, all_estimators = self._validate_estimators()
        self._validate_final_estimator()

        stack_method = [self.stack_method] * len(all_estimators)

        # Fit the base estimators on the whole training data. Those
        # base estimators will be used in transform, predict, and
        # predict_proba. They are exposed publicly.
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(clone(est), X, y, sample_weight)
            for est in all_estimators if est != 'drop'
        )

        self.named_estimators_ = Bunch()
        est_fitted_idx = 0
        for name_est, org_est in zip(names, all_estimators):
            if org_est != 'drop':
                self.named_estimators_[name_est] = self.estimators_[
                    est_fitted_idx]
                est_fitted_idx += 1
            else:
                self.named_estimators_[name_est] = 'drop'

        # To train the meta-classifier using the most data as possible, we use
        # a cross-validation to obtain the output of the stacked estimators.

        # To ensure that the data provided to each estimator are the same, we
        # need to set the random state of the cv if there is one and we need to
        # take a copy.
        cv = check_cv(self.cv, y=y, classifier=is_classifier(self))
        if hasattr(cv, 'random_state') and cv.random_state is None:
            cv.random_state = np.random.RandomState()

        self.stack_method_ = [
            self._method_name(name, est, meth)
            for name, est, meth in zip(names, all_estimators, stack_method)
        ]
        fit_params = ({"sample_weight": sample_weight}
                      if sample_weight is not None
                      else None)
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(cross_val_predict)(clone(est), X, y, cv=deepcopy(cv),
                                       method=meth, n_jobs=self.n_jobs,
                                       fit_params=fit_params,
                                       verbose=self.verbose)
            for est, meth in zip(all_estimators, self.stack_method_)
            if est != 'drop'
        )

        # Only not None or not 'drop' estimators will be used in transform.
        # Remove the None from the method as well.
        self.stack_method_ = [
            meth for (meth, est) in zip(self.stack_method_, all_estimators)
            if est != 'drop'
        ]

        X_meta = self._concatenate_predictions(X, predictions)
        _fit_single_estimator(self.final_estimator_, X_meta, y,
                              sample_weight=sample_weight_meta)

        return self