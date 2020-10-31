from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from S3EN.network import create_ensemble
from S3EN.helpers import duplicate

class s3enEstimatorClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""
    def __init__(self,
                 feature_list,
                 nb_models_per_stack=20,
                 nb_variables_per_model=None,
                 nb_stack_blocks=10,
                 width=1,
                 depth=1,
                 epochs=100,
                 batch_size=128,
                 sample_weight=None,
                 enable_gpu='no',
                 memory_growth='no'):
        """
        Called when initializing the classifier
        """

        self.feature_list = feature_list
        self.epochs = epochs
        self.batch_size = batch_size
        self.sample_weight = sample_weight
        self.enable_gpu = enable_gpu
        self.memory_growth = memory_growth
        self.nb_models_per_stack = nb_models_per_stack
        self.nb_variables_per_model = nb_variables_per_model
        self.nb_stack_blocks = nb_stack_blocks
        self.width = width
        self.depth = depth

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        self._model, self._target_replicas = \
            create_ensemble(self.feature_list,
                            'classification',
                            self.enable_gpu,
                            self.memory_growth,
                            self.nb_models_per_stack,
                            self.nb_variables_per_model,
                            self.nb_stack_blocks,
                            self.width,
                            self.depth)
        X_adjusted = [X[col['feat_nm']].values.reshape(-1, 1) for col in
                      self.feature_list]
        y_adjusted = duplicate(y, self._target_replicas)
        self._model.fit(X_adjusted,
                        y_adjusted,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        sample_weight=self.sample_weight
                        )
        self._fitted = True

        return self


    def predict_proba(self, X, y=None):
        if self._fitted is not True:
            raise RuntimeError(
                "You must train classifer before predicting data!")
        X_adjusted = [X[col['feat_nm']].values.reshape(-1, 1) for col in
             self.feature_list]
        return self._model.predict(X_adjusted)[:, 0]

    def predict(self, X, y=None):
        return (self.predict_proba(X) > 0.5).astype(int)


class s3enEstimatorRegressor(BaseEstimator, RegressorMixin):
    """An example of regressor"""
    def __init__(self,
                 feature_list,
                 nb_models_per_stack=20,
                 nb_variables_per_model=None,
                 nb_stack_blocks=10,
                 width=1,
                 depth=1,
                 epochs=100,
                 batch_size=128,
                 sample_weight=None,
                 enable_gpu='no',
                 memory_growth='no'):
        """
        Called when initializing the classifier
        """

        self.feature_list = feature_list
        self.epochs = epochs
        self.batch_size = batch_size
        self.sample_weight = sample_weight
        self.enable_gpu = enable_gpu
        self.memory_growth = memory_growth
        self.nb_models_per_stack = nb_models_per_stack
        self.nb_variables_per_model = nb_variables_per_model
        self.nb_stack_blocks = nb_stack_blocks
        self.width = width
        self.depth = depth

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        self._model, self._target_replicas = \
            create_ensemble(self.feature_list,
                            'regression',
                            self.enable_gpu,
                            self.memory_growth,
                            self.nb_models_per_stack,
                            self.nb_variables_per_model,
                            self.nb_stack_blocks,
                            self.width,
                            self.depth)
        X_adjusted = [X[col['feat_nm']].values.reshape(-1, 1) for col in
                      self.feature_list]
        y_adjusted = duplicate(y, self._target_replicas)
        self._model.fit(X_adjusted,
                        y_adjusted,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        sample_weight=self.sample_weight
                        )
        self._fitted = True

        return self


    def predict(self, X, y=None):
        if self._fitted is not True:
            raise RuntimeError(
                "You must train classifer before predicting data!")
        X_adjusted = [X[col['feat_nm']].values.reshape(-1, 1) for col in
             self.feature_list]
        return self._model.predict(X_adjusted)[:, 0]