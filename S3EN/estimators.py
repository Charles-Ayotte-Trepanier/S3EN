from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from S3EN.network import create_ensemble
from S3EN.helpers import perf_callback, adjust_data

class s3enEstimatorClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""
    def __init__(self,
                 feature_list,
                 validation_ratio=0,
                 patience=None,
                 nb_models_per_stack=20,
                 nb_variables_per_model=None,
                 nb_stack_blocks=10,
                 width=1,
                 depth=1,
                 epochs=100,
                 batch_size=128,
                 sample_weight=None,
                 nb_cores=None,
                 enable_gpu='no',
                 memory_growth='no'):
        """
        Called when initializing the classifier
        """

        self.feature_list = feature_list
        self.validation_ratio = validation_ratio
        self.patience = patience
        self.epochs = epochs
        self.batch_size = batch_size
        self.sample_weight = sample_weight
        self.nb_cores = nb_cores
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
        target_type = 'classification'
        mode = 'max'
        self.model, self.target_replicas = \
            create_ensemble(self.feature_list,
                            target_type,
                            self.nb_cores,
                            self.enable_gpu,
                            self.memory_growth,
                            self.nb_models_per_stack,
                            self.nb_variables_per_model,
                            self.nb_stack_blocks,
                            self.width,
                            self.depth)

        if self.validation_ratio > 0:
            X_train, X_val, y_train, y_val = \
                train_test_split(X,
                                 y,
                                 test_size=self.validation_ratio)
            X_train_adj, y_train_adj = adjust_data(X_train,
                                                   y_train,
                                                   self.feature_list,
                                                   self.target_replicas)
            X_val_adj, y_val_adj = adjust_data(X_val,
                                               y_val,
                                               self.feature_list,
                                               self.target_replicas)

            early_stop = EarlyStopping(patience=self.patience,
                                       monitor='validation',
                                       mode=mode)

            callbacks = [perf_callback((X_val_adj, y_val_adj), target_type),
                         early_stop]
        else:
            callbacks = None
            X_train_adj, y_train_adj = adjust_data(X,
                                                   y,
                                                   self.feature_list,
                                                   self.target_replicas)

        self.model.fit(X_train_adj,
                       y_train_adj,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       sample_weight=self.sample_weight,
                       callbacks=callbacks)
        self._fitted = True

        return self


    def predict_proba(self, X, y=None):
        if self._fitted is not True:
            raise RuntimeError(
                "You must train classifer before predicting data!")
        X_adjusted = [X[col['feat_nm']].values.reshape(-1, 1) for col in
             self.feature_list]
        return self.model.predict(X_adjusted)[:, 0]

    def predict(self, X, y=None):
        return (self.predict_proba(X) > 0.5).astype(int)


class s3enEstimatorRegressor(BaseEstimator, RegressorMixin):
    """An example of regressor"""
    def __init__(self,
                 feature_list,
                 validation_ratio=0,
                 patience=None,
                 nb_models_per_stack=20,
                 nb_variables_per_model=None,
                 nb_stack_blocks=10,
                 width=1,
                 depth=1,
                 epochs=100,
                 batch_size=128,
                 sample_weight=None,
                 nb_cores=None,
                 enable_gpu='no',
                 memory_growth='no'):
        """
        Called when initializing the classifier
        """

        self.feature_list = feature_list
        self.validation_ratio = validation_ratio
        self.patience = patience
        self.epochs = epochs
        self.batch_size = batch_size
        self.sample_weight = sample_weight
        self.nb_cores = nb_cores
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
        target_type = 'regression'
        mode = 'min'
        self.model, self.target_replicas = \
            create_ensemble(self.feature_list,
                            target_type,
                            self.nb_cores,
                            self.enable_gpu,
                            self.memory_growth,
                            self.nb_models_per_stack,
                            self.nb_variables_per_model,
                            self.nb_stack_blocks,
                            self.width,
                            self.depth)

        if self.validation_ratio > 0:
            X_train, X_val, y_train, y_val = \
                train_test_split(X,
                                 y,
                                 test_size=self.validation_ratio)
            X_train_adj, y_train_adj = adjust_data(X_train,
                                                   y_train,
                                                   self.feature_list,
                                                   self.target_replicas)
            X_val_adj, y_val_adj = adjust_data(X_val,
                                               y_val,
                                               self.feature_list,
                                               self.target_replicas)

            early_stop = EarlyStopping(patience=self.patience,
                                       monitor='validation',
                                       mode=mode)

            callbacks = [perf_callback((X_val_adj, y_val_adj), target_type),
                         early_stop]
        else:
            callbacks = None
            X_train_adj, y_train_adj = adjust_data(X,
                                                   y,
                                                   self.feature_list,
                                                   self.target_replicas)

        self.model.fit(X_train_adj,
                       y_train_adj,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        sample_weight=self.sample_weight,
                        callbacks=callbacks
                        )
        self._fitted = True

        return self


    def predict(self, X, y=None):
        if self._fitted is not True:
            raise RuntimeError(
                "You must train classifer before predicting data!")
        X_adjusted = [X[col['feat_nm']].values.reshape(-1, 1) for col in
             self.feature_list]
        return self.model.predict(X_adjusted)[:, 0]