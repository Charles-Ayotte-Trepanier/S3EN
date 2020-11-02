from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from S3EN.network import S3enNetwork
from S3EN.helpers import perf_callback, adjust_data

def s3enEstimator(feature_list,
                  target_type='classification',
                  validation_ratio=0,
                  patience=None,
                  nb_models_per_stack=20,
                  nb_variables_per_model=None,
                  nb_stack_blocks=10,
                  width=1,
                  depth=1,
                  epochs=100,
                  batch_size=128,
                  activation='elu',
                  batch_norm='no',
                  dropout_rate=0,
                  sample_weight=None,
                  nb_cores=None,
                  enable_gpu='no',
                  memory_growth='no'):

    if target_type == 'classification':
        parent_class = ClassifierMixin
    elif target_type == 'regression':
        parent_class = RegressorMixin

    class s3enEstimatorFlex(BaseEstimator, parent_class):
        """An example of classifier"""

        def __init__(self,
                     feature_list,
                     target_type,
                     validation_ratio,
                     patience,
                     nb_models_per_stack,
                     nb_variables_per_model,
                     nb_stack_blocks,
                     width,
                     depth,
                     epochs,
                     batch_size,
                     activation,
                     batch_norm,
                     dropout_rate,
                     sample_weight,
                     nb_cores,
                     enable_gpu,
                     memory_growth):
            """
            Called when initializing the classifier
            """
            self.feature_list = feature_list
            self.target_type = target_type
            self.validation_ratio = validation_ratio
            self.patience = patience
            self.nb_models_per_stack = nb_models_per_stack
            self.nb_variables_per_model = nb_variables_per_model
            self.nb_stack_blocks = nb_stack_blocks
            self.width = width
            self.depth = depth
            self.epochs = epochs
            self.batch_size = batch_size
            self.activation = activation
            self.batch_norm = batch_norm
            self.dropout_rate = dropout_rate
            self.sample_weight = sample_weight
            self.nb_cores = nb_cores
            self.enable_gpu = enable_gpu
            self.memory_growth = memory_growth
            self.model = None
            self.target_replicas = None

        def fit(self, X, y):
            """
            This should fit classifier. All the "work" should be done here.

            Note: assert is not a good choice here and you should rather
            use try/except blog with exceptions. This is just for short syntax.
            """
            if self.target_type == 'classification':
                mode = 'max'
            elif self.target_type == 'regression':
                mode = 'min'

            nn = S3enNetwork(feature_list=self.feature_list,
                             target_type=self.target_type,
                             nb_cores=self.nb_cores,
                             enable_gpu=self.enable_gpu,
                             memory_growth=self.memory_growth,
                             nb_models_per_stack=self.nb_models_per_stack,
                             nb_variables_per_model=
                             self.nb_variables_per_model,
                             nb_stack_blocks=self.nb_stack_blocks,
                             width=self.width,
                             depth=self.depth,
                             activation=self.activation,
                             batch_norm=self.batch_norm,
                             dropout_rate=self.dropout_rate)
            self.model, self.target_replicas = nn.get_model()

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

                callbacks = [
                    perf_callback((X_val_adj, y_val_adj), target_type),
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

        def predict_wrapper(self, X, y=None):
            if self._fitted is not True:
                raise RuntimeError(
                    "You must train model before predicting data!")

            X_adjusted, _ = adjust_data(X,
                                        y,
                                        self.feature_list,
                                        self.target_replicas)
            return self.model.predict(X_adjusted)[:, 0]

        def predict_proba(self, X, y=None):
            if self._fitted is not True:
                raise RuntimeError(
                    "You must train model before predicting data!")

            return self.predict_wrapper(X, y)

        def predict(self, X, y=None):
            if self._fitted is not True:
                raise RuntimeError(
                    "You must train model before predicting data!")

            if self.target_type == 'classification':
                return (self.predict_wrapper(X, y) > 0.5).astype(int)
            elif self.target_type == 'regression':
                return self.predict_wrapper(X, y)

    return s3enEstimatorFlex(feature_list=feature_list,
                             target_type=target_type,
                             validation_ratio=validation_ratio,
                             patience=patience,
                             nb_models_per_stack=nb_models_per_stack,
                             nb_variables_per_model=nb_variables_per_model,
                             nb_stack_blocks=nb_stack_blocks,
                             width=width,
                             depth=depth,
                             epochs=epochs,
                             batch_size=batch_size,
                             activation=activation,
                             batch_norm=batch_norm,
                             dropout_rate=dropout_rate,
                             sample_weight=sample_weight,
                             nb_cores=nb_cores,
                             enable_gpu=enable_gpu,
                             memory_growth=memory_growth)
