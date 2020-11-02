import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, mean_squared_error

def duplicate(y, duplications):
    return np.repeat(y.reshape(-1, 1), duplications, axis=1).astype(float)

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)

class perf_callback(Callback):
    def __init__(self, data, target_type='classification'):
        self.X = data[0]
        self.y = data[1]
        self.target_type = target_type
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X)
        if self.target_type == 'classification':
            perf = roc_auc_score(self.y, y_pred, average='micro')
        elif self.target_type == 'regression':
            perf = mean_squared_error(self.y, y_pred)
        logs['validation'] = perf

def adjust_data(X, y, feature_list, target_replicas):
    X_adjusted = [X[col['feat_nm']].values.reshape(-1, 1) for col in
                  feature_list]

    if y is not None:
        y_adjusted = duplicate(y, target_replicas)
    else:
        y_adjusted = None
    return X_adjusted, y_adjusted