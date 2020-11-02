from math import ceil
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, mean_squared_error

def build_layers(input_dim, output_dim, width, depth, dropout_rate=0):
    layers_dim = []
    if depth > 0:
        nb_layers = int(np.round(np.log(input_dim)*depth))
        if nb_layers == 0:
            nb_layers += 1
        if depth > 0:
            layer_dim = max(int(float(width)*input_dim), output_dim)
            factor = np.power(float(output_dim)/layer_dim, 1/nb_layers)
            layers_dim.append(input_dim)
            while layer_dim > output_dim:
                layers_dim.append(layer_dim)
                new_layer_dim = int(np.round(layer_dim * factor))
                if layer_dim == new_layer_dim:
                    new_layer_dim -= 1
                layer_dim = max(new_layer_dim, output_dim)
            while len(layers_dim) < nb_layers:
                new_layer_dim = int(np.round(layer_dim * factor))
                if layer_dim == new_layer_dim:
                    new_layer_dim -= 1
                layer_dim = max(new_layer_dim, output_dim)
                layers_dim.append(layer_dim)
            while len(layers_dim) > nb_layers:
                layers_dim = layers_dim[:-1]
            layers_dim.append(output_dim)
        if dropout_rate > 0:
            layers_dim = layers_dim[:1] +\
                         [ceil(layer/(1-dropout_rate))
                          for layer in layers_dim[1:]]
    layers_dim[-1] = output_dim
    return layers_dim

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