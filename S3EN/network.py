import random
import re
import os
from math import ceil
import numpy as np
from tensorflow.keras.layers import Dense,Input, Embedding, concatenate,\
    Flatten, Average, Dropout, BatchNormalization, Activation
from tensorflow.keras import Sequential, Model
from tensorflow import config, distribute


class S3enNetwork:
    def __init__(self,
                 feature_list,
                 target_type,
                 nb_cores=None,
                 enable_gpu='no',
                 memory_growth='no',
                 nb_models_per_stack=20,
                 nb_variables_per_model=None,
                 nb_stack_blocks=10,
                 width=1,
                 depth=1,
                 activation='elu',
                 batch_norm='no',
                 dropout_rate=0):

        self.feature_list = feature_list
        self.target_type = target_type
        self.nb_cores = nb_cores
        self.enable_gpu = enable_gpu
        self.memory_growth = memory_growth
        self.nb_models_per_stack = nb_models_per_stack
        self.nb_variables_per_model = nb_variables_per_model
        self.nb_stack_blocks = nb_stack_blocks
        self.width = width
        self.depth = depth
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        if self.nb_variables_per_model is None:
            self.nb_variables_per_model = \
                max(int(np.sqrt(len(self.feature_list)-1)), 2)

        if self.target_type == 'classification':
            self.loss = 'binary_crossentropy'
            self.metrics = ['binary_accuracy']
        elif self.target_type == 'regression':
            self.loss = 'mse'
            self.metrics = ['mae', 'mse']

    def __build_layers(self,
                       input_dim,
                       output_dim):
        layers_dim = []
        if self.depth > 0:
            nb_layers = int(np.round(np.log(input_dim) * self.depth))
            if nb_layers == 0:
                nb_layers += 1
            if self.depth > 0:
                layer_dim = max(int(float(self.width) * input_dim), output_dim)
                factor = np.power(float(output_dim) / layer_dim, 1 / nb_layers)
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
            if self.dropout_rate > 0:
                layers_dim = layers_dim[:1] + \
                             [ceil(layer / (1 - self.dropout_rate))
                              for layer in layers_dim[1:]]
        layers_dim[-1] = output_dim
        return layers_dim

    def __dense_layers(self,
                       model,
                       dims):
        nb_layers = len(dims)
        for i in range(nb_layers - 1):
            if i < nb_layers - 2:
                act = self.activation
                if self.batch_norm == 'yes':
                    model = \
                        Dense(dims[i + 1],
                              input_shape=(dims[i],),
                              activation=None)(model)
                    model = BatchNormalization()(model)
                    model = Activation(act)(model)
                else:
                    model = \
                        Dense(dims[i + 1],
                              input_shape=(dims[i],),
                              activation=act)(model)
                if self.dropout_rate > 0:
                    model = \
                        Dropout(rate=self.dropout_rate,
                                input_shape=(dims[i + 1],)
                                )(model)
            else:
                if self.target_type == 'classification':
                    act = 'sigmoid'
                elif self.target_type == 'regression':
                    act = 'linear'
                model = Dense(dims[i + 1], \
                              input_shape=(dims[i],), \
                              activation=act)(model)
        return model

    def __get_random_features(self,
                              inputs,
                              initial_layers):

        total_features = len(self.feature_list)
        feature_inx = list(range(total_features))

        # get features:
        cur_indexes = random.sample(feature_inx,
                                    k=self.nb_variables_per_model)
        input_list = []
        layers = []
        for index in cur_indexes:
            input_list.append(inputs[index])
            layers.append(initial_layers[index])
        out_dim = sum([initial_layer['outdim'] for initial_layer in layers])
        return {'tensors': input_list, 'out_dim': out_dim}

    def __create_stacking_block(self,
                                input_lists):
        outputs = []
        for input_list in input_lists:
            tensors = input_list['tensors']
            out_dim = input_list['out_dim']
            outputs.append(
                self.__create_subnetwork(tensors, out_dim)['out'])
        layer_dims = self.__build_layers(len(input_lists), 1)
        prediction = self.__dense_layers(concatenate(outputs), layer_dims)
        stacking_loss = [prediction] + outputs
        return {'final_pred': prediction,
                'loss_output': concatenate(stacking_loss),
                'output_shape': len(stacking_loss)}

    def __create_initial_layers(self):
        in_out_list = []
        for feature in self.feature_list:
            feat_dict = {}
            feat_nm = feature['feat_nm']
            feat_dict['feat_nm'] = feat_nm
            feat_type = feature['type']
            if feat_type == 'numerical':
                input_layer = Input(shape=1)
                feat_dict['in'] = input_layer
                feat_dict['out'] = input_layer
                feat_dict['outdim'] = 1
            elif feat_type == 'categorical':
                input_dim = feature['input_dim']
                output_dim = feature['output_dim']
                f_model = Sequential()
                feat_embedding = Embedding(input_dim, output_dim,
                                           input_length=1)
                f_model.add(feat_embedding)
                f_model.add(Flatten(name=f'embeddings-{feat_nm}'))
                feat_dict['in'] = f_model.input
                feat_dict['out'] = f_model.output
                feat_dict['outdim'] = output_dim
            in_out_list.append(feat_dict)
        return in_out_list

    def __create_subnetwork(self,
                            tensors,
                            out_dim):
        layer_dims = self.__build_layers(out_dim, 1)
        prediction = self.__dense_layers(concatenate(tensors),
                                       layer_dims)
        return {'in': tensors, 'out': prediction}

    def __get_strategy(self):
        if self.enable_gpu == "yes":
            prefix = 'GPU'
        else:
            prefix = 'CPU'

        cores = config.experimental.list_physical_devices(prefix)
        if len(cores) == 0 and self.enable_gpu == "yes":
            cores = config.experimental.list_physical_devices('XLA_GPU')
        if self.memory_growth == "yes" and len(cores) > 0 \
            and self.enable_gpu == "yes":
            for core in cores:
                config.experimental.set_memory_growth(core, True)
        try:
            cores = [i.name for i in cores]
            if self.nb_cores is not None:
                cores = cores[:self.nb_cores]
            cores = [re.search(f"{prefix}:\d$", i).group() for i in cores]
        except:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            cores = []

        strategy = \
            distribute.MirroredStrategy(devices=cores,
                                        cross_device_ops=
                                        distribute.HierarchicalCopyAllReduce())

        return strategy.scope()

    def get_model(self):

        with self.__get_strategy():
            initial_layers = self.__create_initial_layers()
            inputs = [initial_layer['in'] for initial_layer in initial_layers]
            stacking_blocks_output = []
            for i in range(self.nb_stack_blocks):
                input_lists = []
                for j in range(self.nb_models_per_stack):
                    inputs_subset = self.__get_random_features(inputs,
                                                             initial_layers)
                    input_lists.append(inputs_subset)
                stacking_output = self.__create_stacking_block(input_lists)
                stacking_blocks_output.append(stacking_output)

            final = Average()(
                [stacked['final_pred'] for stacked in stacking_blocks_output])

            targets = concatenate([final] +
                                  [stacked['loss_output']
                                  for stacked in stacking_blocks_output])

            output_dim = 1 + sum([stacked['output_shape']
                                  for stacked in stacking_blocks_output])

            model = Model(inputs, targets)
            model.compile(optimizer='adam',
                          loss=self.loss,
                          metrics=self.metrics)

        return model, output_dim
