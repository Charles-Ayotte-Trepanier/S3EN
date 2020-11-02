import random
import re
import os
import numpy as np
from tensorflow.keras.layers import Dense,Input, Embedding, concatenate,\
    Flatten, Average, Dropout, BatchNormalization, Activation
from tensorflow.keras import Sequential, Model
from tensorflow import config, distribute
from S3EN.helpers import build_layers

def dense_layers(model,
                 dims,
                 target_type,
                 activation,
                 batch_norm,
                 dropout_rate):
    nb_layers = len(dims)
    for i in range(nb_layers-1):
        if i < nb_layers - 2:
            act = activation
            if batch_norm == 'yes':
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
            if dropout_rate > 0:
                model = \
                    Dropout(rate=dropout_rate,
                            input_shape=(dims[i + 1],)
                            )(model)
        else:
            if target_type == 'classification':
                act = 'sigmoid'
            elif target_type == 'regression':
                act = 'linear'
            model = Dense(dims[i+1],\
                          input_shape=(dims[i],),\
                          activation=act)(model)
    return model

def create_initial_layers(feature_list=None):
    in_out_list = []
    for feature in feature_list:
        feat_dict = {}
        feat_nm = feature['feat_nm']
        feat_dict['feat_nm'] = feat_nm
        feat_type = feature['type']
        if feat_type == 'numerical':
            input_layer = Input(shape=(1))
            feat_dict['in'] = input_layer
            feat_dict['out'] = input_layer
            feat_dict['outdim'] = 1
        elif feat_type == 'categorical':
            input_dim = feature['input_dim']
            output_dim = feature['output_dim']
            f_model = Sequential()
            feat_embedding = Embedding(input_dim, output_dim, input_length=1)
            f_model.add(feat_embedding)
            f_model.add(Flatten(name=f'embeddings-{feat_nm}'))
            feat_dict['in'] = f_model.input
            feat_dict['out'] = f_model.output
            feat_dict['outdim'] = output_dim
        in_out_list.append(feat_dict)
    return in_out_list

def concat(layers, in_or_out):
    return concatenate([layer[in_or_out] for layer in layers])

def create_subnetwork(tensors,
                      out_dim,
                      target_type,
                      width,
                      depth,
                      activation,
                      batch_norm,
                      dropout_rate):
    layer_dims = build_layers(out_dim,
                              1,
                              width,
                              depth,
                              dropout_rate)
    prediction = dense_layers(concatenate(tensors),
                              layer_dims,
                              target_type,
                              activation,
                              batch_norm,
                              dropout_rate)
    return {'in': tensors, 'out': prediction}

def create_stacking_block(input_lists,
                          target_type,
                          width,
                          depth,
                          activation,
                          batch_norm,
                          dropout_rate):
    outputs = []
    for input_list in input_lists:
        tensors = input_list['tensors']
        out_dim = input_list['out_dim']
        outputs.append(
            create_subnetwork(tensors,
                              out_dim,
                              target_type,
                              width,
                              depth,
                              activation,
                              batch_norm,
                              dropout_rate)['out'])
    layer_dims = build_layers(len(input_lists), 1, width, depth)
    prediction = dense_layers(concatenate(outputs),
                              layer_dims,
                              target_type,
                              activation,
                              batch_norm,
                              dropout_rate)
    stacking_loss = [prediction] + outputs
    return {'final_pred': prediction,
            'loss_output': concatenate(stacking_loss),
            'output_shape': len(stacking_loss)}

def create_ensemble(feature_list,
                    target_type='classification',
                    nb_cores=None,
                    enable_gpu='yes',
                    memory_growth='yes',
                    nb_models_per_stack=20,
                    nb_variables_per_model=None,
                    nb_stack_blocks=5,
                    width=1,
                    depth=1,
                    activation='elu',
                    batch_norm='no',
                    dropout_rate=0):

    if target_type == 'classification':
        loss = 'binary_crossentropy'
        metrics = ['binary_accuracy']
    elif target_type == 'regression':
        loss = 'mse'
        metrics = ['mae', 'mse']

    if enable_gpu == "yes":
        prefix = 'GPU'
    else:
        prefix = 'CPU'

    cores = config.experimental.list_physical_devices(prefix)
    if len(cores) == 0 and enable_gpu == "yes":
        cores = config.experimental.list_physical_devices('XLA_GPU')
    if memory_growth == "yes" and len(cores) > 0 and enable_gpu == "yes":
        for core in cores:
            config.experimental.set_memory_growth(core, True)
    try:
        cores = [i.name for i in cores]
        if nb_cores is not None:
            cores = cores[:nb_cores]
        cores = [re.search(f"{prefix}:\d$", i).group() for i in cores]
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        cores = []


    # Set some defaults number of features to each in each sub-network
    if nb_variables_per_model is None:
        nb_variables_per_model = max(int(np.sqrt(len(feature_list)-1)), 2)

    total_features = len(feature_list)
    feature_inx = list(range(total_features))

    initial_layers = create_initial_layers(feature_list)
    inputs = [initial_layer['in'] for initial_layer in initial_layers]

    def get_output_dim(layers):
        return sum(
            [initial_layer['outdim'] for initial_layer in layers])

    strategy = distribute.MirroredStrategy(devices=cores,
                        cross_device_ops=distribute.HierarchicalCopyAllReduce()
                        )
    with strategy.scope():
        stacking_blocks_output = []
        for i in range(nb_stack_blocks):

            input_lists = []
            for j in range(nb_models_per_stack):
                #get features:
                cur_indexes = random.sample(feature_inx,
                                            k=nb_variables_per_model)
                input_list = []
                layers = []
                for index in cur_indexes:
                    input_list.append(inputs[index])
                    layers.append(initial_layers[index])
                out_dim = get_output_dim(layers)
                input_lists.append({'tensors': input_list, 'out_dim': out_dim})
            stacking_output = create_stacking_block(input_lists,
                                                    target_type,
                                                    width,
                                                    depth,
                                                    activation,
                                                    batch_norm,
                                                    dropout_rate)
            stacking_blocks_output.append(stacking_output)

        final = Average()(
            [stacked['final_pred'] for stacked in stacking_blocks_output])

        targets = concatenate([final] + \
                  [stacked['loss_output']
                   for stacked in stacking_blocks_output])

        output_dim = 1 + sum([stacked['output_shape']
                              for stacked in stacking_blocks_output])

        model = Model(inputs, targets)
        model.compile(optimizer='adam',
                      loss=loss,
                      metrics=metrics)

    return model, output_dim