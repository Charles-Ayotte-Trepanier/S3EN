from S3EN.estimators import s3enEstimatorClassifier, s3enEstimatorRegressor

def s3enEstimator(feature_list,
                 target_type='classification',
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
    if target_type == 'classification':
        return s3enEstimatorClassifier(feature_list,
                                       nb_models_per_stack,
                                       nb_variables_per_model,
                                       nb_stack_blocks,
                                       width,
                                       depth,
                                       epochs,
                                       batch_size,
                                       sample_weight,
                                       enable_gpu,
                                       memory_growth)
    elif target_type == 'regression':
        return s3enEstimatorRegressor(feature_list,
                                       nb_models_per_stack,
                                       nb_variables_per_model,
                                       nb_stack_blocks,
                                       width,
                                       depth,
                                       epochs,
                                       batch_size,
                                       sample_weight,
                                       enable_gpu,
                                       memory_growth)