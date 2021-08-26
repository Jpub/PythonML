#! /usr/bin/env python

import os
import argparse, json
from utils.utils import makedirs
from tensorflow.keras.models import load_model
import tensorflow.contrib.saved_model as saved_model
import tensorflow as tf

def save_with_default(args):
    tf.enable_eager_execution()
    config_path  = args.conf

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    output_path = config['serving']['output_path']

    model = load_model(config['train']['saved_weights_name'])
    saved_model.save_keras_model(model, output_path, serving_only=True)

def save_with_signature(args):
    config_path  = args.conf

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    output_path = config['serving']['output_path']

    model = load_model(config['train']['saved_weights_name'])

    model_input = tf.saved_model.utils.build_tensor_info(model.inputs[0])
    model_output1 = tf.saved_model.utils.build_tensor_info(model.outputs[0])
    model_output2 = tf.saved_model.utils.build_tensor_info(model.outputs[1])
    model_output3 = tf.saved_model.utils.build_tensor_info(model.outputs[2])

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'inputs': model_input},
            outputs={'output1': model_output1, 'output2':model_output2, 'output3':model_output3},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder = tf.saved_model.builder.SavedModelBuilder(output_path)

    with tf.keras.backend.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict':
                    prediction_signature,
            })
 
        builder.save()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    
    args = argparser.parse_args()
    save_with_signature(args)
