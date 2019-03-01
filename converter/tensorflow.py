import os
import shutil

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

def convert_to_tf_saved_model(keras_model, output_path):
    sess = K.get_session()
    output_node_names = [node.op.name for node in keras_model.outputs]
        
    inputs={'input_image': keras_model.input}
    outputs={t.name: t for t in keras_model.outputs}
    
    exported_model_path = os.path.join(output_path, 'exported')
    
    if os.path.exists(exported_model_path):
        shutil.rmtree(exported_model_path)
    
    tf.saved_model.simple_save(sess,
            os.path.join(output_path, 'exported'),
            inputs=inputs,
            outputs=outputs)
    
def convert_to_tf_frozen_model(keras_model, output_path):        
    sess = K.get_session()
    
    # get the names of the output nodes
    output_node_names = [node.op.name for node in keras_model.outputs]
    
    output_model_dir = output_path
    output_model_name = 'dlib_face_recognition_resnet_model_v1.pb'

    constant_graph = graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(),
                output_node_names)

    graph_io.write_graph(constant_graph, output_model_dir , output_model_name, as_text=False)