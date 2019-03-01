import numpy as np

from .xml_analyzer import parse_xml, get_conv_weights, get_affine_weights, get_fc_weights

def load_weights_in_affine_layers(model, reversed_affine_layers_info):
    # Load weights in the affine layers
    for af in reversed_affine_layers_info:    
        g, b = np.split(af['weights'], 2) 
        model.get_layer(af['name']).set_weights([g, b])                    
            
def load_weights_in_conv_layers(model, reversed_conv_layers_info):
    # Load weights in the convolution layers
    for cv in reversed_conv_layers_info:    
        # we need to first separate bias from W
        weights = cv['weights']
        num_filters = cv['num_filters']
        w = weights[:-num_filters]
        b = weights[-num_filters:]    
        mw = model.get_layer(cv['name']).get_weights()
        assert len(mw) == 2    

        # we need to manually reshape and then transpose it
        assert cv['nc'] == cv['nr']
        filter_size = cv['nc']
        depth = int(len(w)/(num_filters * filter_size * filter_size))

        reshaped_w = np.reshape(w, [num_filters, depth, filter_size, filter_size])
        transposed_w = np.transpose(reshaped_w, [2,3,1,0])

        model.get_layer(cv['name']).set_weights([transposed_w, b])
        
def load_weights_in_fc_layer(model, fc_weights):
    # Load weights in the fully connected layer
    fcw = model.get_layer("embedding_layer").get_weights()
    reshaped_fcw = np.reshape(fc_weights, fcw[0].shape)
    model.get_layer("embedding_layer").set_weights([reshaped_fcw])

def load_weights(model, xml_weights):
    xdict = parse_xml(xml_weights)
    conv_layers_info = get_conv_weights(xdict)
    reversed_conv_layers_info = conv_layers_info[::-1]
    affine_layers_info = get_affine_weights(xdict)
    reversed_affine_layers_info = affine_layers_info[::-1]
    fc_weights = get_fc_weights(xdict)

    load_weights_in_affine_layers(model, reversed_affine_layers_info)
    load_weights_in_conv_layers(model, reversed_conv_layers_info)
    load_weights_in_fc_layer(model, fc_weights)