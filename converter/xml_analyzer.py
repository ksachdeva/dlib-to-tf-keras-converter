from collections import Counter

import numpy as np
import xmltodict

def parse_xml(fp_path):
    with open(fp_path) as f:
        xml_content = f.read()
    return xmltodict.parse(xml_content)

def count_number_of_layers(xdict):
    net = xdict['net']   # the first field is 'net'
    print(f"Total number of layer entries - {len(net['layer'])}")
    
    layer_names = []            
    for l in net['layer']:
        for k, v in l.items():
            if not k.startswith('@'):
                layer_names.append(k)
            
    ln_vs_count = dict(Counter(layer_names))
    return ln_vs_count             
            

def weights_from_text(s):        
    weights = s.replace('\n', ' ').replace('\r', '')
    weights = weights.split()
    w = np.array(weights)
    w = w.astype(np.float64)
    return w       

def get_fc_weights(xdict):
    net = xdict['net']            
    fc_weights = None    
    # Ugly, ugly ... super ugly !
    for l in net['layer']:
        for k, v in l.items():
            if not k.startswith('@'):
                if k == 'fc_no_bias':
                    for lk, lv in l[k].items():
                        if lk == '#text':                                
                            fc_weights = weights_from_text(l[k]['#text'])
                            
            
    if fc_weights is None:
        raise ValueError('No FC Weights found')
            
    assert len(fc_weights) == 32768            
    return fc_weights
            
          
def get_conv_weights(xdict):
    net = xdict['net']
            
    conv_layers = []
    starting_index = 29
            
    # find all the conv layers
    for l in net['layer']:
        for k, v in l.items():
            if not k.startswith('@'):
                if k == 'con':                      
                    num_filters = l[k]['@num_filters']
                    nr = l[k]['@nr']
                    nc = l[k]['@nc']
                    sy = l[k]['@stride_y']
                    sx = l[k]['@stride_x']                 
            
                    for lk, lv in l[k].items():
                        if lk == '#text':                                
                            conv_weights = weights_from_text(l[k]['#text'])
            
                    conv_layers.append({
                        'name': 'conv_' + str(starting_index),
                        'id': l['@idx'],
                        'num_filters': int(num_filters),
                        'nr': int(nr),
                        'nc': int(nc),
                        'sx': int(sx),
                        'sy': int(sy),
                        'weights': conv_weights,
                        'total_weights': len(conv_weights)
                    })
            
                    starting_index = starting_index - 1
    
            
    assert len(conv_layers) == 29                
    return conv_layers            
            
          
def get_affine_weights(xdict, layer_prefix='sc'):
    net = xdict['net']
            
    affine_layers = []
    starting_index = 29
            
    # now affine layers do not have any attribute
    # .. just #text which is the value of the 
    # weights
            
    # find all the conv layers
    for l in net['layer']:
        for k, v in l.items():            
            if not k.startswith('@'):
                if k == 'affine_con':                       
                    w = weights_from_text(l[k])    
                    affine_layers.append({
                        'name': layer_prefix + '_' + str(starting_index),
                        'id': l['@idx'],
                        'weights': w,
                        'total_weights': len(w)
                    })
            
                    starting_index = starting_index - 1
            
    
    assert len(affine_layers) == 29            
    return affine_layers