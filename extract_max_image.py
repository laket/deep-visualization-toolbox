#!/usr/bin/env python

"""
This extract images from lmdb which put max values for each layer.

Requirement:
Net has input_dim layer whose name is data like
----------------------
input: "data"
input_dim: 10
input_dim: 3
input_dim: 227
input_dim: 227
----------------------

"""

import os
import argparse
import logging

import numpy as np
import cv2
import lmdb

import caffe
import caffe.proto.caffe_pb2 as pb
from google.protobuf import text_format

formatter = logging.Formatter("[%(levelname)s]%(funcName)s(%(lineno)d) %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

path_net = "/data/project/mnist/model/template/vis_net.prototxt"
path_trained = "/data/project/mnist/model/template/snapshot/model_iter_10000.caffemodel"
path_mean = None
IS_GRAY = True
path_lmdb = "/data/project/mnist/data/mnist_test_lmdb"
# image scaling parameter. (=transform_param.scale of net definition)
scale = 0.00390625


def extract_blob_names(net_def):
    """
    extract blob names which top on convolutional layer and SoftMax Layer.
    
    Arguments
      net_def:  caffe_pb2.NetParameter

    Retrun
      dictionary {layer_name:blob_name}
    """

    blob_names = {}
    
    for layer in net_def.layer:
        name = layer.name
        layertype = layer.type

        if layertype == "Convolution":
            top_name = layer.top[0]
            blob_names[name] = top_name
            
            #top_blob = net.blobs[top_name]
            #logging.debug("%s : shape %s" % (name, str(top_blob.data.shape)))

    return blob_names

def get_images_from_key(txn, keys):
    """
    Arguments:
       txn : lmdb.Transaction for lmdb containing Datum
       keys : 2d list of key of lmdb. keys[k][i] = i-th filter's top-k image

    Return:
       #2d list of images
       #return[i][k] : ndarray, top-k-th image in i-th filter
       list of images
       return[i] : ndarray, combined top-k image in i-th filter
    """
    images = []
    num_filter, num_top = len(keys[0]), len(keys)

    for i in range(num_filter):
        ith_images = []
        
        for k in range(num_top):
            key = keys[k][i]
            v = txn.get(key)

            datum = pb.Datum()
            datum.ParseFromString(v)

            #print ("w: %d h: %d c: %d" % (datum.width, datum.height,datum.channels))        
            img = np.array(bytearray(datum.data))
            img = img.reshape(datum.channels, datum.height, datum.width)
            img = np.transpose(img, axes=(1,2,0))

            ith_images.append(img)

        combined_image = np.concatenate(ith_images,axis=1)
        images.append(combined_image)
        
    return images
        
    
def calculate_score_at_each_layer(net, blob_names):
    """
    calculate score(sum of output values) at each target layer.

    Arguments:
      net : caffe.Net whic have already run forward.
      blob_names : target layer dictionary {layer_name:blob_name}.

    Return:
      dictionary {layer_name : score}

    Memo:
      should we use max instead of sum?
    """
    dict_score = {}
    
    for layer_name, blob_name in blob_names.items():
        # shape of convolution layer is (num_data, channel, height, width)
        # assum num_data = 1
        score = net.blobs[blob_name].data[0].sum(axis=(1,2))
        dict_score[layer_name] = score
    
    return dict_score
    

def forward_data(num_top):
    net = caffe.Net(path_net, path_trained, caffe.TEST)
    net_def = pb.NetParameter()

    with open(path_net) as f:
        text_format.Merge(f.read(), net_def)    

    blob_names = extract_blob_names(net_def)

    lmdb_env = lmdb.open(path_lmdb, readonly=True, lock=False)

    # key : layer_name value : array of score at each data
    # score is 1d array. score[i] is i-th filters score
    ranking_dict = {}
    lmdb_keys = []
    for layer_name in blob_names.keys():
        ranking_dict[layer_name] = []
    
    with lmdb_env.begin() as txn:
        cur = txn.cursor()
        cur.first()
        
        for i in range(100000):
            k,v = cur.item()

            lmdb_keys.append(k)
            
            datum = pb.Datum()
            datum.ParseFromString(v)

            #print ("w: %d h: %d c: %d" % (datum.width, datum.height,datum.channels))        
            img = np.array(bytearray(datum.data))
            
            # (w*h,) -> (num_data, channel, height, width)
            img = img.reshape((1, datum.channels, datum.height, datum.width))
            img = img.astype(np.float)
            img *= scale
            
            net.blobs["data"].data[...] = img
            out = net.forward()

            scores = calculate_score_at_each_layer(net, blob_names)

            for layer_name, score in scores.items():
                ranking_dict[layer_name].append(score)
            
            if not cur.next():
                break

        #key: layer_name value : array of score top-k image for each filter.
        # value[i] : top-k images in i-th filter
        top_image_dict = {}
                
        for layer_name, scores in ranking_dict.items():
            #rankigs[i] = i-th filters rankings (ascent order) 
            rankings = np.array(scores).argsort(axis=0)
            rankings = rankings[:num_top]
            num_filter = rankings.shape[1]

            key_mat = []
            # convert array indices to lmdb keys.
            for k in range(num_top):
                mat = []
                
                for i in range(num_filter):
                    key = lmdb_keys[rankings[k][i]]
                    mat.append(key)
                    
                key_mat.append(mat)

            images = get_images_from_key(txn, key_mat)
            
            top_image_dict[layer_name] = images

    return top_image_dict
            
            
if __name__ == "__main__":
    top_image_dict = forward_data(3)
    out_dir = "top_images"

    for layer_name, images in top_image_dict.items():
        p = os.path.join(out_dir, layer_name)

        if not os.path.exists(p):
            os.mkdir(p)

        for i, image in enumerate(images):
            path_out = os.path.join(p, "%d.png" % i)
            cv2.imwrite(path_out, image)
    
