#!/usr/bin/env python
"""
command line tool for Deep Visualization Toolbox.
This doesn't dependoverride values of setting.py

- caffevis_caffe_root (by environ["CAFFE_ROOT"])
- static_files_dir (by command line)
- caffevis_data_hw (by input image)

Requirement:
require "prob" blob in net of TEST mode file to visualize.

"""
import argparse
import os
import sys
import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

import tempfile

import numpy as np
import cv2
import caffe
import caffe.proto.caffe_pb2 as pb

from core import LiveVis
from bindings import bindings

IS_GRAY = True

try:
    import settings
except:
    print '\nCould not import settings from settings.py. You should first copy'
    print 'settings.py.template to settings.py and edit the caffevis_caffe_root'
    print 'variable to point to your caffe path.'
    print
    print '  $ cp settings.py.template settings.py'
    print '  $ < edit settings.py >\n'
    raise

import os

if "CAFFE_ROOT" not in os.environ:
    print "\nCould not find environment variable CAFFE_ROOT."
    print "set it to caffe installed directory."
    raise
    

def create_empty_mean_file_protobuf(file_path, height, width, channel):
    mean_file = pb.BlobProto()

    #optional int32 num = 1 [default = 0];
    #optional int32 channels = 2 [default = 0];
    #optional int32 height = 3 [default = 0];
    #optional int32 width = 4 [default = 0];
    mean_file.shape.dim.append(1)
    mean_file.shape.dim.append(channel)
    mean_file.shape.dim.append(height)
    mean_file.shape.dim.append(width)

    mean_file.data.extend([0.0] * (height*width*channel))

    with open(file_path, "wb") as f:
        print file_path
        f.write(mean_file.SerializeToString())

def create_empty_mean_file_npy(file_path, height, width, channel):
    # required ndarray(channel, height, width)
    mean = np.zeros((channel, height, width))
    np.save(file_path, mean)
        
def get_net_information(caffe_net_file):
    # see below files to know caffe.Net interface.
    # see caffe/python/caffe/test/test_net.py
    # see caffe/python/caffe/draw.py get_pydot_graph
    net = caffe.Net(net_file, caffe.TRAIN)

    datalayer = None
    labellayer = None

    #probs_flat = self.net.blobs['prob'].data.flatten()
    
    # detect input layer
    for layer in caffe_net.layer:
        name = layer.name
        layertype = layer.type

        if layertype == "Data":
            datalayer = layer
            break
        
def decide_image_size(image_file):
    image = cv2.imread(image_file)
    return image.size

def override_setting(img_path):
    settings.caffevis_caffe_root = os.environ["CAFFE_ROOT"]

    if not os.path.exists(settings.caffevis_caffe_root):
        raise Exception('ERROR: Set caffevis_caffe_root in settings.py first.')

    #im = cv2_read_file_rgb(os.path.join(self.settings.static_files_dir, self.latest_static_filename))
    #static_files_regexp = '.*\.(jpg|jpeg|png)$'
    #static_files_ignore_case = True
    #static_file_stretch_mode = False        # True to stretch to square, False to crop to square. (Can change at runtime via 'stretch_mode' key.)
        
    #caffevis_data_hw         = (227,227)
    # Now this is not used (commit f9e8a7d1b0963e155bdefbf740ec1e5a12254f3e)
    #caffevis_label_layers    = ('fc8', 'prob')
    import input_files

    img_abspath = os.path.abspath(img_path)
    settings.static_files_dir = os.path.dirname(img_abspath)

    if IS_GRAY:
        image = cv2.imread(img_abspath, 0)
        image = image.reshape(image.shape + (1,))
    else:
        image = cv2.imread(img_abspath)
        
    if len(image.shape) == 3:
        im_height, im_width, im_channel = image.shape[:3]
    else:
        im_height, im_width = image.shape[:2]
        im_channel = 1
    
    logger.debug("image size: w %d h %d c %d" % (im_width, im_height, im_channel))
    settings.caffevis_data_hw = (im_height, im_width)
    
    """
    self.net = caffe.Classifier(
        settings.caffevis_deploy_prototxt,
        settings.caffevis_network_weights,
        mean = self._data_mean,
        channel_swap = self._net_channel_swap,
        raw_scale = self._range_scale,
        #image_dims = (227,227),
    )
    """

    settings.caffevis_deploy_prototxt = input_files.caffevis_deploy_prototxt
    settings.caffevis_network_weights = input_files.caffevis_network_weights

    # if you don't use mean image file, this creates empty mean image file.
    # channel order is BGR
    if input_files.caffevis_data_mean is None:
        tempdir = tempfile.gettempdir()
        path_temp = os.path.join(tempdir, "temp_mean.npy")
        create_empty_mean_file_npy(path_temp, im_height, im_width, im_channel)
        settings.caffevis_data_mean = path_temp
    else:
        settings.caffevis_data_mean = input_files.caffevis_data_mean
    settings.caffevis_labels        = input_files.caffevis_labels
    settings.caffevis_unit_jpg_dir  = input_files.caffevis_unit_jpg_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="image", help="input image", required=True)
    args = parser.parse_args()

    img_path = args.image

    if not os.path.exists(img_path):
        sys.err.write("couldn't find image file: %s\n" % img_path)
        sys.exit(-1)
    
    override_setting(img_path)
    
    lv = LiveVis(settings)

    help_keys, _ = bindings.get_key_help('help_mode')
    quit_keys, _ = bindings.get_key_help('quit')
    print '\n\nRunning toolbox. Push %s for help or %s to quit.\n\n' % (help_keys[0], quit_keys[0])
    lv.run_loop()


    
if __name__ == '__main__':
    main()

