import os


"""
root_dir = "/data/project/mnist/model/template"

caffevis_deploy_prototxt = os.path.join(root_dir, 'vis_net.prototxt')
caffevis_network_weights = os.path.join(root_dir, 'snapshot/model_iter_10000.caffemodel')
#caffevis_data_mean       = os.path.join(root_dir, '/models/caffenet-yos/ilsvrc_2012_mean.npy')
caffevis_data_mean       = None
caffevis_labels          = os.path.join(root_dir, '../../data/vis_labels.txt')
caffevis_unit_jpg_dir    = root_dir + '/models/caffenet-yos/unit_jpg_vis'
"""

root_dir = "/data/project/cifar10/model/template"

caffevis_deploy_prototxt = os.path.join(root_dir, 'vis_net.prototxt')
caffevis_network_weights = os.path.join(root_dir, 'snapshot/cifar10_full_iter_4000.caffemodel')
caffevis_data_mean       = os.path.join(root_dir, '../../data/mean.binaryproto')
#caffevis_data_mean       = None
caffevis_labels          = os.path.join(root_dir, '../../data/vis_labels.txt')
caffevis_unit_jpg_dir    = os.path.join(root_dir, 'top_images')

