import os

## MODELS ##
model_dir = "/data/project/cifar10/model/template"

path_net                 = os.path.join(model_dir, 'vis_net.prototxt')
path_weight              = os.path.join(model_dir, 'snapshot/cifar10_full_iter_4000.caffemodel')

### VISUALIZED MODEL ### 
dir_deepvis_img          = os.path.join(model_dir, 'top_images')


## DATA ##
data_dir = "/data/project/cifar10/data"

path_mean                = os.path.join(data_dir, 'mean.binaryproto')
scale                    = 1.0
#path_mean      = None
path_labels              = os.path.join(data_dir, 'vis_labels.txt')



