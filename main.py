# PARAMETERS
BASE = "/"
OUTPUT = BASE + "/output"
INPUT = BASE + "/input"
MODEL = BASE + '/models/d2_tf.pth'

# D2NET METHODS
PREPROCESSING = 'caffe'
USE_RELU = True
OUTPUT_TYPE = 'npz'
MULTISCALE = True

# MAX EDGE SIZE (WIDTH OR HEIGHT)
MAX_EDGE = 1600

# MAX SUM OF EDGES (WIDTH + HEIGHT)
MAX_SUM_EDGES = 3200

# EXTRACTED FILE EXTENSION
OUTPUT_EXTENSION = '.d2-net'

import torch

# IMPORT
import extractor as e
import matcher as m

# FUNCTION
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

e.extract_images(INPUT, MAX_EDGE, MAX_SUM_EDGES, MODEL, USE_RELU, USE_CUDA, PREPROCESSING, DEVICE, OUTPUT_EXTENSION)
m.match_images(needleFolderPath=, haystackFolderPath=, OUTPUT)
