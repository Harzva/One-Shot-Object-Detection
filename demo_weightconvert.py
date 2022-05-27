

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import glob
import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb1 import combined_roidb
from roi_data_layer.roibatchLoader2 import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from harzvatool.Tools import Tools
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def save_weight(weight, time, seen):
  time = np.where(time==0, 1, time)
  weight = weight/time[:,np.newaxis]
  result_map = np.zeros((len(weight), len(weight)))
  for i in range(len(weight)):
    for j in range(len(weight)):
      v1 = weight[i]
      v2 = weight[j]
      # v1_ = np.linalg.norm(v1)
      # v2_ = np.linalg.norm(v2)
      # v12 = np.sum(v1*v2)
      # print(v12)
      # print(v1_)
      # print(v2_)
      distance = np.linalg.norm(v1-v2)
      if np.sum(v1*v2)== 0 :
        result_map[i][j] = 0
      else:
        result_map[i][j] = distance
      

  df = pd.DataFrame (result_map)

  ## save to xlsx file

  filepath = 'similarity_%d.xlsx'%(seen)

  df.to_excel(filepath, index=False)

  weight = weight*255


  cv2.imwrite('./weight_%d.png'%(seen), weight)

def parse_args():
  """
  Parse input arguments
  

python test_net_1.py --dataset coco --net res50 \
                    --s 1 --checkepoch 1 --p 354980 \
                    --cuda --g 4 
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='coco', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="./models/res50/coco/faster_rcnn_4_10_1663.pth",
                      type=str)
  # parser.add_argument('--cuda', dest='cuda',
  #                     help='whether use CUDA',
  #                     action='store_true')
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      default=True)
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      default=True)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      default=True)
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--s', dest='checksession',
                      help='checksession to load model',
                      default=4, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=10, type=int)
  parser.add_argument('--p', dest='checkpoint',
                      help='checkpoint to load network',
                      default=1663, type=int)#354981
  parser.add_argument('--path', dest='path',
                      help='checkpoint to load network',
                      default='./models/res50/coco/exp3_faster_rcnn_1_1_13311.pth', type=str)#354981
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--seen', dest='seen',
                       help='Reserved: 1 training, 2 testing, 3 both', default=2, type=int)
  parser.add_argument('--a', dest='average', help='average the top_k candidate samples', default=1, type=int)
  parser.add_argument('--g', dest='group',
                      help='which group want to training/testing',
                      default=4, type=int) 
  parser.add_argument('--pre_t', dest='pre_trained_path',
                      help='resume checkpoint or not',
                      default='./data/pre-trained/pretrain_imagenet_resnet50/model_best.pth.tar', type=str)
  parser.add_argument('--word_embedding', dest='word_embedding', type=Tools.str2bool,
                      help='whether use multiple GPUs',
                      default=True)
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2017_train"
      args.imdbval_name = "coco_2017_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}_{}.yml".format(args.net, args.group) if args.group != 0 else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)


  # Load dataset
  cfg.TRAIN.USE_FLIPPED = False
  imdb_vu, roidb_vu, ratio_list_vu, ratio_index_vu, query_vu ,class_to_name= combined_roidb(args.imdbval_name, False, seen=args.seen)
  #    imdb, roidb, ratio_list, ratio_index, query, class_to_name = combined_roidb(args.imdb_name, True, seen=args.seen)
  imdb_vu.competition_mode(on=True)
  wi = {}
  with open('./cls_names.txt') as f:
    for i, key in enumerate(f.readlines()):
      wi[key.strip()] = i
  dataset_vu = roibatchLoader(roidb_vu, ratio_list_vu, ratio_index_vu, query_vu, 1, imdb_vu.num_classes, training=False, seen=args.seen, class_to_name=class_to_name,word_name_to_index=wi)
  # initilize the network here.
  if args.net == 'vgg16':
      fasterRCNN = vgg16(imdb_vu.classes, pretrained=True, class_agnostic=args.class_agnostic,word_embedding=args.word_embedding,pre_trained_path=args.pre_trained_path)
  elif args.net == 'res101':
      fasterRCNN = resnet(imdb_vu.classes,101, pretrained=True,
                          class_agnostic=args.class_agnostic,word_embedding=args.word_embedding,pre_trained_path=args.pre_trained_path)
  elif args.net == 'res50':
      fasterRCNN = resnet(imdb_vu.classes, 50, pretrained=True,
                          class_agnostic=args.class_agnostic,word_embedding=args.word_embedding,pre_trained_path=args.pre_trained_path)
  elif args.net == 'res152':
      fasterRCNN = resnet(imdb_vu.classes, 152, pretrained=True,
                          class_agnostic=args.class_agnostic,word_embedding=args.word_embedding,pre_trained_path=args.pre_trained_path)

  else:
    print("network is not defined")
    pdb.set_trace()
  
  fasterRCNN.create_architecture()
  for load_name in  glob.glob("/media/ubuntu/data1/hzh/One-Shot-Object-Detection/models/res50/voc/cls_nowordemb_pascal_voc_0712_bs16_s1_g1/*.pth"):
    print("load checkpoint %s" % (load_name))
    # checkpoint = torch.jit.load(load_name)
    checkpoint = torch.load(load_name)
    # print(checkpoint)
    if 'step' in load_name:
        fasterRCNN.load_state_dict(checkpoint,False)
    else:
        fasterRCNN.load_state_dict(checkpoint['model'],False)
        fasterRCNN.eval()


        torch.save(fasterRCNN.state_dict(),load_name, use_new_zipfile_serialization=False)
