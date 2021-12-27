# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
from datetime import timedelta
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler,SequentialSampler
from torch.nn.parallel import DistributedDataParallel
from roi_data_layer.roidb1 import combined_roidb
from roi_data_layer.roibatchLoader2 import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
'''
    python trainval_net_gai2.py \
    --dataset coco --net res50 \
    --bs 6 --nw 32 \
    --cuda --g 1 --seen 1 --mGPUs



'''

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    # parser.add_argument('--dataset', dest='dataset',
    #                     help='training dataset',
    #                     default='pascal_voc', type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='coco', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='res50', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=10, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=10, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=32, type=int)
    # parser.add_argument('--cuda', dest='cuda',
    #                     help='whether use CUDA',
    #                     action='store_true')
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use multiple GPUs',
                        default=True)
    parser.add_argument('--g', dest='group',
                        help='which group to train, split coco to four group',
                        default=0)
    parser.add_argument('--seen', dest='seen', default=1, type=int)

    # parser.add_argument('--mGPUs', dest='mGPUs',
    #                     help='whether use multiple GPUs',
    #                     action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        default=True)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=6, type=int)
    parser.add_argument('--bs_v', dest='batch_size_val',
                        help='batch_size',
                        default=16, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        default=True)

# config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.01, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=4, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

# set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

# resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default='', type=str)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
# log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        default=True)
    parser.add_argument('--pre_t', dest='pre_trained_path',
                        help='resume checkpoint or not',
                        default='/home/ubuntu/Dataset/Partition1/hzh/lj/One-Shot-Object-Detection-master/data/pre-trained/pretrain_imagenet_resnet50/model_best.pth.tar', type=str)
    parser.add_argument("--local_rank", type=int, default=0,
                        help="local_rank for distributed training on gpus")
    # parser.add_argument('--GPU_ID ', dest='GPU_ID',
    #                     help='resume checkpoint or not',
    #                     default='0,1', type=str)
    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(
                self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(
            self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(
            self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat(
                (self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == '__main__':

    args = parse_args()
    val = False

    print('Called with args:')
    print(args)

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2017_train"
        args.imdbval_name = "coco_2017_val"
        # args.imdb_name = "coco_2017_train"
        # args.imdbval_name = "coco_2017_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    args.cfg_file = "cfgs/{}_{}.yml".format(
        args.net, args.group) if args.group != 0 else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    # pprint.pprint(cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    pprint.pprint(cfg)
        

    # print('Using config:')
    # pprint.pprint(cfg)
    # np.random.seed(cfg.RNG_SEED) 
    
    # cfg.GPU_ID=args.GPU_ID  if args.GPU_ID  else list(range(torch.cuda.device_count())) 
    cfg['CUDA']=True
    print('*'*60,'Using config:')
    pprint.pprint(cfg)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available() and not args.cuda:
        print("*"*60,"WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda

    # create dataloader
    # ===============
    # 获取词向量
    imdb, roidb, ratio_list, ratio_index, query, class_to_name = combined_roidb(args.imdb_name, True, seen=args.seen)
    # imdb, roidb, ratio_list, ratio_index, query, class_to_name = combined_roidb(
    #     args.imdbval_name, False, seen=args.seen)


        #  imdb_vu, roidb_vu, ratio_list_vu, ratio_index_vu, query_vu ,class_to_name= combined_roidb(args.imdbval_name, False, seen=args.seen)
    print('class to name', class_to_name)
    train_size = len(roidb)
    print('{:d} roidb entries'.format(len(roidb)))
    sampler_batch = sampler(train_size, args.batch_size)

    wi = {}
    with open('/home/ubuntu/Dataset/Partition1/hzh/lj/One-Shot-Object-Detection-master/cls_names.txt') as f:
      for i, key in enumerate(f.readlines()):
        wi[key.strip()] = i
    if args.mGPUs:
        torch.distributed.init_process_group(backend="nccl",timeout=timedelta(minutes=60))
        local_rank = torch.distributed.get_rank()
        print('you'*60,local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    print('a'*60)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, query, args.batch_size,
                             imdb.num_classes, training=True, class_to_name=class_to_name, word_name_to_index=wi)
    
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
    #                                 sampler=sampler_batch, num_workers=args.num_workers) 
    print('y'*60)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    sampler=DistributedSampler(dataset,shuffle=False))
    # ==========

    # words = {}
    # f = open("./cls_names.txt", "r")
    # name = f.readline().strip("\n")
    # while name:
    #     if name not in words:
    #         # print('lllllll')
    #         words[name] = []
    #     name = f.readline().strip("\n")
    # print("finish")
    # print("words:", words)
    # f.close()
    # class_word = []
    # f2 = open("./word_w2v.txt", "r")
    # word = f2.readline().strip('\n').split(',')
    # i = 0
    # while i < 500:
    #     class_word.append(word)
    #     word = f2.readline().strip('\n').split(',')
    #     i += 1
    # f2.close()
    # # print("finish2")
    # for j in range(len(class_word)):
    #     i = 0
    #     for pname in words:
    #         words[pname].append(class_word[j][i])
    #         i += 1

    # create output directory
    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    query = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    query_word_vectors = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        query = query.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

        query_word_vectors = query_word_vectors.cuda()
        cfg.CUDA = True

    # make variable
    im_data = Variable(im_data)
    query = Variable(query)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    query_word_vectors = Variable(query_word_vectors)

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic,model_path=args.pre_trained_path)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes,101, pretrained=True,
                            class_agnostic=args.class_agnostic,model_path=args.pre_trained_path)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True,
                            class_agnostic=args.class_agnostic,model_path=args.pre_trained_path)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=True,
                            class_agnostic=args.class_agnostic,model_path=args.pre_trained_path)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr':lr,
                            'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.cuda:
        fasterRCNN.cuda()

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        # load_name = os.path.join(output_dir,
        #                          'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        load_name=args.resume
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))



    if args.mGPUs:
        print('#'*60)
        # local_rank = torch.distributed.get_rank()
        # torch.cuda.set_device(local_rank)
        # device = torch.device("cuda", local_rank)
        # torch.distributed.init_process_group(backend="nccl",timeout=timedelta(minutes=60))
        fasterRCNN = torch.nn.SyncBatchNorm.convert_sync_batchnorm(fasterRCNN)
        fasterRCNN = DistributedDataParallel(fasterRCNN,device_ids=[args.local_rank], output_device=args.local_rank)
        # fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(train_size / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(log_dir=f"logs/{os.path.basename(args.save_dir)}")

    name_list = []
    index_list = []
    print("class_to_name:", class_to_name)
    for name, index in class_to_name.items():
        name_list.append(name)
        index_list.append(index)

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)

        for step in range(iters_per_epoch):
            data = next(data_iter)
            im_data.resize_(data[0].size()).copy_(data[0])
            query.resize_(data[1].size()).copy_(data[1])
            im_info.resize_(data[2].size()).copy_(data[2])
            gt_boxes.resize_(data[3].size()).copy_(data[3])
            num_boxes.resize_(data[4].size()).copy_(data[4])
            query_index = data[5]
            #print('-----------------------',query_index)
            # name_index = query_index[0].int().item()
            # if name_index in index_list:
            #     name_index = index_list.index(name_index)
            # print("name_index:", name_index, name_list[name_index])
            # class_name = name_list[name_index]
            # query_word_vector = words[class_name]

            # for i in range(len(query_word_vector)):
            #     query_word_vector[i] = float(query_word_vector[i])
            # query_word_vector = torch.tensor(query_word_vector)
            # query_word_vector = query_word_vector.cuda()
            # query_word_vector = Variable(query_word_vector)

            # print("query_word_vector", query_word_vector)

            # print("query_index:", query_index[0].int().item())

            # # if(query_index == 0): #背景
            # #   query_word_vectors = words[80]
            # # else:
            # #   index_to_name = ""
            # #   query_word_vectors = words[index]
            # # print("yuancata:",data[5].size()).copy_(data[5])
            # # query_word_vectors.resize_(data[5].size()).copy_(data[5])

            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, margin_loss, RCNN_loss_bbox, \
                rois_label, _ = fasterRCNN(
                    im_data, query, im_info, gt_boxes, num_boxes, query_index)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                + RCNN_loss_cls.mean() + margin_loss.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_margin = margin_loss.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_margin = margin_loss.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e"
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))

                print("\t\t\tfg/bg=(%d/%d), time cost: %f" %
                      (fg_cnt, bg_cnt, end-start))
                      
                print("\t\t\trpn_cls: %.3f, rpn_box: %.3f, rcnn_cls: %.3f, margin: %.3f, rcnn_box %.3f"
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_margin, loss_rcnn_box))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_margin': loss_margin,
                        'loss_rcnn_box': loss_rcnn_box,
                        'epoch':epoch
                    }
                    info2 = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_margin': loss_margin,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars(
                        "logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)
                    logger.add_scalars(
                        "logs_s_{}/losses".format(args.session), info2, epoch)

                    logger.add_scalar('lr', lr, (epoch - 1) * iters_per_epoch + step)
                    logger.add_scalar('loss_rpn_cls', loss_rpn_cls, (epoch - 1) * iters_per_epoch + step)
                    logger.add_scalar('loss_rpn_box', loss_rpn_box, (epoch - 1) * iters_per_epoch + step)
                    logger.add_scalar('loss_rcnn_cls', loss_rcnn_cls, (epoch - 1) * iters_per_epoch + step)
                    logger.add_scalar('loss_margin', loss_margin, (epoch - 1) * iters_per_epoch + step)
                    logger.add_scalar('loss_rcnn_box', loss_rcnn_box, (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()

        save_name = os.path.join(
            output_dir, '{}_{}_{}_{}.pth'.format(os.path.basename(args.save_dir),args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()
