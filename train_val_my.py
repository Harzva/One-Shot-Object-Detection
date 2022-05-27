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
import glob
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from harzvatool.Tools import Tools
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler,SequentialSampler
from torch.nn.parallel import DistributedDataParallel
from roi_data_layer.roidb1 import combined_roidb
from roi_data_layer.roibatchLoader2 import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient
from test_net_1_class import Testtool
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from test_net_1_fun import say
'''
    python trainval_net_gai2.py \
    --dataset coco --net res50 \
    --bs 6 --nw 32 \
    --cuda --g 1 --seen 1 --mGPUs

a

'''
# def str2bool(str):
#     return True if str.lower() == 'true' else False

def parse_args():
    """ 
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='voc', type=str)
    # parser.add_argument('--dataset', dest='dataset',
    #                     help='training dataset',
    #                     default='coco', type=str)
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
                        help='directory to save models', default="./models/res50/test/",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=32, type=int)
    # parser.add_argument('--cuda', dest='cuda',
    #                     help='whether use CUDA',
    #                     action='store_true')

    parser.add_argument('--word_embedding', dest='word_embedding', type=Tools.str2bool,
                        help='whether use multiple GPUs',
                        default='True')
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
    # parser.add_argument('--r', dest='resume',
    #                     help='resume checkpoint or not',
    #                     default='/home/ubuntu/Dataset/Partition1/hzh/lj/One-Shot-Object-Detection/models/res50/coco/cls_coco_bs16_s1_g1/cls_1_3_13311.pth', type=str)
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
                        default='/home/ubuntu/Dataset/Partition1/hzh/lj/One-Shot-Object-Detection/data/pre-trained/pretrain_imagenet_resnet50/model_best.pth.tar', type=str)
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
    # val = False
    debuging= True if 'test' in args.save_dir else False

    # print('Called with args:')
    # print(args) pascal_voc

    if args.dataset == "voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "voc_0712":
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
    # pprint.pprint(cfg)
    


    output_dir = args.save_dir+f'_{args.dataset}_bs{args.batch_size}_s{args.session}_g{args.group}'
    if not args.word_embedding:
        output_dir=output_dir+'_noword'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        if 'test' in args.save_dir:
            log_file=f"logs/test/{args.dataset}/{os.path.basename(output_dir)}/{os.path.basename(output_dir)}.log"
        
        else:
            log_file=f"logs/{args.dataset}/{os.path.basename(output_dir)}/{os.path.basename(output_dir)}.log"
        Tools.makedirs(log_file)
        logger_summary = SummaryWriter(log_dir=os.path.dirname(log_file))

    logger =Tools.get_logger(os.path.basename(output_dir), log_file=log_file,mode='a')
    # np.random.seed(cfg.RNG_SEED)
    # cfg.GPU_ID=args.GPU_ID  if args.GPU_ID  else list(range(torch.cuda.device_count())) 
    cfg['CUDA']=True
    logger.info('Using config:')
    pprint.pprint(cfg)
    logger.info(cfg)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available() and not args.cuda:
        logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
    logger.info('Called with args:')
    logger.info(args)
    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    
    cfg.USE_GPU_NMS = args.cuda
    logger.info('*'*60)
    logger.info(f'your output_dir is output_dir{output_dir}')
    # create dataloader
    # ===============
    # 获取词向量
    wi = {}
    with open('./cls_names.txt') as f:
      for i, key in enumerate(f.readlines()):
        wi[key.strip()] = i


    def demo(args,seen,batch_size,training):
        cfg.TRAIN.USE_FLIPPED = training
        imdb, roidb, ratio_list, ratio_index, query, class_to_name = combined_roidb(args.imdb_name, training, seen)
        logger.info('class to name', class_to_name)
        train_size = len(roidb)#212996
        logger.info('{:d} roidb entries'.format(len(roidb)))
        sampler_batch = sampler(train_size, batch_size)
        wi = {}
        with open('/home/ubuntu/Dataset/Partition1/hzh/lj/One-Shot-Object-Detection/cls_names.txt') as f:
            for i, key in enumerate(f.readlines()):
                wi[key.strip()] = i

        if seen==1:
            dataset = roibatchLoader(roidb, ratio_list, ratio_index, query, batch_size,
                                imdb.num_classes, training=training, class_to_name=class_to_name, word_name_to_index=wi)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        sampler=sampler_batch, num_workers=args.num_workers)
        elif seen==2:
            dataset_vu = roibatchLoader(roidb, ratio_list, ratio_index, query, batch_size,
                                imdb.num_classes, training=training, class_to_name=class_to_name, word_name_to_index=wi)
            dataloader = torch.utils.data.DataLoader(dataset_vu, batch_size=1,shuffle=False, num_workers=0,pin_memory=True)
        return dataloader,class_to_name,train_size,imdb,ratio_index
    train_dataloader,class_to_name,train_size,imdb,_=demo(args,seen=1,batch_size=args.batch_size,training=True)

    # test_dataloader,class_to_name,test_size,imdb_vu,ratio_index_vu=demo(args,seen=2,batch_size=1,training=False)
    # imdb_vu.competition_mode(on=True)
    # ==========
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
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic,word_embedding=args.word_embedding,pre_trained_path=args.pre_trained_path)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes,101, pretrained=True,
                            class_agnostic=args.class_agnostic,word_embedding=args.word_embedding,pre_trained_path=args.pre_trained_path)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True,
                            class_agnostic=args.class_agnostic,word_embedding=args.word_embedding,pre_trained_path=args.pre_trained_path)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=True,
                            class_agnostic=args.class_agnostic,word_embedding=args.word_embedding,pre_trained_path=args.pre_trained_path)
    else:
        logger.info("network is not defined")
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
        logger.info("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        logger.info("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    iters_per_epoch = int(train_size / args.batch_size)
    name_list = []
    index_list = []
    for name, index in class_to_name.items():
        name_list.append(name)
        index_list.append(index)
    loss_list=[]
    map_acc_list=[]
    if args.dataset=='voc':
        args.disp_interval=args.disp_interval
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma
        data_iter = iter(train_dataloader)

        for step in range(iters_per_epoch):
            if 'test' in args.save_dir:
                if step>=10:
                    continue
            data = next(data_iter)
            im_data.resize_(data[0].size()).copy_(data[0])
            query.resize_(data[1].size()).copy_(data[1])
            im_info.resize_(data[2].size()).copy_(data[2])
            gt_boxes.resize_(data[3].size()).copy_(data[3])
            num_boxes.resize_(data[4].size()).copy_(data[4])
            query_index = data[5]
            fasterRCNN.zero_grad()#tensor(0.1040, device='cuda:0', grad_fn=<MeanBackward0>)
            rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, margin_loss, RCNN_loss_bbox, \
                rois_label, _ = fasterRCNN(im_data, query, im_info, gt_boxes, num_boxes, query_index)  #(self, im_data, query, im_info, gt_boxes, num_boxes, query_word_idx):
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
            + RCNN_loss_cls.mean() + margin_loss.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()###step in val loss =0 mgpu

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

                logger.info("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e"
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))

                logger.info("\t\t\tfg/bg=(%d/%d), time cost: %f" %
                      (fg_cnt, bg_cnt, end-start))
                      
                logger.info("\t\t\trpn_cls: %.3f, rpn_box: %.3f, rcnn_cls: %.3f, margin: %.3f, rcnn_box %.3f"
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_margin, loss_rcnn_box))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_margin': loss_margin,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    info2 = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_margin': loss_margin,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger_summary.add_scalars(
                        "logs_s_step{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)
                    logger_summary.add_scalars(
                        "logs_s_epcoh{}/losses".format(args.session), info2, epoch)

                    logger_summary.add_scalar('lr', lr, (epoch - 1) * iters_per_epoch + step)
                    logger_summary.add_scalar('loss_rpn_cls', loss_rpn_cls, (epoch - 1) * iters_per_epoch + step)
                    logger_summary.add_scalar('loss_rpn_box', loss_rpn_box, (epoch - 1) * iters_per_epoch + step)
                    logger_summary.add_scalar('loss_rcnn_cls', loss_rcnn_cls, (epoch - 1) * iters_per_epoch + step)
                    logger_summary.add_scalar('loss_margin', loss_margin, (epoch - 1) * iters_per_epoch + step)
                    logger_summary.add_scalar('loss_rcnn_box', loss_rcnn_box, (epoch - 1) * iters_per_epoch + step)
                save_name_temp=os.path.join(output_dir, 'step_{}_{}_{}_{}.pth'.format(os.path.basename(output_dir),epoch, step,round(loss_temp,3)))
                tem_epoch_loss=round(loss_temp,3)

                if step==0:
                    save_checkpoint(fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(), save_name_temp)
                loss_list.append(loss_temp)
                if loss_temp<=min(loss_list):
                    for rm_path in glob.glob(f"{output_dir}/step*.pth"):
                        os.remove(rm_path)
                    logger.info('save  step model: {}'.format(save_name_temp))
                    save_checkpoint(fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(), save_name_temp)
                loss_temp = 0
                start = time.time()
        if epoch% 1== 0:
            test_tool=Testtool() #mgpu 则不能zip zip argument #1 must support iteration
            load_name=os.path.join(output_dir, 'val_{}_{}_{}.pth'.format(os.path.basename(output_dir),epoch,round(loss_temp,3)))
            save_checkpoint(fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(), load_name)
            map_acc,ap_dict,rec_dict,prec_dict=test_tool.math_test(load_name=load_name,trainval=True,debuging=debuging)
            map_acc_list.append(map_acc)
            if map_acc<max(map_acc_list):
                os.remove(load_name)
                logger.info(f'remove {load_name}')
                pass
            for cls,ap in ap_dict.items():
                logger_summary.add_scalar(f'{cls}', ap, epoch)
            # logger_summary.add_scalar('rec_dict', rec_dict, range(len(rec_dict)))
            # logger_summary.add_scalar('prec_dict', prec_dict,range(len(prec_dict)))
            end = time.time()
            logger.info("test time: %0.4fs" % (end - start))
        save_name = os.path.join(
            output_dir, '{}_{}_{}_{}_{}.pth'.format(os.path.basename(args.save_dir),args.session, epoch, step,tem_epoch_loss))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        logger.info('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger_summary.close()


'''

        # args.val_freq =args.disp_interval*100
        # if ((epoch - 1) * iters_per_epoch + step )% args.val_freq == 0 and ((epoch - 1) * iters_per_epoch + step)!=0:
        # if epoch% 1== 0:
        #     test_tool=Testtool(fasterRCNN) #mgpu 则不能zip zip argument #1 must support iteration
        #     # map_acc,ap_dict,rec_dict,prec_dict=test_tool.math_test(load_name='models/res50/voc/cls_pascal_voc_bs16_s1_g1/step_cls_pascal_voc_bs16_s1_g1_10_30_0.437.pth')
        #     load_name=os.path.join(output_dir, 'val_{}_{}_{}_{}.pth'.format(os.path.basename(output_dir),epoch, step,round(loss_temp,3)))
        #     map_acc,ap_dict,rec_dict,prec_dict=test_tool.math_test(load_name=load_name,net=fasterRCNN)
        #     map_acc_list.append(map_acc)
        #     if map_acc>max(map_acc_list):
        #         save_checkpoint(fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(), load_name)
        #     for cls,ap in ap_dict.items():
        #         logger_summary.add_scalar(f'{cls}', ap, (epoch - 1) * iters_per_epoch + step)
        #     # logger_summary.add_scalar('rec_dict', rec_dict, range(len(rec_dict)))
        #     # logger_summary.add_scalar('prec_dict', prec_dict,range(len(prec_dict)))
        #     end = time.time()
        #     print("test time: %0.4fs" % (end - start))


            # if rpn_loss_cls==0:
            #     rpn_loss_cls=torch.tensor(float(rpn_loss_cls)).cuda().requires_grad_(True)
            #     rpn_loss_box=torch.tensor(float(rpn_loss_box)).cuda().requires_grad_(True)
            #     margin_loss=torch.tensor(float(margin_loss)).cuda().requires_grad_(True) 
            #     RCNN_loss_bbox=torch.tensor(float( RCNN_loss_bbox)).cuda().requires_grad_(True)
            #     RCNN_loss_cls=torch.tensor(float(RCNN_loss_cls)).cuda().requires_grad_(True)
            #     loss =rpn_loss_box+RCNN_loss_cls+margin_loss+RCNN_loss_bbox+RCNN_loss_cls
            # else:



'''
