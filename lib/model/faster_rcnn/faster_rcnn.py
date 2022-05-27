import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
import numpy as np
from model.roi_layers import ROIAlign, ROIPool
# from model.roi_crop.modules.roi_crop import _RoICrop
# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.utils.net_utils import *

class match_block(nn.Module):
    def __init__(self, inplanes):
        super(match_block, self).__init__()

        self.sub_sample = False

        self.in_channels = inplanes
        self.inter_channels = None

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.Q = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.Q[1].weight, 0)
        nn.init.constant_(self.Q[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )
        
        self.ChannelGate = ChannelGate(self.in_channels)
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)


        
    def forward(self, detect, aim):

        

        batch_size, channels, height_a, width_a = aim.shape
        batch_size, channels, height_d, width_d = detect.shape


        #####################################find aim image similar object ####################################################

        d_x = self.g(detect).view(batch_size, self.inter_channels, -1)
        d_x = d_x.permute(0, 2, 1).contiguous()

        a_x = self.g(aim).view(batch_size, self.inter_channels, -1)
        a_x = a_x.permute(0, 2, 1).contiguous()

        theta_x = self.theta(aim).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(detect).view(batch_size, self.inter_channels, -1)

        

        f = torch.matmul(theta_x, phi_x)

        N = f.size(-1)
        f_div_C = f / N

        f = f.permute(0, 2, 1).contiguous()
        N = f.size(-1)
        fi_div_C = f / N

        non_aim = torch.matmul(f_div_C, d_x)
        non_aim = non_aim.permute(0, 2, 1).contiguous()
        non_aim = non_aim.view(batch_size, self.inter_channels, height_a, width_a)
        non_aim = self.W(non_aim)
        non_aim = non_aim + aim

        non_det = torch.matmul(fi_div_C, a_x)
        non_det = non_det.permute(0, 2, 1).contiguous()
        non_det = non_det.view(batch_size, self.inter_channels, height_d, width_d)
        non_det = self.Q(non_det)
        non_det = non_det + detect

        ##################################### Response in chaneel weight ####################################################

        c_weight = self.ChannelGate(non_aim)
        act_aim = non_aim * c_weight  #torch.Size([8, 1024, 8, 8])* torch.Size([8, 1024, 1, 1])=torch.Size([8, 1024, 8, 8])    self.globalAvgPool ---> torch.Size([8, 1024, 1, 1])
        act_det = non_det * c_weight  #torch.Size([8, 1024, 38, 57])** torch.Size([8, 1024, 1, 1])=torch.Size([8, 1024, 38, 57])  non_det=rpn_fet act=coae

        return non_det, act_det, act_aim, c_weight

def read_weight(data,path_word_w2v):
    with open(path_word_w2v) as f:
        for row, line in enumerate(f.readlines()):
            lines = [float(i) for i in line.strip().split(',')]
            for col, value in enumerate(lines):
                data[row,col] = value
    return data.T
def read_weight_voc(data,path_word_w2v):
    with open(path_word_w2v) as f:
        for row, line in enumerate(f.readlines()):
            lines =  [float(i) for i in line.split(' ')[1:]]
            for col, value in enumerate(lines):
                data[row,col] = value
    return data
class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic,word_embedding):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.word_embedding=word_embedding

        
        self.match_net = match_block(self.dout_base_model)

        
        # if len(self.classes)==81:
        #     self.word_embs = nn.Embedding(1,500)
        #     self.trans = nn.Linear(500,500)
        #     self.trans1= nn.Linear(500 + 2048, 2048)

        if len(self.classes)==81:
            self.word_embs = nn.Embedding(len(self.classes)-1,500)
            self.trans = nn.Linear(500,500)
            self.trans1= nn.Linear(500 + 2048, 2048)
        # if len(self.classes)==81:
        #     self.word_embs = nn.Embedding(201,500)
        #     path_word_w2v='./word_w2v.txt'
        #     temp_data= np.zeros((500,201),dtype=np.float32)
        #     self.temp_word=read_weight(temp_data,path_word_w2v) 
        #     self.word_embs.weight.data.copy_(torch.from_numpy(self.temp_word))
        #     self.trans = nn.Linear(500,500)
        #     self.trans1= nn.Linear(500 + 2048, 2048)
        #     self.word_embs.weight.requires_grad=False
        elif len(self.classes)==21:
            self.word_embs = nn.Embedding(20,300)
            # path_word_w2v='./glove.42B.300d_voc.txt'
            # temp_data= np.zeros((20,300),dtype=np.float32)
            # temp_word=read_weight_voc(temp_data,path_word_w2v) 
            # self.word_embs.weight.data.copy_(torch.from_numpy(temp_word))
            self.trans = nn.Linear(300,300)
            self.trans1= nn.Linear(300 + 2048, 2048)

        
        # self.word_embs = nn.Embedding(201,500) if len(self.classes)==81 else nn.Embedding(21,500)
        # path_word_w2v='./word_w2v.txt' if len(classes)==81 else './glove.42B.300d_voc.txt'
        # temp_data=data = np.zeros((500,201),dtype=np.float32) if len(classes)==81 else np.zeros((300,20),dtype=np.float32) 
        # temp_word=read_weight(temp_data,path_word_w2v) 
        # self.word_embs.weight.data.copy_(torch.from_numpy(temp_word))

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        # self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        # self.RCNN_roi_crop = _RoICrop()
        # ==================
        # 语义视觉融合模块


        # ====================
        self.triplet_loss = torch.nn.MarginRankingLoss(margin = cfg.TRAIN.MARGIN)

    def forward(self, im_data, query, im_info, gt_boxes, num_boxes, query_word_idx):
        batch_size = im_data.size(0)
        #print("batch_size:",batch_size)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        query_word_idx = query_word_idx.cuda()

        query_word_vector = self.word_embs(query_word_idx)
        # query_word_vector = self.word_embs(torch.tensor(0).cuda()).unsqueeze(0).repeat(batch_size,1)#201 里面去自己想要的类 3 300  kanakn zheli当voc的时候you没够取过21  为啥不减一呢
        # if 200 in query_word_idx:
        #     raise RuntimeError("Something bad happend")
        # if self.word_embs.weight.data.cpu().equal(torch.from_numpy(self.temp_word)):
        #     a=self.word_embs.weight.data.cpu()-torch.from_numpy(self.temp_word)
        #     print("word_embs相同--------------------")
        #     print("word_embs相减",a)

        # else:
        #     print("word_embs-----butong")
        # print((self.word_embs.weight.data.cpu()-torch.from_numpy(self.temp_word)).sum(dim=1))

        # feed image data to base model to obtain base feature map
        detect_feat = self.RCNN_base(im_data) #torch.Size([3, 1024, 38, 57])=torch.Size([3, 3, 600, 908])   
        query_feat = self.RCNN_base(query)

        rpn_feat, act_feat, act_aim, c_weight = self.match_net(detect_feat, query_feat)


        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(rpn_feat, im_info, gt_boxes, num_boxes)


        # if it is training phrase, then use ground trubut bboxes for refining



        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            margin_loss = 0
            rpn_loss_bbox = 0
            score_label = None

        rois = Variable(rois)
        # do roi pooling based on predicted rois
        #print('rois_label:',rois_label)


        # if cfg.POOLING_MODE == 'crop':
        #     # pdb.set_trace()
        #     # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
        #     grid_xy = _affine_grid_gen(rois.view(-1, 5), act_feat.size()[2:], self.grid_size)
        #     grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
        #     pooled_feat = self.RCNN_roi_crop(act_feat, Variable(grid_yx).detach())
        #     if cfg.CROP_RESIZE_WITH_MAX_POOL:
        #         pooled_feat = F.max_pool2d(pooled_feat, 2, 2)

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(act_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(act_feat, rois.view(-1,5))
        else:
            raise ValueError(f"you got mode is{cfg.POOLING_MODE} but must align or pool  or crop")

        '''
        def _head_to_tail(self, pool5):
            fc7 = self.RCNN_top(pool5).mean(3).mean(2)
            return fc7




        # Build resnet.
        self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
        resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

        self.RCNN_top = nn.Sequential(resnet.layer4)


        self.RCNN_cls_score = nn.Sequential(
                            nn.Linear(2048*2, 8),
                            nn.Linear(8, 2)
                            )
        '''
        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)  #torch.Size([1024, 2048])=torch.Size([1024, 1024, 7, 7])  1024/8=128    8=batchsize
        query_feat  = self._head_to_tail(act_aim)  #torch.Size([8, 2048])=torch.Size([8, 1024, 8, 8])  里面的两次mean算作全局平均池化了

        # ===========
        # 不更新语义向量直接和视觉向量级联--no work

        if self.word_embedding:
            new_query_word_vector = self.trans(query_word_vector)#更新query  .detach()停止反向传播
            query_feat = torch.cat((query_feat,new_query_word_vector),dim=1)
            query_feat = self.trans1(query_feat)  #融合层
        # =============

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)


        pooled_feat = pooled_feat.view(batch_size, rois.size(1), -1)#torch.Size([8, 128, 2048])
        query_feat = query_feat.unsqueeze(1).repeat(1,rois.size(1),1)#torch.Size([8, 128, 2048])
        

        
        
        
        # print(pooled_feat.shape)
        # print(query_feat.shape)
        # print(new_query_word_vector.shape)

        #分别计算与词向量的距离



        pooled_feat = torch.cat((pooled_feat,query_feat,), dim=2).view(-1, 2*2048)   #torch.Size([1024, 4096])
        #pooled_feat = torch.cat((pooled_feat,new_query_word_vector),dim=1)
        #print("pooled_feat.shape",pooled_feat.shape)

        # compute object classification probability
        score = self.RCNN_cls_score(pooled_feat)   #度量模块
        '''
            self.RCNN_cls_score = nn.Sequential(
                            nn.Linear(2048*2, 8),
                            nn.Linear(8, 2)
                            )
        '''

        score_prob = F.softmax(score, 1)[:,1]


        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        

        if self.training:
            # classification loss
            
            score_label = rois_label.view(batch_size, -1).float()
            gt_map = torch.abs(score_label.unsqueeze(1)-score_label.unsqueeze(-1))

            score_prob = score_prob.view(batch_size, -1)
            pr_map = torch.abs(score_prob.unsqueeze(1)-score_prob.unsqueeze(-1))
            target = -((gt_map-1)**2) + gt_map
            
            RCNN_loss_cls = F.cross_entropy(score, rois_label)#torch.Size([384, 2]) torch.Size([384])   
            #gaigaigiagiagiagi
            #RCNN_loss_cls = RCNN_loss_cls + sp_score.int().item() * RCNN_loss_cls + sq_score.int().item() * RCNN_loss_cls

            margin_loss = 3 * self.triplet_loss(pr_map, gt_map, target)

            # RCNN_loss_cls = similarity + margin_loss
    
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = score_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, margin_loss, RCNN_loss_bbox, rois_label, c_weight

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score[0], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score[1], 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
