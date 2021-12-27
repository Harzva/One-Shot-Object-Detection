from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
import copy
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class resnet(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False,word_embedding=True,model_path=None):
    if num_layers==50:
      # self.model_path = '/home/ubuntu/Dataset/Partition1/hzh/lj/One-Shot-Object-Detection-master/data/pre-trained/pretrain_imagenet_resnet50/model_best.pth.tar'
      self.model_path =model_path# '/home/ubuntu/Dataset/Partition1/hzh/lj/One-Shot-Object-Detection-master/data/pre-trained/coco_resent50/resnet50.pth'
      # self.model_path = '/home/ubuntu/Dataset/Partition1/hzh/lj/One-Shot-Object-Detection-master/data/pre-trained/coco_resent50/checkpoint.pth'
      #self.model_path = '../data/pretrain_imagenet_resnet50/model_best.pth.tar'
    elif num_layers==101:
      self.model_path = '/home/ubuntu/Dataset/Partition1/hzh/lj/One-Shot-Object-Detection-master/data/pre-trained/pretrain_imagenet_resnet101/model_best.pth.tar'
    else:
      raise ValueError("50  or 101")
      #self.model_path = '../data/pretrain_imagenet_resnet101/model_best.pth.tar'
    self.dout_base_model = 1024
    self.pretrained = pretrained

    self.class_agnostic = class_agnostic
    self.num_layers = num_layers
    super().__init__(classes, class_agnostic,word_embedding)
    # _fasterRCNN.__init__(self,classes, class_agnostic)#qudiao model_path

  def _init_modules(self):
    if self.num_layers==50:
      resnet = resnet50()
    else:
      resnet = resnet101()

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      if "BT" not in self.model_path:
        state_dict = state_dict['state_dict']
      # with open('/home/ubuntu/Dataset/Partition1/hzh/lj/One-Shot-Object-Detection-master/resnet50.txt',"w") as f:
      #   for key, value in state_dict.items():
      #     f.write(f"{key}\t {value.shape}\n")  
      #OrderedDict([('conv1.weight', tensor([[[[-1.5844e-...='cuda:0')), ('bn1.weight', tensor([ 3.4324,  2....='cuda:0')), ('bn1.bias', tensor([ 3.8206,  1....='cuda:0')), ('bn1.running_mean', tensor([ 0.0862, -0....='cuda:0')), ('bn1.running_var', tensor([0.0623, 0.01...='cuda:0')), ('bn1.num_batches_tracked', tensor(307644, devic...='cuda:0')), ('layer1.0.conv1.weight', tensor([[[[-0.2754]]...='cuda:0')), ('layer1.0.bn1.weight', tensor([ 1.9264,  0....='cuda:0')), ('layer1.0.bn1.bias', tensor([-2.2319,  5....='cuda:0')), ('layer1.0.bn1.running_mean', tensor([ -1.5654,   ...='cuda:0')), ('layer1.0.bn1.running_var', tensor([ 29.2059,  3...='cuda:0')), ('layer1.0.bn1.num_bat...es_tracked', tensor(307644, devic...='cuda:0')), ('layer1.0.conv2.weight', tensor([[[[ 0.0483, ...='cuda:0')), ('layer1.0.bn2.weight', tensor([ 5.8823e+00,...='cuda:0')), ...])


      state_dict_v2 = copy.deepcopy(state_dict)

      for key in state_dict:
        if "BT" not in self.model_path:
          pre, post = key.split('module.')
        else:
          post=key
        state_dict_v2[post] = state_dict_v2.pop(key)
        """
        module.fc.weight	 torch.Size([1000, 2048])
        module.fc.bias	 torch.Size([1000])
        """
      # alisure
      state_dict_v2['fc.weight']=torch.randn(1000, 2048)
      state_dict_v2['fc.bias']=torch.randn(1000)

      resnet.load_state_dict(state_dict_v2)

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    '''
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    '''
    self.RCNN_cls_score = nn.Sequential(
                          nn.Linear(2048*2, 8),
                          nn.Linear(8, 2)####改成一层试试看
                          )


    # self.RCNN_cls_score =nn.Linear(2048*2, 2)####改成一层试试看
                          

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4)####只需要目标有四个坐标就可以
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

    # for num_layer,child in enumerate(self.RCNN_base.children()):
    #   for param in child.parameters():
    #     param.requires_grad = False

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[5].train()
      self.RCNN_base[6].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)#torch.Size([1024, 1024, 7, 7])---》torch.Size([1024, 2048, 4, 4])--> torch.Size([1024, 2048, 4])-->torch.Size([1024, 2048])
    return fc7
