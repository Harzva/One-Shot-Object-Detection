# 验收


## 论文

Word版论文+PDF版论文


## 程序

每一章都按照以下格式书写
### 第1章

#### 1. Readme

* 本章方法介绍
     一种基于K-组合均值特征增强的小样本目标检测方法及系统，利用每类k个带标签样本的标签信息，得到对应的特征，通过对这k个目标的特征进行
Cm^K组合平均，为每个组合计算平均特征。然后将这些平均特征添加到原有特征样本集合中，增加特征样本的数量。在微调阶段，除了新类带标签样本对
分类和回归器进行微调外，还使用增加后的特征样本集合对分类器进行微调。实验结果证明通过基于K-组合均值的特征增强方法，通过增加特征样本的数
量，为分类器提供了更多的特征样本，缓解了模型的过拟合问题，提升了基于微调的小样本目标检测模型的检测精度。

* 本章各模块简要介绍
    特征提取模块：
        采用ResNet-50作为特征提取模块的骨干网络，用于提取查询图像与目标图像的特征
    RPN网络：
        RPN网络的目的是为了在查询图像中找到与目标图像中的目标属于同一类前景对象的候选框。通过RPN网络得到候选框位置后，再使用ROI Pooling
    得到每个候选框的固定大小的特征，方便后面与查询图像中的目标的特征进行比较。
    分类器：基于cosine相似性度量函数的分类器，输出候选框为某一类的概率
    回归器：输出候选框的位置信息



#### 2. 数据
COCO数据集：取COCO数据集中的20类为新类，其余作为基类
* 训练所用数据
    第一阶段：基类
    第二阶段——微调：新类
* 测试所用数据
    新类、基类


#### 3. 环境
* 系统
   * Ubuntu 16.04
* Python包
    * Linux with Python >= 3.6
    * [PyTorch](https://pytorch.org/get-started/locally/) >= 1.3
    * [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
    * Dependencies: ```pip install -r requirements.txt```
    * pycocotools: ```pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'```
    * [fvcore](https://github.com/facebookresearch/fvcore/): ```pip install 'git+https://github.com/facebookresearch/fvcore'```
    * [OpenCV](https://pypi.org/project/opencv-python/), optional, needed by demo and visualization ```pip install opencv-python```
    * GCC >= 4.9
* 编译
    python setup.py build develop



#### 4. 代码说明
* 目录结构
- **configs**: 配置文件
- **datasets**: Dataset files (see [Data Preparation](#data-preparation) for more details)
- **fsdet**
  - **checkpoint**: Checkpoint code.
  - **config**: Configuration code and default configurations.
  - **data**: 数据集处理代码
  - **engine**: Contains training and evaluation loops and hooks.
  - **evaluation**: Evaluation code for different datasets.
  - **layers**: Implementations of different layers used in models.
  - **modeling**: Code for models, including backbones, proposal networks, and prediction heads.
  - **solver**: Scheduler and optimizer code.
  - **structures**: Data types, such as bounding boxes and image lists.
  - **utils**: Utility functions.
- **tools**
  - **train_net.py**: Training script.
  - **test_net.py**: Testing script.
  - **ckpt_surgery.py**: Surgery on checkpoints.
  - **run_experiments.py**: Running experiments across many seeds.
  - **aggregate_seeds.py**: Aggregating results from many seeds.
* 核心模块代码说明
    few-shot-object-detection-master\fsdet\modeling\roi_heads\roi_heads.py中：
    class StandardROIHeads1(ROIHeads)：
    def _forward_box_ground_truth：求组合平均
    def _forward_box_an：添加到已知类别的特征集合中



#### 5. 模型

* 预训练模型
    few-shot-object-detection-master\fsdet\model_zoo
* 训练完成后的模型
    few-shot-object-detection-master\checkpoints\coco\faster_rcnn\


#### 6. 训练

* 训练超参数
    K值的设置，可以取1、3、5、10、30

* 运行训练
    python tools/train_net.py --num-gpus 1 \
        --config-file few-shot-object-detection-master\configs\COCO-detection\faster_rcnn_R_101_FPN_base.yaml
    更改保存训练后的输出结果：
    few-shot-object-detection-master\configs\COCO-detection文件夹中对应文件的输出路径


#### 7. 测试

* 评估指标说明
    模型在数据集上的目标检测精度mAP
* 运行测试
   python demo/demo.py --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS fsdet://coco/tfa_cos_1shot/model_final.pth
  config-file：few-shot-object-detection-master\configs\COCO-detection\：微调时更新全部模型的或只更新分类器和回归器
  MODEL.WEIGHTS：few-shot-object-detection-master\checkpoints\coco\faster_rcnn\faster_rcnn_R_101_FPN_ft_novel_3shotgai4



#### 8. 推理
* 运行推理



## 其他

### 第2章

#### 1. Readme

* 本章方法介绍
     一种基于目标互换和度量学习的小样本目标检测方法，利用基类中目标图像与查询图像已有的标签信息，将查询图像与目标图像中的同类目标进行
互换，构成新的查询-目标图像对作为输入。通过这种数据增强方式增加了目标图像中各目标之间的对比，为相似性度量模块提供了更多的可对比样本，
减小了目标图像中其他类目标或背景对于待检测目标的影响，提升了基于度量学习的小样本目标检测模型在基类与新类上的检测精度。
* 本章各模块简要介绍
    基于目标互换的数据增强模块：
        在基类上训练时，利用查询-目标图像对的标签信息，将查询-目标图像对中的同类目标进行互换，构成新的查询-目标图像对作为训练样本
    特征提取模块：
        采用ResNet-50作为特征提取模块的骨干网络，用于提取查询图像与目标图像的特征
    RPN网络：
        RPN网络的目的是为了在查询图像中找到与目标图像中的目标属于同一类前景对象的候选框。通过RPN网络得到候选框位置后，再使用ROI Pooling
    得到每个候选框的固定大小的特征，方便后面与查询图像中的目标的特征进行比较。
    度量模块：
        使用两层的MLP网络，并以softmax二分类为结尾。度量每个候选框与查询图像中目标特征之间的相似性，输出两者间的相似度，保留相似度高
    的候选框作为检测结果。



#### 2. 数据
COCO数据集：将COCO数据集划分为4组，选其中一组作为新类，则剩余三组为基类
* 训练所用数据
    基类
* 测试所用数据
    新类


#### 3. 环境
* 系统
    * Ubuntu 16.04
* 软件环境
    * Python or 3.6
    * Pytorch 1.0
* Python包
    pip install -r requirements.txt
    编译：
    cd lib
    python setup.py build develop



#### 4. 代码说明
* 目录结构
    cfgs:网络结构的配置文件
    data：预训练模型、coco所涉及的图像（用于获取得到查询图像）
    images：总体框架图
    lib:
        build:coco数据集所需的一些包等
        dataset：对于不同数据集的一些处理操作
        faster_rcnn.egg-info:faster_rcnn的一些信息
        model:模型的各个模块
        pycocotools:coco数据集的操作文件夹
        roi_data_layer：roi所涉及的一些操作文件夹
    logs:模型训练时的一些日志文件
    models:训练后保留的某些模型
    output：测试时模型的结果
    trainval_net:训练文件
    test_net:测试文件
* 核心模块代码说明
One-Shot-Object-Detection-master\lib\roi_data_layer\roibatchLoader1.py中:
def __getitem__(self, index)方法的272-303行添加的查询-目标图像互换的处理部分实现


#### 5. 模型

* 预训练模型
    ResNet50:
    One-Shot-Object-Detection-master\data\pre-trained\pretrain_imagenet_resnet50
* 训练完成后的模型
    One-Shot-Object-Detection-master\models\res50\coco
    如：One-Shot-Object-Detection-master\models\res50\coco\faster_rcnn_1_1_354981.pth


#### 6. 训练

* 训练超参数

* 运行训练
    python trainval_net.py \
    --dataset coco --net res50 \
    --bs $BATCH_SIZE --nw $WORKER_NUMBER \
    --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
    --cuda --g $SPLIT --seen $SEEN
    #--nw 一般设置为1或2
    #--bs 一般为6或8，否则跑不起来
    #--lr、--lr_decay_step 直接用的默认的
    #--g $SPLIT (1~4)将第几组数据作为新类，其余3类为基类
    "*--seen*". 训练时选1
    * 1 --> Training, session see train_categories(config file) class
    * 2 --> Testing, session see test_categories(config file) class
    * 3 --> session see train_categories + test_categories class
    若取消目标互换时则运行trainval_net_yuan.py

#### 7. 测试

* 评估指标说明
    模型在新类数据集上的目标检测精度mAP
* 运行测试
    python test_net.py --dataset coco --net res50 \
                       --s $SESSION --checkepoch $EPOCH --p $CHECKPOINT \
                       --cuda --g $SPLIT
    例如：
    SESSION=1, EPOCH=10, CHECKPOINT=1663



#### 8. 推理
* 运行推理



## 其他



### 第3章

#### 1. Readme

* 本章方法介绍
    一种基于语义特征和度量学习的小样本目标检测方法及系统，将查询图对应的类别语义名称作为知识，使用自然语言处理领域的word2vec工具计算
对应的词向量作为语义特征，并与查询图的图像特征进行融合。通过将语义特征嵌入视觉域中，利用同类别目标在语义空间中具有的语义一致性，减少同
类别目标间的距离，利用不同语义类别目标在语义空间中具有的语义差异性，增加不同类目标之间的距离，缓解现有基于度量学习的小样本目标检测模型
出现的不同类别但外观视觉较相似目标的错检问题和相同语义类别但外观视觉差异较大的漏检的问题，提升在基类与新类上的检测精度。

* 本章各模块简要介绍
    特征提取模块：
        采用ResNet-50作为特征提取模块的骨干网络，用于提取查询图像与目标图像的特征
    RPN网络：
        RPN网络的目的是为了在查询图像中找到与目标图像中的目标属于同一类前景对象的候选框。通过RPN网络得到候选框位置后，再使用ROI Pooling
    得到每个候选框的固定大小的特征，方便后面与查询图像中的目标的特征进行比较。

    基于语义特征的特征对齐模块：
        将语义知识中所包含的类别信息传递到视觉域信息中。
    度量模块：
        使用两层的MLP网络，并以softmax二分类为结尾。度量每个候选框与查询图像中目标特征之间的相似性，输出两者间的相似度，保留相似度高
    的候选框作为检测结果。



#### 2. 数据

    COCO数据集：将COCO数据集划分为4组，选其中一组作为新类，则剩余三组为基类
* 训练所用数据
    基类
* 测试所用数据
    新类


#### 3. 环境
* 系统
    * Ubuntu 16.04
* 软件环境
    * Python or 3.6
    * Pytorch 1.0
* Python包
    pip install -r requirements.txt
    编译：
    cd lib
    python setup.py build develop



#### 4. 代码说明

* 目录结构
    cfgs:网络结构的配置文件
    data：预训练模型、coco所涉及的图像（用于获取得到查询图像）
    images：总体框架图
    lib:
        build:coco数据集所需的一些包等
        dataset：对于不同数据集的一些处理操作
        faster_rcnn.egg-info:faster_rcnn的一些信息
        model:模型的各个模块
        pycocotools:coco数据集的操作文件夹
        roi_data_layer：roi所涉及的一些操作文件夹
    logs:模型训练时的一些日志文件
    models:训练后保留的某些模型
    output：测试时模型的结果
    trainval_net:训练文件
    test_net:测试文件
    cls_names:类别名称
    word_w2v：类别名称对应的词向量

* 核心模块代码说明
    One-Shot-Object-Detection-master\lib\roi_data_layer\roibatchLoader2.py中：
        def __getitem__(self, index)方法的289行：读取当前数据的类别名称所对应的类别id，以获取对应词向量
    One-Shot-Object-Detection-master\lib\model\faster_rcnn_gai2\faster_rcnn.py中的
        127行：def read_weight():读取词向量
        170行：def forward()中的词向量与特征向量融合部分179-228行



#### 5. 模型

* 预训练模型
    ResNet50:
    One-Shot-Object-Detection-master\data\pre-trained\pretrain_imagenet_resnet50

* 训练完成后的模型
    One-Shot-Object-Detection-master\models\res50\coco
    如：One-Shot-Object-Detection-master\models\res50\coco\faster_rcnn_1_1_354982.pth



#### 6. 训练

* 训练超参数

* 运行训练
    python trainval_net_gai2.py \
    --dataset coco --net res50 \
    --bs $BATCH_SIZE --nw $WORKER_NUMBER \
    --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
    --cuda --g $SPLIT --seen $SEEN
    #--nw 一般设置为1或2
    #--bs 一般为6或8，否则跑不起来
    #--lr、--lr_decay_step 直接用的默认的
    #--g $SPLIT (1~4)将第几组数据作为新类，其余3类为基类
    "*--seen*". 训练时选1
    * 1 --> Training, session see train_categories(config file) class
    * 2 --> Testing, session see test_categories(config file) class
    * 3 --> session see train_categories + test_categories class
    训练前将faster_rcnn_gai2文件夹重命名为faster_rcnn
    保存模型时的命名的更改：
    trainval_net_gai2.py中458行左右

    python trainval_net_gai2.py \
    --dataset coco --net res50 \
    --bs 6 --nw 32 \
    --cuda --g 1 --seen 1 --mGPUs--------------------


    g=0



   CUDA_VISIBLE_DEVICES=0 python trainval_net_gai2.py \
    --dataset coco --net res50 \
    --bs 8 --nw 32 --pre_t data/pre-trained/BT_coco_resnet50/resnet50.pth\
      --g 1 --seen 1 --save_dir /home/ubuntu/Dataset/Partition1/hzh/lj/One-Shot-Object-Detection-master/models/res50/BT_coco_pre-coco

    CUDA_VISIBLE_DEVICES=1 python trainval_net_gai2.py \
    --dataset coco --net res50 \
    --bs 8 --nw 32  --pre_t data/pre-trained/BT_imagenet_resnet50/resnet50.pth \
    --g 1 --seen 1 --save_dir  /home/ubuntu/Dataset/Partition1/hzh/lj/One-Shot-Object-Detection-master/models/res50/BT_imagenet_pre-coco




    CUDA_VISIBLE_DEVICES=1 python trainval_net_gai2.py \
    --dataset coco --net res50 \
    --bs 8 --nw 32  --pre_t data/pre-trained/pretrain_imagenet_resnet50/model_best.pth.tar \
    --g 1 --seen 1 --save_dir  /home/ubuntu/Dataset/Partition1/hzh/lj/One-Shot-Object-Detection-master/models/res50/CLS_imagenet_pre-coco
#### 7. 测试

* 评估指标说明
    模型在新类数据集上的目标检测精度mAP
* 运行测试
    python test_net_1.py --dataset coco --net res50 \
                       --s $SESSION --checkepoch $EPOCH --p $CHECKPOINT \
                       --cuda --g $SPLIT
    例如：
    SESSION=1, EPOCH=10, CHECKPOINT=1663



#### 8. 推理
* 运行推理
    测试结果保存在新建的文件夹中


## 其他

