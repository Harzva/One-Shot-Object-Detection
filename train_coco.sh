# CUDA_VISIBLE_DEVICES=0,1 python trainval_net_gai3.py \
# --dataset coco --net res50 \
# --bs 16 --nw 8 \
# --cuda True --g 1 --seen 1   --mGPUs True --save_dir ./models/res50/coco/cls_word_embedding_normal  --word_embedding True 

# CUDA_VISIBLE_DEVICES=0,1 python trainval_net_gai3.py \
# --dataset coco --net res50 \
# --bs 16 --nw 8 \
# --cuda True --g 2 --seen 1   --mGPUs True --save_dir ./models/res50/coco/cls --word_embedding False 
# CUDA_VISIBLE_DEVICES=0,1 python trainval_net_gai3.py \
# --dataset coco --net res50 \
# --bs 16 --nw 8 \
# --cuda True --g 3 --seen 1   --mGPUs True --save_dir ./models/res50/coco/cls --word_embedding False 
# CUDA_VISIBLE_DEVICES=0,1 python trainval_net_gai3.py \
# --dataset coco --net res50 \
# --bs 16 --nw 8 \
# --cuda True --g 4 --seen 1   --mGPUs True --save_dir ./models/res50/coco/cls --word_embedding False 




# CUDA_VISIBLE_DEVICES=0,1 python trainval_net_gai3.py \
# --dataset coco --net res50 \
# --bs 16 --nw 8 \
# --cuda True --g 2 --seen 1   --mGPUs True --save_dir ./models/res50/coco/cls  --word_embedding True 
# CUDA_VISIBLE_DEVICES=0,1 python trainval_net_gai3.py \
# --dataset coco --net res50 \
# --bs 16 --nw 8 \
# --cuda True --g 3 --seen 1   --mGPUs True --save_dir ./models/res50/coco/cls  --word_embedding True 
# CUDA_VISIBLE_DEVICES=0,1 python trainval_net_gai3.py \
# --dataset coco --net res50 \
# --bs 16 --nw 8 \
# --cuda True --g 4 --seen 1   --mGPUs True --save_dir ./models/res50/coco/cls  --word_embedding True 

'''



screen -hzh_v1_80
CUDA_VISIBLE_DEVICES=0,1 python trainval_net_gai3.py \
--dataset coco --net res50 \
--bs 16 --nw 4 \
--cuda True --g 1 --seen 1   --mGPUs True --save_dir ./models/res50/coco/cls_word_embedding_normal_80 --word_embedding True 

CUDA_VISIBLE_DEVICES=0,1 python trainval_net_gai3.py \
--dataset coco --net res50 \
--bs 16 --nw 4 \
--cuda True --g 1 --seen 1   --mGPUs True --save_dir ./models/res50/coco/cls_word_embedding_normal_1 --word_embedding True 


CUDA_VISIBLE_DEVICES=2,3 python trainval_net_gai3.py \
--dataset coco --net res50 \
--bs 16 --nw 4 \
--cuda True --g 1 --seen 1   --mGPUs True --save_dir ./models/res50/coco/cls_lj_word --word_embedding True 



CUDA_VISIBLE_DEVICES=0 python trainval_net_gai3.py \
--dataset coco --net res50 \
--bs 8 --nw 4 \
--cuda True --g 1 --seen 1   --mGPUs True --save_dir ./models/res50/coco/cls_lj_word --word_embedding True 




baseline  
cls_coco_bs16_s1_g1                                     Coae(1,2,3,4)
cls_lj_word_coco_bs16_s1_g1                             lj()
cls_dynamic_word_coco_bs16_s1_g1                        lj_v2(1)
cls_dynamic_word_embedding_normal_80_coco_bs16_s1_g1    hzh_v1_80
cls_dynamic_word_embedding_normal_coco_bs16_s1_g1       hzh_v1(1,2,3,4)

BT_coco_bs16_s1_g1
'''
CUDA_VISIBLE_DEVICES=2,3 python trainval_net_gai3.py \
--dataset coco --net res50 \
--bs 16 --nw 8 \
--cuda True --g 1 --seen 1   --mGPUs True --save_dir ./models/res50/coco/cls_word_embedding_normal_8 --word_embedding True 

