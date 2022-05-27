for i in {1..10};do
	CUDA_VISIBLE_DEVICES=0 python test_write.py --dataset coco --net res50 --s 1 --checkepoch $i --p 26623 --cuda --g 1
done

# CUDA_VISIBLE_DEVICES=2 python test_net_1_class.py  --dataset coco --net res50 --cuda True --g 1 --seen 2   

# CUDA_VISIBLE_DEVICES=2 python test_net_1.py  --dataset coco --net res50 --cuda True --g 1 --seen 2   --load_dir /media/ubuntu/data1/hzh/One-Shot-Object-Detection/models/res50/coco/cls_coco_bs16_s1_g1/step_cls_coco_bs16_s1_g1_10_0_0.351.pth

 CUDA_VISIBLE_DEVICES=2 python test_net_1.py  --dataset voc_0712 --net res50 --cuda True --g 1 --seen 2   --load_dir /media/ubuntu/data1/hzh/One-Shot-Object-Detection/models/res50/voc/cls_nowordemb_pascal_voc_0712_bs16_s1_g1/step_cls_nowordemb_pascal_voc_0712_bs16_s1_g1_9_0_0.277.pth

 CUDA_VISIBLE_DEVICES=1 python test_net_1_class.py  --dataset coco --net res50 --cuda True --g 1 --seen 2   --load_dir /media/ubuntu/data1/hzh/One-Shot-Object-Detection/models/res50/coco/cls_alisure_epochs10_coco_bs16_s1_g1_noword/step_cls_alisure_epochs10_coco_bs16_s1_g1_noword_10_5140_0.289.pth