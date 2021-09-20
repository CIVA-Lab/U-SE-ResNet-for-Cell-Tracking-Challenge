# 2D
# python train.py --dataset BF-C2DL-HSC --erosion 5 --use_gold_truth True
# python train.py --dataset PhC-C2DL-PSC --erosion 5 --use_gold_truth True
python train.py --dataset PhC-C2DH-U373 --erosion 2 --use_gold_truth True
# python train.py --dataset BF-C2DL-MuSC --erosion 5 --use_gold_truth True
# python train.py --dataset DIC-C2DH-HeLa --erosion 20 --use_gold_truth True
# python train.py --dataset Fluo-C2DL-MSC --erosion 2 --resize_to 800 992 --use_gold_truth True
# python train.py --dataset Fluo-N2DL-HeLa --erosion 2 --use_gold_truth True
# python train.py --dataset Fluo-N2DH-GOWT1 --erosion 10 --use_gold_truth True

# 3D
# python train.py --dataset Fluo-C3DH-A549 --erosion 5 --resize_to 320 416 --train_resolution 320 416 --use_gold_truth True
# python train.py --dataset Fluo-C3DH-H157 --erosion 5 --use_gold_truth True
# python train.py --dataset Fluo-C3DL-MDA231 --erosion 5 --use_gold_truth True
# python train.py --dataset Fluo-N3DH-CHO --erosion 5 --train_resolution 256 256 --use_gold_truth True
# python train.py --dataset Fluo-N3DH-CE --erosion 2 --resize_to 512 712 --use_gold_truth True