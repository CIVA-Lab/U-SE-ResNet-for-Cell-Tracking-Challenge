# 2D
# python train_gt.py --dataset BF-C2DL-HSC --erosion 5  
# python train_gt.py --dataset PhC-C2DL-PSC --erosion 5  
python train_gt.py --dataset PhC-C2DH-U373 --erosion 2  
# python train_gt.py --dataset BF-C2DL-MuSC --erosion 5  
# python train_gt.py --dataset DIC-C2DH-HeLa --erosion 20 
# python train_gt.py --dataset Fluo-C2DL-MSC --erosion 2 --resize_to 800 992
# python train_gt.py --dataset Fluo-N2DL-HeLa --erosion 2
# python train_gt.py --dataset Fluo-N2DH-GOWT1 --erosion 10

# 3D
# python train_gt.py --dataset Fluo-C3DH-A549 --erosion 5 --resize_to 320 416 --train_resolution 320 416
# python train_gt.py --dataset Fluo-C3DH-H157 --erosion 5
# python train_gt.py --dataset Fluo-C3DL-MDA231 --erosion 5
# python train_gt.py --dataset Fluo-N3DH-CHO --erosion 5 --train_resolution 256 256
# python train_gt.py --dataset Fluo-N3DH-CE --erosion 2 --resize_to 512 712