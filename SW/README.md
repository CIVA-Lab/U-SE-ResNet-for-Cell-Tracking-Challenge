# Train and Test Models
All scripts are given inside this folder to train models and inference them. ```requirement.txt``` is provided to set up the environment to run given scripts. 

<br/>

## Train Models
All scripts are given inside ```train_codes``` folder to train all necessary models required for Cell Tracking Challenge 2021. By simple modification the scripts can be used for another dataset too. 

When training individual models, such as using only ```Silver Truth (ST)``` as ground truth for training the ```erosion``` paremeter is used to erode ```ST``` and obtain ```markers```. Each erosion parameters used for each dataset are given inside ```.sh``` files. 

```bash
python train.py --dataset Fluo-C2DL-MSC --erosion 2 --resize_to 800 992
```

For example, the erosion parameter to get marker for dataset ```Fluo-C2DL-MSC``` is ```2```, and since the size of image are different in ```01``` and ```02```, resize parameter is used to make both video sequence same size. 

The trained models are stored inside ```trained_models``` folder. 

## Test Models
All scripts are given inside ```test_codes``` folder to test all trained models required for Cell Tracking Challenge 2021. 

```bash
python eval_st.py --dataset PhC-C2DH-U373  --sequence_id 01
```

For example, to inference sequence ```01``` of ```PhC-C2DH-U373``` dataset using trained ```ST``` model the above script is called. 
