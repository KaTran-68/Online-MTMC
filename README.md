## Online-MTMC-vehicle-tracking

Paper:  https://link.springer.com/article/10.1007/s11042-022-11923-2



# Setup & Running
**Requirements**

The repository has been tested in the following software.
* Ubuntu 16.04
* Python 3.7
* Anaconda
* Pycharm

**Download AIC19 dataset**

The dataset can be downloaded at https://drive.google.com/file/d/13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC/view

**Prepare AIC19 dataset**

Move the downloaded folders *aic19-track1-mtmc/train* and *aic19-track1-mtmc/test* to the *./datasets/AIC19/* repository folder.

Preprocess the data to extract the images from the .avi files by running:

```
python preprocessing_data\preprocess_data.py   
```


The set of data can be changed, by default it will preprocess */test/S02* scenario.


**Download pretrained model**

The model weights trained on AIC19 S01 scenario can be downloaded at:
http://www-vpu.eps.uam.es/publications/Online-MTMC-Tracking/ResNet50_AIC20_VERI_layer5_imaugclassifier_latest.pth.tar


Place the weights file under *./models/*

Training details can be found in the paper.

**Training Re-id<Beta version>**

Gen Crops:
python prepare_reid_dataset.py --dataset-root datasets/AIC19 --set train --out datasets_reid --val-ratio 0.1 --min-area 500

Train Re-id:
python train_reid.py --data datasets_reid --out models/reid_finetune.pth.tar --epochs 80 --batch-size 32 --lr 0.1 --workers 4

python train_reid.py --data datasets_reid --out models/reid_finetune.pth.tar --epochs 120 --batch-size 32 --lr 0.1 --workers 4 --pretrained models/reid_finetune.pth.tar --resume

**Running**

To run the tracking algorithm over the S02 scenario run:

```
python main.py --ConfigPath ./config/config.yaml  
```

**Visualize**
To visualize the results (e.g.):

```
python visualize_tracks.py --scene S01 --camera c001 --results results\S01\prueba.txt --dataset-root datasets\AIC19 --out results\S01\c001_viz.mp4

python visualize_tracks.py --scene S01 --camera c001 --results results\S01\prueba.txt --dataset-root datasets\AIC19 --out results\S01\c001_viz.mp4 --use-imgs
```


# Citation

If you find this code and work useful, please consider citing:
```
@article{luna2022online,
  title={Online clustering-based multi-camera vehicle tracking in scenarios with overlapping FOVs},  
  author={Luna, Elena and SanMiguel, Juan C and Mart{\'\i}nez, Jos{\'e} M and Escudero-Vi{\~n}olo, Marcos},  
  journal={Multimedia Tools and Applications},
  pages={1--21},  
  year={2022},  
  publisher={Springer}
}
```



