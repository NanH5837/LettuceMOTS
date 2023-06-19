# LettuceMOTS
## Demo
https://github.com/NanH5837/LettuceMOTS/assets/82307771/f1b3af82-ed68-4f66-b941-5f5131d96fc6
## LettuceMOTS Dataset
The **LettuceMOTS dataset** is publicly available and can be found [here](https://drive.google.com/drive/folders/1HIoiyUOu4zYh8jHgqebnbZF_Ewn6Hq62?usp=sharing).
## Getting started
### Prerequisites
Clone repo and create a 
**Python>=3.7.0** environment, including **PyTorch>=1.7**.
We recommend that you install Anaconda, set up a virtual environment and run:

```bash
git clone https://github.com/NanH5837/LettuceMOTS  # clone
cd LettuceMOTS
pip install -r requirements.txt  # install
```

### Data
First, you need to download the LettuceMOTS dataset. 
For the convenience of training, we also provide the labels in the format required for YOLO-V5 training. After the download is complete, place it in the following way:

```
LettuceMOTS
│   images
│   │    0000
│   │   │   000000.png
│   │   │   000001.png
│   │   │   ...
│   labels
│   │    0000
│   │   │   000000.txt
│   │   │   000001.txt
│   │   │   ...
```

After that, you need to create train.txt and val.txt. The content of the two files is the absolute path of the image in the above folder as follows:

```
/XXX/LettuceMOTS/images/0000/000000.png
/XXX/LettuceMOTS/images/0000/000001.png
...
```

### Pretrained weights
We provide three pre-trained models trained on the LettuceMOTS dataset, and you can also choose a pre-trained model trained on COCO from [YOLOv5](https://github.com/ultralytics/yolov5) training. 
You can find three trained models [here](https://drive.google.com/drive/folders/1SMZTGZ3_H2yc86TUFeOvKptj_iwBS2Nf?usp=sharing).
### Train
After completing the above steps, you can perform the following operations to train the segmentation network:

```
cd segment
Python train.py --data lettuceMOTS.yaml --cfg yolov5m-seg.yaml  --batch-size 16 --weights yolov5m-seg.pt
```

### Test
After training, you will get a best segmentation network model, then run the following to track：

```
cd tracking
Python track.py --weights best.pt --data lettuceMOTS.yaml --source /Path/to/your/souce
```
## Reference
We borrow some code from [YOLOv5](https://github.com/ultralytics/yolov5) and [SORT](https://github.com/abewley/sort).
