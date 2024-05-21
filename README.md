# YOLOv5-dangerous_driving_behavior_detection
基于YOLOv5s+CBAM注意力机制的危险驾驶行为识别系统

整个项目在GitHub上开源的https://github.com/JingyibySUTsoftware/Yolov5-deepsort-driverDistracted-driving-behavior-detection 基于深度学习的驾驶员分心驾驶行为（疲劳+危险行为）预警系统的基础上**增加了语音警告功能**，使用YOLOv5s-6.0进行目标检测，在C3模块中增加了**CBAM注意力机制**，提高算法的特征提取能力和检测精度。


### 环境
| 环境        |       版本号 |
| ----------- | -----------: |
| python      |       3.8.10 |
| torch       | 1.8.1 +cu101 |
| torchaudio  |        0.8.1 |
| torchvision |  0.9.1+cu101 |
**如果要使用GPU，cuda版本要 >=10.1**
**如果不使用GPU，后面三个不用安装**

```python
   pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
```
### 数据集介绍
使用飞浆AI Studio中的吸烟喝水玩手机数据集https://aistudio.baidu.com/datasetdetail/80631 ，里面有*VOCDATA*文件夹，可以直接导入到项目的VOCDATA中。

- 运行VOCData/split_train_val.py，会生成 ImagesSets\Main 文件夹，且在其下生成测试集、训练集、验证集。
- 运行 VOCData/text_to_yolo.py ，运行后会生成如下 labels 文件夹和 dataSet_path 文件夹。
- 运行VOCData/kmeans.py 以及 VOCData/clauculate_anchors.py 用来通过聚类获得先验框，如果有修改，根据 生成的anchors.txt 中的 Best Anchors 修改yolov5s.yaml，数值需要取整。

**如果要重新训练，注意要修改data/myvoc.yaml文件和VOCDATA/clauculate_anchors.py，把文件路径成自己的路径。**
### YOLO模型训练

```python
python train.py --weights weights/yolov5s.pt  --cfg models/yolov5s.yaml  --data data/myvoc.yaml --epoch 200 --batch-size 8 --img 640   --device cpu
```
- –weights weights/yolov5s.pt ：我是将yolov5的pt文件都放在weights目录下，你可能没有，需要更改路径。

- –epoch 200 ：训练200次

- –batch-size 8：训练8张图片后进行权重更新

- –device cpu：使用CPU训练。

如果采用手动法获取anchors，可以选择补充添加参数 --noautoanchor；如果是自动获取，不要加。
> 因为训练时会计算BPR，并且得到的BPR应该是为1的（或者极为接近1），所以不会更新anchors。因此，手动法的话这个参数添不添加无所谓的。

### YOLO模型检验
```python
python detect.py --weights weights/best.pt --source 0 --save-txt
```
### 系统模块功能
- 疲劳检测部分，使用Dlib中用于检测人脸关键点的shape_predictor_68_face_landmarks.dat模型，然后通过计算眼睛和嘴巴的开合程度来判断是存在否闭眼或者打哈欠，并使用Perclos模型计算疲劳程度。
- 分心行为检测部分，使用改进后的YOLOv5s，检测是否存在玩手机、抽烟、喝水这三种行为。

### 系统运行
直接运行main.py
### 关键技术
根据
> 陈跃虎. 基于深度学习的危化品运输罐车危险识别与预警研究[D]. 中国矿业大学,2023. 

提到的在C3模块加入注意力机制比在其他模块效果都好，其中添加CBAM效果最优。

[![](https://img2.imgtp.com/2024/05/21/D5lbEkY1.png)](https://img2.imgtp.com/2024/05/21/D5lbEkY1.png)

原YOLOv5模型图
[![](https://img2.imgtp.com/2024/05/21/k4MDL9Am.png)](http://https://img2.imgtp.com/2024/05/21/k4MDL9Am.png)

在C3模块中添加CBAM注意力机制
[![](https://img2.imgtp.com/2024/05/21/y14zduFP.png)](https://img2.imgtp.com/2024/05/21/y14zduFP.png)

### 整个系统的技术路线
[![](https://img2.imgtp.com/2024/05/21/Aibw3Ssr.png)](https://img2.imgtp.com/2024/05/21/Aibw3Ssr.png)
### GUI设计
[![](https://img2.imgtp.com/2024/05/21/85DGyHcp.png)](https://img2.imgtp.com/2024/05/21/85DGyHcp.png)
==========