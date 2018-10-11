# Caffe--实现YOLOv2目标检测

https://blog.csdn.net/lwplwf/article/details/83011667

DarkNet转Caffe中有很多潜在的问题，在YOLOv1、v2、v3几个网络中有一些特殊的层。要在Caffe中跑YOLO，就得在Caffe中源码实现这些层。这些层的Caffe源码实现可以在网上找到很多。

在Caffe平台实现YOLO系列，可以分成以下两种方式：

（1）DarkNet平台训练完成YOLO模型，然后将.cfg文件和.weights文件通过脚本转换为Caffe框架下的.prototxt文件和.caffemodel文件，最后在Caffe下使用转换好的.prototxt文件和.caffemodel文件进行目标检测任务。

（2）手动写好.prototxt模型结构文件，直接在Caffe下进行训练，训练完成后进行目标检测任务。

以上两种方式，都是以当前Caffe已经源码实现上面那几个特殊层为前提。

具体按哪种方式看自己实际需求，比如，我现在是已经有在DarkNet下训练好可用YOLOv2_tiny模型，所以我选择将训练好的模型转换到Caffe再使用，而不是从头训练。个人感觉YOLO在原生框架DarkNet下训练起来更方便一些，更重要的是，在Caffe实现YOLO后可以将中间参数以及输出结果拿出来，再和DarkNet下的YOLO做对比分析，这点还是很关键的，相同的模型结构和权重参数，经过对比可以很清楚的知道转换是否正确、Caffe新加层实现是否正确。

## 1、YOLOv1

YOLOv1主要是需要实现detection层，再就是YOLO系列中使用的激活函数是Leaky，可以选择单独实现，也可以用ReLU实现（配置参数）。

Leaky层源码实现资源：https://download.csdn.net/download/lwplwf/10712919

detection层源码实现资源：https://download.csdn.net/download/lwplwf/10712961

Leaky层的添加和detection层是一致的，具体实现参考：

（1）YOLOv1的Detection层实现：https://blog.csdn.net/lwplwf/article/details/82788376

（2）实现YOLOv1目标检测：https://blog.csdn.net/lwplwf/article/details/82685347

**注**：该detection的实现不支持GPU模式，通过这里初步了解Caffe下源码实现添加新层也还是可以的。

## 2、YOLOv2

重点说YOLOv2的实现，YOLOv2的Caffe实现就已经很多了，网上可以找到很多个版本，都挺好的，但也都存在的不同的问题。

还不错的几个实现：

- https://github.com/gklz1982/caffe-yolov2
- Caffe源码：https://github.com/hustzxd/z0, 相关转换工具：https://github.com/hustzxd/z1
- https://github.com/marvis/pytorch-caffe-darknet-convert

上面几个都有很大的参考价值，综合以上和其他一些实现，踩了很多坑，总算是把这条路走下来了。

**结合前人成果，整理得到目前可用的Caffe源码**：
https://github.com/lwplw/caffe_yolov2，（已经包含了YOLO一些层的实现）


**具体步骤如下（以YOLOv2_tiny为例）：**

**（1）训练YOLOv2_tiny**

在DarkNet下完成，参考官网，https://pjreddie.com/darknet/yolo/

**这里有个问题需要注意！！！**

下图中，对于YOLOv2_tiny，416416的输入，经过最后一个max pool（size=2，stride=1），可以看到特征图`13*13`处理后还是`13*13`。
这就有点问题了，测试发现，Caffe中经过这一层特征图由`13*13`变成了`12*12`，会导致在Caffe下检测结果的box有偏差。

![image](https://github.com/lwplw/repository_image/blob/master/4871D0BC-DFB5-4354-9ED1-E0A324FCF640.png)

![image](https://github.com/lwplw/repository_image/blob/master/CDA747F4-51D4-4c2c-967E-49512FD2B6DE.png)

**解决方案**：
对这层max pool使用pad，由于Caffe和DarkNet对pool的处理逻辑有些差异，需要指定DarkNet中该层padding=2，Caffe种该层pad=1。
去看DarkNet源码maxpool_layer.c发现：

简单说一下，在DarkNet中有pad和padding两个东西，是不一样的，重点体现在卷积层中，而对于池化层就没那么复杂了，用padding指定参数就行。

处理之后，经该max pool层处理特征图会由13*13变为14*14，DarkNet和Caffe两个框架下达成统一。

**（2）训练完成后得到.weights模型权重文件**

**（3）模型转换**

使用脚本将.cfg文件和.weights文件转换为.prototxt文件和.caffemodel文件。

模型转换参考：https://github.com/lwplw/darknet2caffe

**（4）在Caffe框架下进行测试**

- 下载caffe_yolov2源码：https://github.com/lwplw/caffe_yolov2
- 解压得到文件夹caffe_yolov2-master
- 编译安装

```
make all -j8
sudo make pycaffe -j8
```

具体参考：https://blog.csdn.net/lwplwf/article/details/82415620

编译过程中有warning，但不影响。

- 测试

进入`caffe_yolov2-master/examples/yolov2`目录下，执行命令：

```
python detect.py
```

注：具体测试时使用的数据集，以及测多少张等，自行在脚本detect.py中进行修改。
