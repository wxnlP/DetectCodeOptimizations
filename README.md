# 优化目标检测代码结构

## 目录结构介绍

```
--pic------测试照片
--videos---测试视频
--G-final.onnx---训练结果
--Main.py-----应用示例（QT）
--detect.py---重构的检测代码
  --面向对象形式
  --直接以函数形式调用即可
  --便于添加属性和功能
--Data.py-----自定义检测数据类型
  --可自行按需修改
```

## 下载源文件

打开Linux终端，克隆源码

```shell
git clone https://github.com/wxnlP/DetectCodeOptimizations.git
```

将<kbd>detect.py</kbd>和<kbd>Data.py</kbd>放到自己的工程文件夹根目录，安装依赖

```shell
#更新系统包列表
sudo apt update
#安装opencv-python
pip install opencv-python
#若opencv报关于GStreamer的错，可选择安装包含FFmpeg、GStreamer的完整版本
pip install opencv-python-headless
#安装numpy
pip install numpy
#安装onnxruntime
pip install onnxruntime
#安装pyserial
pip install pyserial
```

终端检验安装

```shell
python3 -c "import cv2, numpy, onnxruntime, serial; print('All libraries loaded successfully')"
```

## detect.py使用方法

### 初始化

根据自己的文件目录修改必要参数

```python
from detect import Detect
# 创建对象
APP = Detect()
# 初始化个性参数
APP.model_pb_path = "Your ONNX file's path" # Linux要求使用绝对路径
APP.label_key = list(range(num))  # num--你的模型的标签的个数（或者说标签ID）
APP.lable_value = ["", "",....]   # num--标签ID对应的标签名称
```

默认参数如下

```python
labels = [
        "0-plastic_bottle",
        "0-drink_can",
        "0-paper",
        "0-carton",
        "0-milkCarton",
        "1-pericarp",
        "1-vegetable_leaf",
        "1-radish",
        "1-potato",
        "1-fruits",
        "2-battery",
        "2-Expired_drug",
        "2-button cell",
        "2-thermometer",
        "3-tile",
        "3-cobblestone",
        "3-brick",
        "3-paperCup",
        "3-tableware",
        "3-chopsticks",
        "3-butt",
        "3-mask"]
self.model_pb_path = "/home/sunrise/DefectDetect/G_final.onnx"
self.label_key = list(range(22))
self.lable_value = labels
```

### 使用方法

将<kbd>detect.py</kbd>和<kbd>Data.py</kbd>放到工程根目录，<kbd>from detect import Detect</kbd>即可：

```python
from detect import Detect
# 创建对象
APP = Detect()
# 照片识别，ch参数--1:返回DetectData数据类型--0:返回img数据（默认ch=0）
APP.detect_pic(img, ch=1)
# 视频处理，传入视频地址--默认输出视频存放地址--'/home/sunrise/DefectDetect/videos/output_video.mp4'
# 根据设备自行修改
APP.detect_video(path)
```

### 实时检测

```python
"""摄像头处理示例"""
video = 0
cap = cv2.VideoCapture(video)
while True:
    success, img = cap.read()
    if success:
        data = APP.detect_pic(img, ch=1)
        print("*"*50)
        data.show()
```

### 单个照片处理

```python
 """照片处理示例"""
    img = cv2.imread("/home/sunrise/DefectDetect/pic/test.jpg")
    APP.detect_pic(img)
```

### 视频处理

```python
 """视频处理示例"""
    cap = cv2.VideoCapture('/home/sunrise/DefectDetect/videos/2.mp4')
    APP.detect_video('/home/sunrise/DefectDetect/videos/2.mp4')
```


