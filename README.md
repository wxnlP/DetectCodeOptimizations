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

## detect.py使用方法

### 使用简介

将<kbd>detect.py</kbd>放到工程根目录，<kbd>from detect import Detect</kbd>即可：

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
    # img = cv2.imread("/home/sunrise/DefectDetect/pic/test.jpg")
    # APP.detect_pic(img)
```

### 视频处理

```python
 """视频处理示例"""
    # cap = cv2.VideoCapture('/home/sunrise/DefectDetect/videos/2.mp4')
    # APP.detect_video('/home/sunrise/DefectDetect/videos/2.mp4')
```


