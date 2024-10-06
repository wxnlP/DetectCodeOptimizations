import cv2
import numpy as np
import onnxruntime as ort
import time
import random as rd
import serial
from Data import DetectData


# 创建目标检测数据存储对象
detect_data = DetectData()

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
        
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [rd.randint(0, 255) for _ in range(3)]
    x=x.squeeze()
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
 
def _make_grid( nx, ny):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)
 
def cal_outputs(outs,nl,na,model_w,model_h,anchor_grid,stride):
    
    row_ind = 0
    grid = [np.zeros(1)] * nl
    for i in range(nl):
        h, w = int(model_w/ stride[i]), int(model_h / stride[i])
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = _make_grid(w, h)
 
        outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
            grid[i], (na, 1))) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
            anchor_grid[i], h * w, axis=0)
        row_ind += length
    return outs
 
def post_process_opencv(outputs,model_h,model_w,img_h,img_w,thred_nms,thred_cond):
    conf = outputs[:,4].tolist()
    c_x = outputs[:,0]/model_w*img_w
    c_y = outputs[:,1]/model_h*img_h
    w  = outputs[:,2]/model_w*img_w
    h  = outputs[:,3]/model_h*img_h
    p_cls = outputs[:,5:]
    if len(p_cls.shape)==1:
        p_cls = np.expand_dims(p_cls,1)
    cls_id = np.argmax(p_cls,axis=1)
 
    p_x1 = np.expand_dims(c_x-w/2,-1)
    p_y1 = np.expand_dims(c_y-h/2,-1)
    p_x2 = np.expand_dims(c_x+w/2,-1)
    p_y2 = np.expand_dims(c_y+h/2,-1)
    areas = np.concatenate((p_x1,p_y1,p_x2,p_y2),axis=-1)
    
    areas = areas.tolist()
    ids = cv2.dnn.NMSBoxes(areas,conf,thred_cond,thred_nms)
    if len(ids)>0:
        return  np.array(areas)[ids],np.array(conf)[ids],cls_id[ids]
    else:
        return [],[],[]
    
def infer_img(img0,net,model_h,model_w,nl,na,stride,anchor_grid,thred_nms=0.4,thred_cond=0.5):
    # 图像预处理
    img = cv2.resize(img0, (model_w,model_h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
 
    # 模型推理
    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
 
    # 输出坐标矫正
    outs = cal_outputs(outs,nl,na,model_w,model_h,anchor_grid,stride)
 
    # 检测框计算
    img_h,img_w,_ = np.shape(img0)
    boxes,confs,ids = post_process_opencv(outs,model_h,model_w,img_h,img_w,thred_nms,thred_cond)
 
    return  boxes,confs,ids
 
class Detect:
    def __init__(self):
        self.model_pb_path = "/home/sunrise/DefectDetect/G_final.onnx"
        self.label_key = list(range(22))
        self.lable_value = labels
        self.label_dic = {}
        so = ort.SessionOptions()
        self.net = ort.InferenceSession(self.model_pb_path, so)
        # 模型参数
        self.model_h = 320
        self.model_w = 320
        self.nl = 3
        self.na = 3
        self.stride=[8.,16.,32.]
        self.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(self.anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.Init()


    def Init(self):
        for i in range(len(self.label_key)):
            self.label_dic[i] = self.lable_value[i]

    
    def detect_pic(self, img0, ch=0):
        t1 = time.time()
        det_boxes,scores,ids = infer_img(img0,self.net,self.model_h,self.model_w,self.nl,self.na,self.stride,self.anchor_grid,thred_nms=0.4,thred_cond=0.5)
        t2 = time.time()
        print("-"*50)
        for box,score,id in zip(det_boxes,scores,ids):
            label = '%s:%.2f'%(self.label_dic[id.item()],score)
            # 物品名称
            detect_data.name = self.label_dic[id.item()]
            # 取中心坐标
            detect_data.coordinate_x = ((box[0] + box[2])*0.5).round(2)
            detect_data.coordinate_y = ((box[1] + box[3])*0.5).round(2)
            # 置信度
            detect_data.confidence = score.round(2)
            # 显示检测数据
            detect_data.show()
            plot_one_box(box.astype(np.int16), img0, color=(0,0,255), label=label, line_thickness=None)
            id = id.item()
        str_FPS = "FPS: %.2f"%(1./(t2-t1))
        # cv2.putText(img0,str_FPS,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        # cv2.imwrite('/home/sunrise/DefectDetect/pic/output.jpg', img0)
        
        print(str_FPS)
        if ch:
            return detect_data
        else:
            return img0
    
    def detect_video(self, video):
        cap = cv2.VideoCapture(video)
        # 获取视频的宽度、高度和帧率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码格式
        out = cv2.VideoWriter('/home/sunrise/DefectDetect/videos/output_video.mp4', fourcc, fps, (width, height))
        while True: 
            ret, frame = cap.read()
            print(ret)
            if not ret:
                break  # 如果没有读取到帧，退出循环
            self.detect_pic(frame) 
            # 将处理后的帧写入新视频
            out.write(frame)
            cv2.waitKey(int(1000 / (fps*3)))  # 控制播放速度，保持帧率
         

if __name__ == "__main__":
    APP = Detect()
    """摄像头处理示例"""
    video = 0
    cap = cv2.VideoCapture(video)
    while True:
        success, img = cap.read()
        if success:
            data = APP.detect_pic(img, ch=1)
            print("*"*50)
            data.show()
    """照片处理示例"""
    # img = cv2.imread("/home/sunrise/DefectDetect/pic/test.jpg")
    # APP.detect_pic(img)
    """视频处理示例"""
    # cap = cv2.VideoCapture('/home/sunrise/DefectDetect/videos/2.mp4')
    # APP.detect_video('/home/sunrise/DefectDetect/videos/2.mp4')
    
    