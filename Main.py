from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from Data import DetectData
from detect import Detect
import numpy as np
import cv2
import sys

class InferenceThread(QThread):
    frame_processed = pyqtSignal(DetectData, np.ndarray)  # 信号，用于传递推理结果和处理后的帧

    def __init__(self, capture):
        super(InferenceThread, self).__init__()
        self.capture = capture
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                continue
            # 进行检测
            data = APP.detect_pic(frame, ch=1)  # 假设 `APP.detect_pic()` 返回一个 DetectData 对象
            # 发射信号，将检测数据和处理后的帧传递回主线程
            self.frame_processed.emit(data, frame)

    def stop(self):
        self.running = False
        self.wait()  # 等待线程停止


APP = Detect()

class UIWidgets(QMainWindow):
    def __init__(self):
        super(UIWidgets, self).__init__()
        self.button_begin = QPushButton("Detect-ing")
        self.button_stop = QPushButton("Stop-ed")
        self.button_sources = QPushButton("Sources...")
        self.button_clear = QPushButton("Clear")

        self.header_labels = ["零件名称", "缺陷类别", "中心坐标", "置信度"]
        self.table_data = QTableWidget(0, 4)
        self.table_data.setHorizontalHeaderLabels(self.header_labels)

        self.table_data.horizontalHeader().setStretchLastSection(True)
        for i in range(4):
            self.table_data.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)

        self.label_video = QLabel(self)
        self.label_video.setText("正在加载摄像头...")
        self.label_video.setScaledContents(True)

        self.capture = None
        self.inference_thread = None

        self.layoutInit()
        self.signals()

    def layoutInit(self):
        layout_button1 = QHBoxLayout()
        layout_button2 = QHBoxLayout()
        layout1 = QVBoxLayout()
        layout2 = QHBoxLayout()

        layout_button1.addWidget(self.button_begin)
        layout_button1.addWidget(self.button_stop)
        layout_button2.addWidget(self.button_sources)
        layout_button2.addWidget(self.button_clear)

        layout1.addWidget(self.label_video)
        layout1.addLayout(layout_button1)
        layout1.addLayout(layout_button2)

        layout2.addWidget(self.table_data)
        layout2.addLayout(layout1)
        layout2.setStretch(0, 3)
        layout2.setStretch(1, 7)

        central_widget = QWidget()
        central_widget.setLayout(layout2)
        self.setCentralWidget(central_widget)

    def add_data(self, detect_data: DetectData):
        row_position = self.table_data.rowCount()
        self.table_data.insertRow(row_position)
        self.table_data.setItem(row_position, 0, QTableWidgetItem(detect_data.name))
        self.table_data.setItem(row_position, 1, QTableWidgetItem(detect_data.kind))
        self.table_data.setItem(row_position, 2, QTableWidgetItem(f"X:{detect_data.coordinate_x}/Y:{detect_data.coordinate_y}"))
        self.table_data.setItem(row_position, 3, QTableWidgetItem(f"{detect_data.confidence}"))
        self.table_data.scrollToBottom()

    def upload_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Images (*.png *.jpg *.jpeg *.gif);;Videos (*.mp4 *.avi)", options=options)
        if file_name:
            file_extension = file_name.split('.')[-1].lower()
            if file_extension in ['png', 'jpg', 'jpeg', 'gif']:
                img = cv2.imread(file_name)
                if img is not None:
                    target_size = (640, 480)
                    img = cv2.resize(img, target_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    height, width, channel = img.shape
                    bytes_per_line = channel * width
                    APP.detect_pic(img)
                    q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    self.label_video.setPixmap(QPixmap.fromImage(q_image))
                else:
                    QMessageBox.warning(self, "错误", "无法读取图像文件。")
            elif file_extension in ['mp4', 'avi']:
                APP.detect_video(file_name)
                self.label_video.setText("视频处理完成")

    def signals(self):
        self.button_sources.clicked.connect(self.upload_file)
        self.button_begin.clicked.connect(self.start_camera)
        self.button_stop.clicked.connect(self.stop_camera)
        self.button_clear.clicked.connect(self.label_clear)

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.label_video.setText("无法打开摄像头。")
            return
        self.start_inference_thread()

    def stop_camera(self):
        self.stop_inference_thread()
        self.label_video.clear()
        if self.capture is not None:
            self.capture.release()
            
    def label_clear(self):
        self.label_video.clear()

    def start_inference_thread(self):
        self.inference_thread = InferenceThread(self.capture)
        self.inference_thread.frame_processed.connect(self.update_frame)  # 连接信号
        self.inference_thread.start()

    def stop_inference_thread(self):
        if self.inference_thread is not None:
            self.inference_thread.stop()
            self.inference_thread = None

    def update_frame(self, data, frame):
        self.add_data(data)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(q_image))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = UIWidgets()
    ex.show()
    sys.exit(app.exec_())
