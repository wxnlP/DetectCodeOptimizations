class DetectData:
    def __init__(self):
        """ 自定义检测数据类型 
        name = 零件名称
        kind = 缺陷类别
        coordinate_x = x坐标
        coordinate_y = y坐标
        confidence = 置信度
        """
        self.name = "0"
        self.kind = "良品"
        self.coordinate_x = 0
        self.coordinate_y = 0
        self.confidence = 0 
    
    def show(self):
        print(f"物件名称--{self.name}")
        print(f"缺陷类型--{self.kind}")
        print(f"中心坐标X--{self.coordinate_x}")
        print(f"中心坐标Y--{self.coordinate_y}")
        print(f"置信度--{self.confidence}")
 