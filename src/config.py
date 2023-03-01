import os

# 本地的数据根目录
root = "../assets"

arrythmia = os.path.join(root, "arrhythmia_v1.txt")  # 类别标签文件
dir_24h = os.path.join(root, "bisha_data")  # 24小时数据路径
beats_24h = os.path.join(root, "bisha_24hbeat")  # 24小时心拍列表，由24小时数据路径内提取出
R_24h = os.path.join(root, "bisha_24hRpeak")  # 24小时R波列表
mybeats_24h = os.path.join(root, "bisha_24hMybeat")
report_24h = os.path.join(root, "bisha_24hreport")

# label的类别数
num_classes = 10

# 是否进行数据标准化
data_standardization = True

# 训练的模型名称，模型类名或函数名，可以返回对应名字的模型对象
model_name = "resnet34_cbam_ch1"

# 目标的采样长度
target_point_num = 360
