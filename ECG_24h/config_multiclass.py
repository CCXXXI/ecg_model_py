# -*- coding: utf-8 -*-
import os


class Config:
    onserver = True  # 是否在服务器上训练
    # 本地的数据根目录
    # root = 'E:/Work/ECG_AI/上海数创医疗/data/20210207 筛选数据26分类数据各1000条' # 第一批数据本地根目录
    # root = 'E:/Work/ECG_AI/上海数创医疗/data/20210401 筛选数据90258条'  # 第二批数据本地根目录
    root = 'E:/Work/ECG_AI/上海数创医疗/data/20210519_120000'
    # root = 'E:/Work/ECG_AI/上海数创医疗/data/20210813_3x3000'


    if onserver:
        # root = '/home/bywang/ECG_AI/shuchuang/data/20210207_26_1000'  # 第一批数据服务器根目录
        # root = '/home/bywang/ECG_AI/shuchuang/data/20210401_90285_91158'  # 第二批数据服务器根目录
        root = '/home/bywang/ECG_AI/shuchuang/data/20210519_120000'
        # root = '/home/ShuChuang/works/data/600000'
        # root = '/home/ShuChuang/works/data/20210519_120000'

    train_dir = os.path.join(root, 'alldata_npy')  # 训练集文件夹
    test_dir = os.path.join(root, 'alldata_npy')  # 测试集文件夹
    train_label = os.path.join(root, 'train_label_v23.txt')  #训练验证集的标签
    test_label = os.path.join(root, 'test_label_v23.txt')  # 测试集的标签
    alldata_label = os.path.join(root, 'labels_v23.txt')  # 整个数据集的标签，包括测试集验证集训练集，在划分测试集时使用
    arrythmia = os.path.join(root, 'arrythmia_v23.txt')  # 类别标签文件
    train_data = os.path.join(root, 'train_v23.pth')  # 划分好的训练集验证集数据
    # train_labelrelationship_matrix = os.path.join(root, 'train_v7_labelrelationship_matrix.csv')

    all_arrythmia = os.path.join(root, 'arrythmia_ori_v7.txt')
    alldata_full_label = os.path.join(root, 'ori_labels_v7_120000.csv')

    experiment_name = 'v23_st16_lr1e-3_bsz64_halfaug-Sft100-gn0.005mv'  # 实验名称，会追加到保存模型的文件夹名

    # 迁移学习相关配置，使用迁移学习时需要
    # pretrained_model_ckpt = './ckpt_bin_supcon/Supcon_resnet34_cbam_pretrain_202108271644_v19_MLBavg_1p_doteachi_1normsqnavg_st32_lr1e-3_nrlval_warmup16_1e-5_bsz256_augSft100-gn0.005mv/best_w.pth'  # 预训练模型路径
    # classifier_model_name = 'mlp_fv2'  # 迁移模型名，用于决定对预训练模型结构做何种修改


    DEBUG = False  # 用于DEBUG打印信息的开关

    test_ratio = 0.1  # 测试集占整个数据集的比例，划分数据集时使用
    val_ratio = 0.1  # 验证集占训练集验证集的比例，划分数据集时使用

    # label的类别数
    num_classes = 3

    # 输入原始数据的单位值，微伏
    inputUnit_uv = 2.4
    # 送入模型里的数据的单位值，微伏
    targetUnit_uv = 4.88
    # 是否进行数据标准化
    data_standardization = False

    # 优化器 默认Adam 可选SGD
    # optimizer = 'SGD'
    # momentum = 0.9  # SGD 优化器参数

    # 学习率衰减方式，如果注释掉则为默认衰减方式，不使用pytorch库中的lr_schedular，在stage_epoch包含的epoch处进行学习率衰减
    # lr_scheduler = 'CosineAnnealingWarmRestarts'  # 现支持‘CosineAnnealingWarmRestarts’和‘CosineAnnealingLR’

    # CosineAnnealingWarmRestarts 的参数
    # T_0 = 80  # 多少个epoch第一次重启
    # T_mult = 2  # 后续重启的间隔时间是上一次重启间隔的几倍（整数）
    # eta_min = 1e-6  # 最低学习率阈值

    # CosineAnnealingLR的参数
    # T_0 = 80  # 余弦函数半个周期
    # eta_min = 1e-6  # 最低学习率阈值


    # 是否在开始新stage（学习率衰减时）重新加载验证集上表现最好的参数，注释掉则为默认开启
    # reload_newstage = False

    # 损失函数选择 'WeightedMultilabel' or 'FocalLoss' or 'WeightedMultilabelWithLabelSmooth'，'SupconLoss' 和 'MLB_SupConLoss', 多分类的'WeightedCrossEntropyLoss' 和 'MultiClassFocalLoss', 注释掉则为默认的WeightedMultilabel
    loss_function = 'WeightedCrossEntropyLoss'

    # MLB_SupConLoss的设置
    # 类别错误率，与all_arrythmia对应
    MLB_error_rate_path = './ckpt_bin_supcon/Supcon_resnet34_cbam_202108252101_v19_st16_24_lr1e-3_warmup16_1e-5_bsz64_halfaug-same_mlp_fv1_Supcon_resnet34_cbam_pretrain_202108242206_v19_st32_nrlval_lr1e-3_warmup16_1e-5_bsz256_aug-Sft100-gn0.005mv/val_th0.18_error_rate_1normsq_navg.csv'
    MLB_multi_label_error_rate_mode = 'aver'  # 'aver' or 'max' 处理多标签数据错误率的方式，平均还是取最大值
    MLB_weight_mode = 'oneplus'  # 'oneplus' or 'onesub' 从错误率到权重的方式 1+error_rate or 1-error_rate
    MLB_formula_mode = 'dot_eachi'  # 损失函数公式选择，direct_dot直接将权重点乘到全部特征向量上,dot_eachi将权重乘在每个i上

    # 是否进行数据增强，注释掉为默认进行数据增强
    data_augmentation = True
    # 按以下方式进行数据增强，注释掉即为默认方式
    # 目前可选 Shift, GaussianNoise, ChannelResize, RandomResizedCrop, Verflip, Scaling
    transforms = ["Shift", "GaussianNoise"]

    # 每个数据增强有一半的概率应用到数据上
    half_chance_data_augmentation = True

    # 数据增强参数
    Shift_interval = 100 # 20
    GaussianNoise_scale = 0.005 * 1000 / targetUnit_uv  # 0.005
    ChannelResize_magnitude_range = [0.5, 2]
    RandomResizedCrop_crop_ratio_range = [0.5, 1.0]
    Scaling_std = 0.1

    # ----------------------------------------------------

    # consider_labelrelationship = True
    # labelrelationship_self_ratio = 0.0
    # labelrelationship_self_weight = 0.5
    # labelrelationship_other_weight = 0.2

    epoch_val = True  # 是否以整个epoch为单位计算验证集F1和ACC等指标，注释掉即为不是
    # for train
    #训练的模型名称，模型类名或函数名，可以返回对应名字的模型对象
    model_name = 'resnet34_cbam'
    # model_name = 'HighResolutionNet'

    # 模型参数
    # ---------------------------
    # 针对MLGCN
    MLGCN_t_threshold = 0.4
    MLGCN_gen_A_method = 'default'  # default or t_threshold

    # ---------------------------

    # SupConLoss参数
    SupCon_temperature = 0.07
    SupCon_base_temperature = 0.07
    SupCon_contrast_mode = 'all'

    # Warmup 是否进行学习率预热
    warmup = False
    warmup_epoch = 16
    warmup_lr_from = 1e-5


    #在第几个epoch进行到下一个state,调整lr为lr/=lr_decay
    # stage_epoch = [40, 80, 120, 160,200]
    # stage_epoch = [32, 64, 128]
    stage_epoch = [16, 24, 40, 64, 80, 128]
    #训练时的batch大小
    batch_size = 64
    #最大训练多少个epoch
    max_epoch = 256
    #目标的采样长度
    target_point_num = 2048
    #保存模型的文件夹
    ckpt = 'ckpt_multiclass'
    #保存提交文件的文件夹
    sub_dir = 'submit'
    #初始的学习率
    lr = 1e-3
    #保存模型当前epoch的权重
    current_w = 'current_w.pth'
    #保存最佳的权重
    best_w = 'best_w.pth'
    # 学习率衰减 lr/=lr_decay
    lr_decay = 10

    #for test
    temp_dir=os.path.join(root,'temp')


config = Config()

