# -*- coding: utf-8 -*-
'''
@time: 2019/7/23 19:42

@ author: javis
'''
import torch, time, os, shutil
import models, utils
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import *
from data_process import *
from config import config
from tqdm import tqdm
import math
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(41)
torch.cuda.manual_seed(41)


# 保存当前模型的权重，并且更新最佳的模型权重
def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, config.current_w)
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, current_w)
    if is_best:
        shutil.copyfile(current_w, best_w)
    epoch = state['epoch']
    if epoch % 40 == 0:
        best_10k = best_w.split('.pth')[0]+'_ep'+str(epoch)+'.pth'
        shutil.copyfile(best_w, best_10k)
        current_10k = current_w.split('.pth')[0] + '_ep{}.pth'.format(epoch)
        shutil.copyfile(current_w, current_10k)


def train_epoch(model : nn.Module, optimizer, criterion, train_dataloader, show_interval=10):
    model.train()
    load_data_time = 0
    forward_time = 0
    backward_time = 0
    other_time = 0

    pretime = time.time()

    f1_meter, loss_meter, it_count, acc_meter, sensitivity_meter, precision_meter = 0, 0, 0, 0, 0, 0
    for inputs, target in train_dataloader:
        inputs = inputs.to(device)
        target = target.to(device)

        if 'DEBUG' in dir(config) and config.DEBUG:
            curtime = time.time()
            load_data_time += curtime - pretime
            pretime = curtime
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        output = model(inputs)
        loss = criterion(output, target)

        if 'DEBUG' in dir(config) and config.DEBUG:
            curtime = time.time()
            forward_time += curtime - pretime
            pretime = curtime

        if 'DEBUG' in dir(config) and config.DEBUG:
            # print('batch {}:'.format(it_count))
            # print('loss = {}'.format(loss))
            pass

        loss.backward()

        if 'DEBUG' in dir(config) and config.DEBUG:
            curtime = time.time()
            backward_time += curtime - pretime
            pretime = curtime

        if 'MLGCN' in config.model_name:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)

        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        output = output.cpu().detach()
        output = torch.softmax(output, dim=1, dtype=torch.float32)
        y_pred = torch.argmax(output, dim=1, keepdim=False)
        y_pred = y_pred.numpy()

        target = target.cpu().detach().numpy()

        # f1 = metrics.f1_score(target, y_pred, average='micro')
        # f1_meter += f1
        acc = metrics.accuracy_score(target, y_pred)
        acc_meter += acc
        # sensitivity = metrics.recall_score(target, y_pred, average='micro')
        # sensitivity_meter += sensitivity
        # precision = metrics.precision_score(target, y_pred, average='micro')
        # precision_meter += precision

        if 'DEBUG' in dir(config) and config.DEBUG:
            curtime = time.time()
            other_time += curtime - pretime
            pretime = curtime

        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e acc:%.3f" % (it_count, loss.item(), acc))
            if 'DEBUG' in dir(config) and config.DEBUG:
                print('load data time per batch {}s'.format(load_data_time / it_count))
                print('forward time per batch {}s'.format(forward_time / it_count))
                print('backward time per batch {}s'.format(backward_time / it_count))
                print('other time per batch {}s'.format(other_time / it_count))
        if 'DEBUG' in dir(config) and config.DEBUG:
            pretime = time.time()

    # return loss_meter / it_count, f1_meter / it_count, acc_meter / it_count, sensitivity_meter / it_count, precision_meter / it_count
    return loss_meter / it_count, acc_meter / it_count


def val_epoch(model, criterion, val_dataloader):
    model.eval()
    f1_meter, loss_meter, it_count, acc_meter, sensitivity_meter, precision_meter = 0, 0, 0, 0, 0, 0
    epoch_pred = []
    epoch_target = []
    with torch.no_grad():
        for inputs, target in val_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.softmax(output, dim=1, dtype=torch.float32)
            y_pred = torch.argmax(output, dim=1, keepdim=False)
            if 'epoch_val' in dir(config) and config.epoch_val:
                y_pred = y_pred.cpu().detach().numpy()
                target = target.cpu().detach().numpy()
                epoch_pred.append(y_pred)
                epoch_target.append(target)
            else:
                # f1 = metrics.f1_score(target, y_pred, average='micro')
                # f1_meter += f1
                acc = metrics.accuracy_score(target, y_pred)
                acc_meter += acc
                # sensitivity = metrics.recall_score(target, y_pred, average='micro')
                # sensitivity_meter += sensitivity
                # precision = metrics.precision_score(target, y_pred, average='micro')
                # precision_meter += precision

    if 'epoch_val' in dir(config) and config.epoch_val:
        y_pred = np.concatenate(epoch_pred)
        target = np.concatenate(epoch_target)
        # f1 = metrics.f1_score(target, y_pred, average='micro')
        acc = metrics.accuracy_score(target, y_pred)
        # sensitivity = metrics.recall_score(target, y_pred, average='micro')
        # precision = metrics.precision_score(target, y_pred, average='micro')
        # return loss_meter / it_count, f1, acc, sensitivity, precision
        return loss_meter / it_count, acc
    else:
        return loss_meter / it_count, acc_meter / it_count
        # return loss_meter / it_count, f1_meter / it_count, acc_meter / it_count, sensitivity_meter / it_count, precision_meter / it_count


# 一般训练和迁移学习训练的公共部分
def train_procedure(args, model : nn.Module, model_save_dir : str) -> None:
    print(model)

    # data
    train_dataset = ECGDataset(data_path=config.train_data, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=6 if config.onserver else 1)
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4 if config.onserver else 1)
    print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))

    # optimizer
    optimizer = None
    if 'optimizer' not in dir(config) or config.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    elif config.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, momentum=config.momentum)
    elif config.optimizer == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

    # loss function
    criterion = None
    w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
    if 'loss_function' not in dir(config) or config.loss_function == 'WeightedCrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(w)
    elif 'loss_function' not in dir(config) or config.loss_function == 'MultiClassFocalLoss':
        criterion = utils.MultiClassFocalLoss()

    lr_scheduler = None

    if 'lr_scheduler' in dir(config) and config.lr_scheduler == 'CosineAnnealingWarmRestarts':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config.T_0, config.T_mult, config.eta_min)
    elif 'lr_scheduler' in dir(config) and config.lr_scheduler == 'CosineAnnealingLR':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_0, eta_min=config.eta_min)

    best_acc = -1
    lr = config.lr
    start_epoch = 1
    stage = 1

    if 'warmup' in dir(config) and config.warmup:
        lr = config.warmup_lr_from
        for i in range(len(config.stage_epoch)):
            config.stage_epoch[i] += config.warmup_epoch

    # logger = Logger(logdir=model_save_dir, flush_secs=2)
    summary_writer = SummaryWriter(log_dir=model_save_dir, flush_secs=2)

    # 复制config.py文件
    shutil.copy('./config.py', os.path.join(model_save_dir, 'config.py'))

    # =========>开始训练<=========
    for epoch in tqdm(range(start_epoch, config.max_epoch + 1)):
        since = time.time()

        if 'DEBUG' in dir(config) and config.DEBUG:
            print('epoch {}:'.format(epoch))

        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
        # train_loss, train_f1, train_acc, train_sen, train_precision = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
        # val_loss, val_f1, val_acc, val_sen, val_precision = val_epoch(model, criterion, val_dataloader)
        val_loss, val_acc = val_epoch(model, criterion, val_dataloader)
        print(
            '#epoch:%02d stage:%d train_loss:%.3e train_acc:%.3f val_loss:%0.3e val_acc:%.3f time:%s\n' %
            (epoch, stage, train_loss, train_acc, val_loss, val_acc,
              utils.print_time_cost(since)))
        # print(
        #     '#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f train_acc:%.3f train_sen:%.3f train_prec:%.3f  val_loss:%0.3e val_f1:%.3f, val_acc:%.3f val_sen:%.3f, val_prec:%.3f time:%s\n' %
        #     (epoch, stage, train_loss, train_f1, train_acc, train_sen, train_precision, val_loss, val_f1, val_acc,
        #      val_sen, val_precision, utils.print_time_cost(since)))

        # print('#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s\n'
        #       % (epoch, stage, train_loss, train_f1, val_loss, val_f1, utils.print_time_cost(since)))

        if lr_scheduler is not None:
            lr = lr_scheduler.get_last_lr()[0]

        summary_writer.add_scalar('train_loss', train_loss, global_step=epoch)
        # summary_writer.add_scalar('train_f1', train_f1, global_step=epoch)
        summary_writer.add_scalar('train_acc', train_acc, global_step=epoch)
        # summary_writer.add_scalar('train_sensitivity', train_sen, global_step=epoch)
        # summary_writer.add_scalar('train_presision', train_precision, global_step=epoch)
        summary_writer.add_scalar('val_loss', val_loss, global_step=epoch)
        # summary_writer.add_scalar('val_f1', val_f1, global_step=epoch)
        summary_writer.add_scalar('val_acc', val_acc, global_step=epoch)
        # summary_writer.add_scalar('val_sensitivity', val_sen, global_step=epoch)
        # summary_writer.add_scalar('val_precision', val_precision, global_step=epoch)
        summary_writer.add_scalar('learning_rate', lr, global_step=epoch)

        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'acc': val_acc, 'lr': lr,
                 'stage': stage}
        save_ckpt(state, best_acc < val_acc, model_save_dir)
        best_acc = max(best_acc, val_acc)

        if lr_scheduler is not None:
            lr_scheduler.step()

        if 'warmup' in dir(config) and config.warmup:
            if epoch < config.warmup_epoch:
                p = epoch / config.warmup_epoch
                lr = config.warmup_lr_from + p * (config.lr - config.warmup_lr_from)
                utils.adjust_learning_rate(optimizer, lr)
            elif epoch == config.warmup_epoch:
                lr = config.lr

        if epoch in config.stage_epoch:
            stage += 1

            if 'reload_newstage' not in dir(config) or config.reload_newstage:
                best_w = os.path.join(model_save_dir, config.best_w)
                model.load_state_dict(torch.load(best_w)['state_dict'])

            if lr_scheduler is None:
                lr /= config.lr_decay
                utils.adjust_learning_rate(optimizer, lr)

            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))


# 训练模型
def train(args):
    # model
    model = None
    if 'MLGCN' in config.model_name:
        dd = torch.load(config.train_data)
        idx2name = dd['idx2name']
        wordembedding = load_wordembedding(config.word_embedding_path, idx2name)
        adj = np.loadtxt(config.train_labelrelationship_matrix, delimiter=',')
        for i in range(adj.shape[0]):
            adj[i][i] = 0.0
        model = getattr(models, config.model_name)(num_classes=config.num_classes, adj=adj, word_embedding=wordembedding, t=config.MLGCN_t_threshold, gen_A_method=config.MLGCN_gen_A_method, device=device)
    else:
        model = getattr(models, config.model_name)(num_classes=config.num_classes)
    if args.ckpt and not args.resume:
        state = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        print('train with pretrained weight val_f1', state['f1'])
    model = model.to(device)

    # 模型保存文件夹
    model_save_dir = '%s/%s_%s_%s' % (config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"), config.experiment_name)
    if args.ex: model_save_dir += args.ex

    train_procedure(args, model, model_save_dir)


#用于验证加载模型
def val(args):
    list_threhold = [0.5]
    model = getattr(models, config.model_name)(num_classes=config.num_classes)
    if args.ckpt: model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
    model = model.to(device)

    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)

    # loss function
    criterion = None
    if 'loss_function' not in dir(config) or config.loss_function == 'WeightedCrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()

    for threshold in list_threhold:
        val_loss, val_f1, val_acc, val_sen, val_precision = val_epoch(model, criterion, val_dataloader)
        print('val_loss:%0.3e val_f1:%.3f, val_acc:%.3f val_sen:%.3f, val_prec:%.3f\n' % (val_loss, val_f1, val_acc, val_sen, val_precision))


def get_motified_transfered_model(model: nn.Module, transfer_model_name: str, train: bool = True) -> None:
    if 'trv1' in transfer_model_name:
        if transfer_model_name == 'trv1_fv1':
            #  trv1只将最后的全连接层成新的全连接层
            #  fv1冻结新加的全连接层以外的部分的参数
            if train:
                for param in model.parameters():
                    param.requires_grad = False
        elif transfer_model_name == 'trv1_fv2':
            #  trv1只将最后的全连接层成新的全连接层
            #  fv2不冻结任何参数
            pass

        inchannel = model.fc.in_features
        model.fc = nn.Linear(inchannel, config.num_classes)
    elif 'trv2' in transfer_model_name:
        if transfer_model_name == 'trv2_fv1':
            #  trv2删掉平均池化，加入一组共两个残差块，步长为2
            #  fv1冻结新加的层以外的部分的参数
            if train:
                for param in model.parameters():
                    param.requires_grad = False
        elif transfer_model_name == 'trv2_fv2':
            #  trv2删掉平均池化，加入一组共两个残差块，步长为2
            #  fv2不冻结任何参数
            pass
        from models.resnet import BasicBlock
        inchannel = model.fc.in_features
        newavgpool_layer = model._make_layer(BasicBlock, inchannel * 2, 2, stride=2)
        newavgpool_layer.add_module('newavgpool', nn.AdaptiveAvgPool1d(1))
        model.avgpool = newavgpool_layer
        for m in model.avgpool.modules():  # 为新加的残差块初始化权重
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        model.fc = nn.Linear(inchannel * 2, config.num_classes)

    print(model)


def trans_train(args):
    pretrained_state = torch.load(config.pretrained_model_ckpt, map_location='cpu')
    pretrained_num_classes = pretrained_state['state_dict']['fc.weight'].shape[0]

    model = getattr(models, config.model_name)(num_classes=pretrained_num_classes)
    model.load_state_dict(pretrained_state['state_dict'])

    get_motified_transfered_model(model, config.transfer_model_name)

    model = model.to(device)

    # 模型保存文件夹
    model_save_dir = '%s/%s_%s_%s_%s_%s' % (
    config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"), config.experiment_name, config.transfer_model_name, os.path.basename(os.path.dirname(os.path.abspath(config.pretrained_model_ckpt)))
    )
    if args.ex: model_save_dir += args.ex

    train_procedure(args, model, model_save_dir)


# 测试模型
def test(args):
    sttest = time.time()
    from dataset import transform
    from data_process import name2index
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    utils.mkdirs(config.sub_dir)

    # model
    model = getattr(models, config.model_name)(num_classes=config.num_classes)


    if 'transfer_model_name' in dir(config):
        get_motified_transfered_model(model, config.transfer_model_name, train=False)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
    model = model.to(device)
    model.eval()

    savedir = os.path.dirname(args.ckpt)
    sub_file = os.path.join(savedir, # os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))+'_'+
                            os.path.basename(args.ckpt)[:-4]
                            + '_' + os.path.basename(config.test_label)[:-4] +config.test_expri_name+ '.txt')
    fout = open(sub_file, 'w', encoding='utf-8')
    predicttime = 0
    test_cnt = 0
    with torch.no_grad():
        for line in tqdm(open(config.test_label, encoding='utf-8')):
            strlist = line.split(',')
            rid = strlist[0].strip()
            fout.write(rid)
            file_path = os.path.join(config.test_dir, rid+'.npy')
            x = np.load(file_path)
            x.astype(np.float32)
            if config.data_standardization:
                x = (x-np.mean(x))/np.std(x)
            # x = x * config.inputUnit_uv / config.targetUnit_uv  # 符值转换
            x = x.T
            test_cnt+=1
            sttime = time.time()
            x = transform(x).unsqueeze(0).to(device)
            output = torch.softmax(model(x), dim=1).squeeze()
            output = torch.softmax(output, dim=0, dtype=torch.float32)
            y_pred = torch.argmax(output, dim=0, keepdim=False)
            y_pred = y_pred.item()

            predicttime += time.time() - sttime

            fout.write(',{}\n'.format(idx2name[y_pred]))
    print('Full time{}s'.format(time.time()-sttest))
    print('Predict time without file operation for {} records: {}s'.format(test_cnt, predicttime))
    print('Predict time without file operation per: {}s'.format(predicttime / test_cnt))
    fout.close()

    ### 测试结果
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    true_dict = file2index(config.test_label, name2idx)
    predict_dict = file2index(sub_file, name2idx)

    idxTP = np.zeros(len(idx2name), dtype=np.int32)
    idxTN = np.zeros(len(idx2name), dtype=np.int32)
    idxFP = np.zeros(len(idx2name), dtype=np.int32)
    idxFN = np.zeros(len(idx2name), dtype=np.int32)
    idxcnt = np.zeros(len(idx2name), dtype=np.int32)

    multiClassConfusionMatrix = np.zeros(shape=(len(idx2name), len(idx2name)), dtype=np.int32)

    cnt_correct = 0

    for rid in true_dict.keys():
        y = true_dict[rid]
        pre_y = predict_dict[rid]

        if y == pre_y:
            cnt_correct += 1

        multiClassConfusionMatrix[y[0]][pre_y[0]] += 1

        for idx in range(len(idx2name)):
            if idx in y:
                idxcnt[idx] += 1
                if idx in pre_y:
                    idxTP[idx] += 1
                else:
                    idxFN[idx] += 1
            else:
                if idx in pre_y:
                    idxFP[idx] += 1
                else:
                    idxTN[idx] += 1

    specificity = idxTN / (idxTN + idxFP)
    sensitivity = idxTP / (idxTP + idxFN)  # recall
    precision = idxTP / (idxTP + idxFP)
    accuracy = (idxTP + idxTN) / (idxTP + idxTN + idxFP + idxFN)
    idxF1 = 2 * precision * sensitivity / (precision + sensitivity)

    sumTP = np.sum(idxTP)
    sumFP = np.sum(idxFP)
    sumFN = np.sum(idxFN)
    sumTN = np.sum(idxTN)
    microF1 = 2 * sumTP / (2 * sumTP + sumFP + sumFN)
    microSpecificity = sumTN / (sumTN + sumFP)
    microSensitivity = sumTP / (sumTP + sumFN)  # recall
    microPrecision = sumTP / (sumTP + sumFP)
    testdata_num = len(true_dict)
    allAccuracy = cnt_correct / testdata_num


    with open(sub_file.split('.txt')[0] + '.csv', 'w', encoding='GB2312') as output:
        output.write('idx,arraythmia,quantity,TP,TN,FP,FN,specificity,sensitivity,precision,accuracy,F1,\n')
        for idx in range(len(idx2name)):
            output.write(
                "{},{},{},{},{},{},{},{},{},{},{},{},\n".format(idx, idx2name[idx], idxcnt[idx], idxTP[idx], idxTN[idx],
                                                                idxFP[idx], idxFN[idx], specificity[idx],
                                                                sensitivity[idx], precision[idx], accuracy[idx],
                                                                idxF1[idx]))
        output.write('microF1:{},\n'.format(microF1))
        output.write('testdata_number:{},\n'.format(testdata_num))
        output.write('{},{},{},{},{},{},{},{},{},{},{},{},\n'.format('', '', '', sumTP, sumTN,
                                                                sumFP, sumFN, microSpecificity, microSensitivity, microPrecision, allAccuracy,
                                                                microF1))

        output.write('\n\n\n\n')

        output.write(',')
        for i in range(len(name2idx)):
            output.write(',{}'.format(i))
        output.write('\n')
        output.write(',')
        for i in range(len(idx2name)):
            output.write(',{}'.format(idx2name[i]))
        output.write('\n')
        for i in range(len(idx2name)):
            output.write('{},{}'.format(i, idx2name[i]))
            for j in range(len(idx2name)):
                output.write(',{}'.format(multiClassConfusionMatrix[i][j]))
            output.write('\n')
def test_24h(args):
    pass


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="train or infer or trans_train")
    parser.add_argument("--ckpt", type=str, help="the path of model weight file")
    parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()
    if (args.command == "train"):
        train(args)
    if (args.command == "test"):
        test(args)
    if (args.command == "val"):
        val(args)
    if (args.command == 'trans_train'):
        trans_train(args)
