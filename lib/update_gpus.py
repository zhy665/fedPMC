#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
from losses import ConLoss
from utils import add_noise_img
from torchstat import stat
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Optimizer
from torch.utils.data import Subset
import torch
import torch.distributed as dist
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
from utils import agg_func

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class MySGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None, beta = 0):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(-beta, d_p)
                else:     
                    p.data.add_(-group['lr'], d_p)
        return loss

class SVRG(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SVRG, self).__init__(params, defaults)

    def step(self, full_grad, pre_old_grad):
        """
            full_grad:μ full grad
            pre_old_grad:pre old grad
            p.grad:new_grad
        """
        for group in self.param_groups:
            for p, full, pre in zip( group['params'], full_grad, pre_old_grad):
                p.data = p.data - group['lr'] *(p.grad.data - pre + full)

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if isinstance(image, torch.Tensor):
            image = image.clone().detach()
        else:
            # image = torch.tensor(image)
            image = image
        if isinstance(label, torch.Tensor):
            label = label.clone().detach()
        else:
            label = torch.tensor(label)
        return image, label

def process_data(data):
    # 创建一个空字典来存储结果
    result_dict = {}

    # 遍历数据
    for item in data:
        labels = item[2]  # 获取标签
        features = item[1]  # 获取特征

        # 如果标签不在字典中，则将其作为键添加到字典中
        # 并将特征作为值，初始化为一个列表
        for feature, label in zip(features, labels):
            if label.item() not in result_dict:
                result_dict[label.item()] = []

            # 将特征添加到对应标签的列表中
            result_dict[label.item()].append(feature)

    # 遍历字典，计算每个标签对应特征的平均值
    for label, feature_list in result_dict.items():
        # 将特征列表转换为一个张量，维度为 (样本数, 特征维度)
        features_tensor = torch.stack(feature_list)

        # 计算特征的平均值，沿着第一个维度求平均，即对每个特征维度求平均
        average_features = torch.mean(features_tensor, dim=0)

        # 将平均特征作为对应标签的值存储到字典中
        result_dict[label] = average_features

    return result_dict

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = args.device
        self.criterion_CE = nn.NLLLoss().to(self.device)

    def average_model_state_dicts(self, state_dicts):
        avg_state_dict = copy.deepcopy(state_dicts[0])
        for key in avg_state_dict.keys():
            for i in range(1, len(state_dicts)):
                avg_state_dict[key] += state_dicts[i][key]
            avg_state_dict[key] = avg_state_dict[key] / len(state_dicts)
        return avg_state_dict
    
    def average_dicts(self, dicts):
        avg_dict = {}
        all_keys = set(key for d in dicts for key in d)
        for key in all_keys:
            # 取得所有包含当前 key 的字典中对应的值
            values = [d[key] for d in dicts if key in d]            
            # 如果 values 中的元素是列表
            if isinstance(values[0], list):
                # 将所有张量连接起来
                combined_tensor = []
                for value in values:
                    combined_tensor += value
                avg_dict[key] = combined_tensor
            else:
                # 否则，直接计算平均值
                avg_dict[key] = sum(values) / len(values)
        return avg_dict  

    def run_training_pcl(self, args, idx, global_protos, global_avg_protos, backbone_list, model, global_round):
        world_size = 4  # Number of GPUs
        manager = mp.Manager()
        return_dict = manager.dict()
        mp.spawn(self.update_weights_pcl, args=(world_size, args, idx, global_protos, global_avg_protos, backbone_list, model,global_round,  return_dict), nprocs=world_size, join=True)
        # Gather results from the return_dict
        model_state_dicts,urt,epoch_losses,agg_protos_labels = zip(*return_dict.values())
        model_state_dicts = self.average_model_state_dicts(model_state_dicts)
        # print(model_state_dicts.keys())
        urt = []
        epoch_losses = self.average_dicts(epoch_losses)
        agg_protos_labels = agg_func(self.average_dicts(agg_protos_labels))
        # 释放 Manager 资源
        manager.shutdown()
        return model_state_dicts, urt, epoch_losses, agg_protos_labels

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        return trainloader

    def update_weights_solo(self, idx, backbone_list, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        # Set optimizer for the local updates
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)
        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images = images[0]
                images, labels = images.to(self.device), labels.to(self.device)

                # generate representations by different backbone
                for i in range(len(backbone_list)):
                    backbone = backbone_list[i]
                    if i == 0:
                        reps = backbone(images)
                    else:
                        reps = torch.cat((reps, backbone(images)), 1)

                # one steps SGD
                optimizer.zero_grad()
                log_probs, _ = model(reps)
                loss = self.criterion_CE(log_probs, labels)
                loss.backward(retain_graph=True)
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_fedavg(self, idx, backbone_list, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images = images[0]
                images, labels = images.to(self.device), labels.to(self.device)

                # generate representations by different backbone
                for i in range(len(backbone_list)):
                    backbone = backbone_list[i]
                    if i == 0:
                        reps = backbone(images)
                    else:
                        reps = torch.cat((reps, backbone(images)), 1)

                # one steps SGD
                model.zero_grad()
                log_probs, _ = model(reps)
                loss = self.criterion_CE(log_probs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_pcl(self, rank, world_size, args, idx, global_protos, global_avg_protos, backbone_list, model, global_round, return_dict):
        if torch.cuda.device_count() > 1:
            setup(rank, world_size)
            # Set mode to train model
            model = model.to(rank)
            model = DDP(model, device_ids=[rank])
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[]}
        loss_mse = nn.MSELoss().to(args.device)
        criterion_CL = ConLoss(temperature=0.07)
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)
        for iter in range(self.args.train_ep):
            batch_loss = {'1':[],'2':[],'total':[]}
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if args.add_noise_img:
                    images[0] = add_noise_img(images[0], args.scale, args.perturb_coe, args.noise_type)
                    images[1] = images[0]
                images = torch.cat([images[0], images[1]], dim=0)
                images, labels = images.to(rank), labels.to(rank)
                model.zero_grad()
                try:
                    log_probs, features = model(images)
                except Exception as e:
                    continue  # Skip this batch if there is an error
                # print("Model forward pass successful")
                bsz = labels.shape[0]
                lp1, lp2 = torch.split(log_probs, [bsz, bsz], dim=0)
                loss1 = self.criterion_CE(lp1, labels)
                # compute regularized loss term
                loss2 = 0 * loss1
                if len(global_protos) == args.num_users:
                    # compute global proto based CL loss
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                    for i in range(args.num_users):
                        for label in global_avg_protos.keys():
                            if label not in global_protos[i].keys():
                                global_protos[i][label] = global_avg_protos[label]
                        loss2 += criterion_CL(features, labels, global_protos[i])
                    loss2 /= args.num_users
                    loss2 += criterion_CL(features, labels, global_avg_protos)
                # print("loss2 is ok")
                loss = loss2
                # SGD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
                batch_loss['total'].append(loss.item())
                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tLoss2: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images) // 2,
                                    len(self.trainloader.dataset),
                                    100. * batch_idx / len(self.trainloader),
                                    loss.item(),
                                    loss2.item()))
                # 在每个循环结束后
                gc.collect()
                torch.cuda.empty_cache()
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))

        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])

        # generate representation
        agg_protos_label = {}
        model.eval()
        for batch_idx, (images, label_g) in enumerate(self.trainloader):
            images = images[0]
            images, labels = images.to(rank), label_g.to(rank)
            _, features = model(images)
            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label:
                    agg_protos_label[labels[i].item()].append(features[i, :])
                else:
                    agg_protos_label[labels[i].item()] = [features[i, :]]
        model_cpu_state_dict = model.cpu().state_dict()
        detached_agg_protos_label = {k: [t.detach().cpu() for t in v] for k, v in agg_protos_label.items()}
        return_dict[rank] = (model_cpu_state_dict, [], epoch_loss, detached_agg_protos_label)
        gc.collect()
        cleanup()   

    def update_weights_PMe(self, args, idx, optimizer, backbone_list, local_model,  model, global_model,  global_round=round):
        for old_param, new_param, local_param in zip(model.parameters(), global_model.parameters(),local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
        model.train()
        epoch_loss = {'total':[]}
        for iter in range(self.args.train_ep):
            batch_loss = {'total':[]}
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                model.train()
                if args.add_noise_img:
                    images[0] = add_noise_img(images[0], args.scale, args.perturb_coe, args.noise_type)
                    images[1] = images[0]
                images, labels = images[0].to(self.device), labels.to(self.device)
                # generate representations by different backbone
                with torch.no_grad():
                    for i in range(len(backbone_list)):
                        backbone = backbone_list[i]
                        if i == 0:
                            reps = backbone(images)
                        else:
                            reps = torch.cat((reps, backbone(images)), 1)   
                # 获取当前模型的参数
                # current_params = [param.clone() for param in model.parameters()]
                for i in range(self.args.K):
                    # 清零梯度
                    optimizer.zero_grad()
                    # 计算梯度
                    log_probs, features = model(reps)
                    loss = self.criterion_CE(log_probs, labels)
                    loss.backward()
                    # 获取计算出的梯度
                    gradients, _ = optimizer.step(local_model)
                # update local weight after finding aproximate theta
                for new_param, localweight in zip(gradients, local_model):
                    localweight.data = localweight.data - self.args.lamda* self.args.lr * (localweight.data - new_param.data)
                    
                
                batch_loss['total'].append(loss.item())
                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images) // 2,
                                    len(self.trainloader.dataset),
                                    100. * batch_idx / len(self.trainloader),
                                    loss.item(),))
            for param , new_param in zip(model.parameters(), local_model):
                param.data = new_param.data.clone()
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        
        return model.state_dict(), epoch_loss['total']

    def update_weights_peravg(self, idx, backbone_list, model, global_model, global_round):  

        # Set mode to train model
        model.train()
        epoch_loss = []
        model = copy.deepcopy(global_model)
        optimizer = MySGD(model.parameters(), lr=self.args.lr)
        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if batch_idx == len(self.trainloader) - 1 and batch_idx % 2 == 0:
                    # 说明batch的数量是奇数，最后一个batch的数据不参与训练
                    break
                if batch_idx % 2 == 0:
                    # step1
                    temp_model = copy.deepcopy(list(model.parameters()))
                    images = images[0]
                    images, labels = images.to(self.device), labels.to(self.device)
                    # generate representations by different backbone
                    for i in range(len(backbone_list)):
                        backbone = backbone_list[i]
                        if i == 0:
                            reps = backbone(images)
                        else:
                            reps = torch.cat((reps, backbone(images)), 1)
                    # compute CE loss
                    optimizer.zero_grad()
                    log_probs, _ = model(reps)
                    loss = self.criterion_CE(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    # step 2
                    images = images[0]
                    images, labels = images.to(self.device), labels.to(self.device)
                    # generate representations by different backbone
                    for i in range(len(backbone_list)):
                        backbone = backbone_list[i]
                        if i == 0:
                            reps = backbone(images)
                        else:
                            reps = torch.cat((reps, backbone(images)), 1)
                    # compute CE loss
                    optimizer.zero_grad()
                    log_probs, _ = model(reps)
                    loss = self.criterion_CE(log_probs, labels)
                    loss.backward()
                    # restore the model parameters to the one before first update
                    for old_p, new_p in zip(model.parameters(), temp_model):
                        old_p.data = new_p.data.clone()
                    optimizer.step(beta = self.args.beta)
                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def update_weights_fedrep(self, idx, backbone_list, model, global_model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        # 加载服务器模型的表示层参数
        model.fc1.load_state_dict(global_model.fc1.state_dict())
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)
        for iter in range(self.args.train_ep):
            batch_loss = []
            
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images = images[0]
                images, labels = images.to(self.device), labels.to(self.device)
            
                # generate representations by different backbone
                for i in range(len(backbone_list)):
                    backbone = backbone_list[i]
                    if i == 0:
                        reps = backbone(images)
                    else:
                        reps = torch.cat((reps, backbone(images)), 1)
                # update head layer , freezing representation layer
                for i in range(self.args.K):
                    # compute CE loss
                    optimizer.zero_grad()
                    model.fc1.requires_grad = False
                    model.fc2.requires_grad = True
                    log_probs, _ = model(reps)
                    loss = self.criterion_CE(log_probs, labels)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                # update representation layer , freezing head layer
                for i in range(self.args.k):
                    optimizer.zero_grad()
                    model.fc1.requires_grad = True
                    model.fc2.requires_grad = False
                    log_probs, _ = model(reps)
                    loss = self.criterion_CE(log_probs, labels)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.fc1.state_dict(), model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def update_weights_fedproto(self, args, idx, local_model, global_protos,global_avg_protos,  backbone_list, model, global_round=round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[]}
        
        loss_mse = nn.MSELoss().to(args.device)
        criterion_CL = ConLoss(temperature=0.07)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)

        for iter in range(self.args.train_ep):
            batch_loss = {'1':[],'2':[],'total':[]}
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if args.add_noise_img:
                    images[0] = add_noise_img(images[0], args.scale, args.perturb_coe, args.noise_type)
                    images[1] = images[0]
                images, labels = images[0].to(self.device), labels.to(self.device)

                # generate representations by different backbone
                with torch.no_grad():
                    for i in range(len(backbone_list)):
                        backbone = backbone_list[i]
                        if i == 0:
                            reps = backbone(images)
                        else:
                            reps = torch.cat((reps, backbone(images)), 1)

                # compute supervised contrastive loss
                model.zero_grad()
                log_probs, features = model(reps)
                loss1 = self.criterion_CE(log_probs, labels)
                # features denote C_i
                
                # compute regularized loss term
                loss2 = 0 * loss1
                if len(global_protos) == args.num_users:
                    # compute global proto-based distance loss
                    num, xdim = features.shape
                    features_global = torch.zeros_like(features)
                    for i, label in enumerate(labels):
                        features_global[i, :] = copy.deepcopy(global_avg_protos[label.item()].data)
                    loss2 = loss_mse(features_global, features) / num * args.ld
                loss = loss1 + loss2

                # SGD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
                batch_loss['total'].append(loss.item())
                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tLoss2: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images) // 2,
                                    len(self.trainloader.dataset),
                                    100. * batch_idx / len(self.trainloader),
                                    loss.item(),
                                    loss2.item()))

            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))

        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])

        # generate representation
        agg_protos_label = {}
        model.eval()
        for batch_idx, (images, label_g) in enumerate(self.trainloader):
            images = images[0]
            images, labels = images.to(self.device), label_g.to(self.device)

            with torch.no_grad():
                for i in range(len(backbone_list)):
                    backbone = backbone_list[i]
                    if i == 0:
                        reps = backbone(images)
                    else:
                        reps = torch.cat((reps, backbone(images)), 1)
            _, features = model(reps)
            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label:
                    agg_protos_label[labels[i].item()].append(features[i, :])
                else:
                    agg_protos_label[labels[i].item()] = [features[i, :]]

        return model.state_dict(), epoch_loss, agg_protos_label
    
    def update_weights_svrg(self, args, idx, global_protos, global_avg_protos, backbone_list, model, global_round=round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[]}
        loss_mse = nn.MSELoss().to(args.device)
        criterion_CL = ConLoss(temperature=0.07)
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)
        
        for iter in range(self.args.train_ep):
            batch_loss = {'1':[],'2':[],'total':[]}
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if args.add_noise_img:
                    images[0] = add_noise_img(images[0], args.scale, args.perturb_coe, args.noise_type)
                    images[1] = images[0]
                images = torch.cat([images[0], images[1]], dim=0)
                images, labels = images.to(self.device), labels.to(self.device)
                # generate representations by different backbone
                with torch.no_grad():
                    for i in range(len(backbone_list)):
                        backbone = backbone_list[i]
                        if i == 0:
                            reps = backbone(images)
                        else:
                            reps = torch.cat((reps, backbone(images)), 1)
                # compute supervised contrastive loss
                model.zero_grad()
                log_probs, features = model(reps)
                bsz = labels.shape[0]
                lp1, lp2 = torch.split(log_probs, [bsz, bsz], dim=0)
                loss1 = self.criterion_CE(lp1, labels)
                # compute regularized loss term
                loss2 = 0 * loss1
                if len(global_protos) == args.num_users:
                    # compute global proto based CL loss
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    for i in range(args.num_users):
                        for label in global_avg_protos.keys():
                            if label not in global_protos[i].keys():
                                global_protos[i][label] = global_avg_protos[label]
                        loss2 += criterion_CL(features, labels, global_protos[i])
                    loss2 /= args.num_users
                    loss2 += criterion_CL(features, labels, global_avg_protos)
                    loss = loss2
                    # svrg
                    # 保留全梯度
                    # 以字典的形式保留全梯度
                    if batch_idx == 0:
                        optimizer.zero_grad()
                        loss.backward()
                        pre_old_grad = torch.cat([param.grad.data.flatten() for param in model.parameters()])
                        full_grad = torch.cat([param.grad.data.flatten() for param in model.parameters()])
                    elif batch_idx % 10 != 0:
                        optimizer.zero_grad()
                        loss.backward()
                        full_grad += torch.cat([param.grad.data.flatten() for param in model.parameters()])
                    else:
                        # update grad
                        full_grad /= 10
                        optimizer.zero_grad()
                        loss.backward()
                        for p, full, pre in zip(model.parameters(), full_grad, pre_old_grad):
                            p.data = p.data - self.args.lr *(p.grad.data - pre + full)
                        pre_old_grad = torch.cat([param.grad.data.flatten() for param in model.parameters()])
                        full_grad = torch.cat([param.grad.data.flatten() for param in model.parameters()])
                    
                else:
                    loss = loss2
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
                batch_loss['total'].append(loss.item())
                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss1: {:.3f}\tLoss: {:.3f}\t'.format(
                            global_round, idx, iter, batch_idx * len(images) // 2,
                                    len(self.trainloader.dataset),
                                    100. * batch_idx / len(self.trainloader),
                                    loss1.item(),
                                    loss.item()))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        # generate representation
        agg_protos_label = {}
        model.eval()
        for batch_idx, (images, label_g) in enumerate(self.trainloader):
            images = images[0]
            images, labels = images.to(self.device), label_g.to(self.device)
            with torch.no_grad():
                for i in range(len(backbone_list)):
                    backbone = backbone_list[i]
                    if i == 0:
                        reps = backbone(images)
                    else:
                        reps = torch.cat((reps, backbone(images)), 1)
            _, features = model(reps)
            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label:
                    agg_protos_label[labels[i].item()].append(features[i, :])
                else:
                    agg_protos_label[labels[i].item()] = [features[i, :]]

        return model.state_dict(), [], epoch_loss, agg_protos_label
    
    def update_weights_ours(self, args, idx, global_protos, global_avg_protos, backbone_list, model, global_round=round):
        # Set mode to train model
        model_k = copy.deepcopy(model)
        model_k.train()
        model.train()
        
        epoch_loss = {'total':[],'1':[], '2':[]}
        loss_mse = nn.MSELoss().to(args.device)
        criterion_CL = ConLoss(temperature=0.07)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)
        
        queue = []
        
        for iter in range(self.args.train_ep):
            batch_loss = {'1':[],'2':[],'total':[]}
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if args.add_noise_img:
                    images[0] = add_noise_img(images[0], args.scale, args.perturb_coe, args.noise_type)
                    images[1] = images[0]
                images = torch.cat([images[0], images[1]], dim=0)
                images, labels = images.to(self.device), labels.to(self.device)
                model_k = copy.deepcopy(model)
                # generate representations by different backbone
                with torch.no_grad():
                    for i in range(len(backbone_list)):
                        backbone = backbone_list[i]
                        if i == 0:
                            reps = backbone(images[:self.args.local_bs])
                            reps_k = backbone(images[self.args.local_bs:])
                        else:
                            reps = torch.cat((reps, backbone(images[:self.args.local_bs])), 1)
                            reps_k =  torch.cat((reps_k, backbone(images[self.args.local_bs:])), 1)
                # compute supervised contrastive loss
                model.zero_grad()
                log_probs, features = model(reps)
                log_probs_k, features_k = model_k(reps_k)
                model_k.detach()
                loss1 = self.criterion_CE(log_probs, labels)
                loss2 = 0 * loss1
                if batch_idx < self.args.moco_k:
                    queue.append([model_k.state_dict(), features_k, labels])
                else:
                    # compute regularized loss term
                    if len(global_protos) == args.num_users:
                        # compute global proto based CL loss
                        f1, f2 = features, features_k
                        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                        # update way 1:
                        concatenated_tensor = torch.cat([item[1] for item in queue], dim=0)
                        concatenated = torch.cat([f1, concatenated_tensor], dim=0)
                        concatenated_local = torch.cat([features_k, concatenated_tensor], dim=0)
                        features_local = torch.cat([concatenated.unsqueeze(1), concatenated_local.unsqueeze(1)], dim=1)
                        label_con = torch.cat([item[2] for item in queue], dim=0)
                        labels_local = torch.cat([labels, label_con], dim=0)    
                        # update way 2:
                        global_protos[idx] = process_data(queue)
                        for i in range(args.num_users):
                            for label in global_avg_protos.keys():
                                if label not in global_protos[i].keys():
                                    global_protos[i][label] = global_avg_protos[label]
                            loss2 += criterion_CL(features_local, labels_local, global_protos[i])
                        loss2 /= args.num_users
                        loss2 += criterion_CL(features, labels, global_avg_protos)
                        loss = loss2
                        
                        # v3
                        # for label in global_avg_protos.keys():
                        #     if label not in global_protos[idx].keys():
                        #         global_protos[idx][label] = global_avg_protos[label]
                        # loss2 += criterion_CL(features_local, labels_local, global_protos[idx])
                        # loss2 += criterion_CL(features, labels, global_avg_protos)
                        # loss = loss2
                        
                        # adam for query encoder
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        # update model_k
                        for new_param, old_param in zip(model_k.parameters(), model.parameters()):
                            new_param.data = self.args.moco_m * new_param.data + (1-self.args.moco_m)*old_param.data
                        # update queue
                        queue.append([model_k.state_dict(), features_k, labels])
                        queue.pop()
                loss = loss2

                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
                batch_loss['total'].append(loss.item())
                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss1: {:.3f}\tLoss: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images) // 2,
                                    len(self.trainloader.dataset),
                                    100. * batch_idx / len(self.trainloader),
                                    loss1.item(),
                                    loss.item()))

            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))

        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])

        # generate representation
        agg_protos_label = {}
        model_k.eval()
        for batch_idx, (images, label_g) in enumerate(self.trainloader):
            images = images[0]
            images, labels = images.to(self.device), label_g.to(self.device)
            with torch.no_grad():
                for i in range(len(backbone_list)):
                    backbone = backbone_list[i]
                    if i == 0:
                        reps = backbone(images)
                    else:
                        reps = torch.cat((reps, backbone(images)), 1)
            _, features = model_k(reps)
            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label:
                    agg_protos_label[labels[i].item()].append(features[i, :])
                else:
                    agg_protos_label[labels[i].item()] = [features[i, :]]

        return model_k.state_dict(), [], epoch_loss, agg_protos_label

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion_SupCL(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy = correct/total
        return accuracy, loss

    def generate_protos(self, backbone_list, model):
        model.eval()
        agg_protos_label = {}
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images = images[0]
            images, labels = images.to(self.device), labels.to(self.device)
            for i in range(len(backbone_list)):
                backbone = backbone_list[i]
                if i == 0:
                    reps = backbone(images)
                else:
                    reps = torch.cat((reps, backbone(images)), 1)
            _, features = model(reps)
            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label:
                    agg_protos_label[labels[i].item()].append(features[i, :])
                else:
                    agg_protos_label[labels[i].item()] = [features[i, :]]
        return agg_protos_label

class LocalTest(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.testloader = self.test_split(args, dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(args.device)

    def test_split(self, args, dataset, idxs):
        idxs_test = idxs[:int(1 * len(idxs))]

        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                 batch_size=args.test_bs, shuffle=False)
        return testloader

    def test_inference(self, idx, args, backbone_list, local_model):
        device = args.device
        criterion = nn.NLLLoss().to(device)

        model = local_model
        model.to(args.device)
        loss, total, correct = 0.0, 0.0, 0.0

        # test (only use local model)
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(device), labels.to(device)

            # generate representations by different backbone
            for i in range(len(backbone_list)):
                backbone = backbone_list[i]
                if i == 0:
                    reps = backbone(images)
                else:
                    reps = torch.cat((reps, backbone(images)), 1)
            probs, _ = model(reps)
            # prediction
            _, pred_labels = torch.max(probs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        loss /= (batch_idx + 1)
        print('| User: {} | Test Acc: {:.5f} | Test Loss: {:.5f}'.format(idx, acc, loss))

        return acc, loss

    def test_inference_twoway(self, idx, args, global_protos, local_protos, backbone_list, local_model):
        device = args.device
        criterion = nn.NLLLoss().to(device)
        loss_mse = nn.MSELoss().to(device)

        model = local_model
        model.to(args.device)
        loss, total, correct = 0.0, 0.0, 0.0
        # 确保 local_protos[j] 在同一个设备上
        for j in range(args.num_classes):
            if j in local_protos.keys():
                local_protos[j] = local_protos[j].to(device)
        # test (only use local model)
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(device), labels.to(device)

            # # generate representations by different backbone
            # for i in range(len(backbone_list)):
            #     backbone = backbone_list[i]
            #     if i == 0:
            #         reps = backbone(images)
            #     else:
            #         reps = torch.cat((reps, backbone(images)), 1)

            probs, features = model(images)
            feature = features.to(device)
            # compute the dist between features and input protos
            a_large_num = 100
            dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
            for i in range(images.shape[0]):
                for j in range(args.num_classes):
                    if j in local_protos.keys():
                        d = loss_mse(features[i, :], local_protos[j]) # compare with local protos
                        dist[i, j] = d
            _, pred_labels = torch.min(dist, 1)

            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        loss /= (batch_idx + 1)
        print('| User: {} | Test Acc: {:.5f} | Test Loss: {:.5f}'.format(idx, acc, loss))

        return acc, loss

def save_protos(round, args, backbone_list, local_model_list, train_dataset_list, user_groups, global_protos):
    """ Returns the test accuracy and loss.
    """
    device = args.device

    agg_protos_label = {}
    for idx in range(1):
        agg_protos_label[idx] = {}
        model = local_model_list[idx]
        model.eval()
        model.to(args.device)
        trainloader = DataLoader(DatasetSplit(train_dataset_list[idx], user_groups[idx]), batch_size=32, shuffle=True, drop_last=True)
        for batch_idx, (images, labels) in enumerate(trainloader):
            images = images[0]
            images, labels = images.to(device), labels.to(device)

            # generate representations by different backbone
            for i in range(len(backbone_list)):
                backbone = backbone_list[i]
                if i == 0:
                    reps = backbone(images)
                else:
                    reps = torch.cat((reps, backbone(images)), 1)

            # compute features
            model.zero_grad()
            _, protos = model(reps)

            for k in range(len(labels)):
                if labels[k].item() in agg_protos_label[idx].keys():
                    agg_protos_label[idx][labels[k].item()].append(protos[k, :])
                else:
                    agg_protos_label[idx][labels[k].item()] = [protos[k, :]]

    x = []
    y = []
    u = []
    for idx in range(1):
        for label in [0]:
            for proto in agg_protos_label[idx][label]:
                if args.device == 'cuda':
                    tmp = proto.cpu().detach().numpy()
                else:
                    tmp = proto.detach().numpy()
                x.append(tmp)
                y.append(label)
                u.append(round)
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    np.save('./save/local_protos_' + str(round) + 'r_' + args.alg + '_protos.npy', x)
    np.save('./save/local_protos_' + str(round) + 'r_' + args.alg + '_labels.npy', y)
    np.save('./save/local_protos_' + str(round) + 'r_' + args.alg + '_rounds.npy', u)

    xx = []
    yy = []
    uu = []
    for label in [0]:
        if args.device == 'cuda':
            xx.append(global_protos[label].cpu().detach().numpy())
        else:
            xx.append(global_protos[label].detach().numpy())
        yy.append(label)
        uu.append(round)
    np.save('./save/global_protos_' + str(round) + 'r_' + args.alg + '_protos.npy', xx)
    np.save('./save/global_protos_' + str(round) + 'r_' + args.alg + '_labels.npy', yy)
    np.save('./save/global_protos_' + str(round) + 'r_' + args.alg + '_rounds.npy', uu)

    print("Save protos and labels successfully.")
