import os, sys, copy
import time, random
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from pathlib import Path
from datetime import datetime
import torch.nn as nn
from collections import OrderedDict
import seaborn as sns
lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from models.resnet import resnet18
import matplotlib.pyplot as plt
# from models.vision_transformer import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224
from options import args_parser
from update import LocalUpdate, LocalTest
# from update_gpus import LocalUpdate, LocalTest
from models.models import ProjandDeci, ProjandDeci_backbones
from models.multibackbone import alexnet, vgg11, mlp_m
from utils import add_noise_proto, prepare_data_real_noniid, prepare_data_domainnet_noniid, prepare_data_office_noniid, prepare_data_digits_noniid, prepare_data_caltech_noniid, prepare_data_mnistm_noniid, average_weights, exp_details, proto_aggregation, agg_func, prepare_data_digits, prepare_data_office, prepare_data_domainnet
from torch.optim import Optimizer
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import gc


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu = mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)
    
    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:                                                                                                                      
            for p, localweight in zip( group['params'], weight_update):
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu']*p.data)
        return  group['params'], loss
    
    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = localweight.data
        #return  p.data
        return  group['params']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def plot_loss(args, loss_list):
    plt.figure()
    plt.plot(loss_list, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'lr:{args.lr}, weight_decay:{args.weight_decay},{args.alg} Training Loss Curve')
    plt.legend()
    # 保存图像
    plt.savefig(f'images/{args.alg}-training_loss_curve-{args.lr}-{args.weight_decay}.jpg', dpi=500, format='jpg')
    # plt.show()

def plot_heapmap(args, lr, wd, acc_list):
    plt.figure()
    ax = sns.heatmap(acc_list, xticklabels=wd, yticklabels=lr, cmap='magma')
    ax.set_title('Optimal accuracy for validation')  # 图标题
    ax.set_xlabel('weight decay')  # x轴标题
    ax.set_ylabel('learning rate')
    # plt.show()
    figure = ax.get_figure()
    figure.savefig(f'images/{args.alg}-heat_map.jpg', dpi=500)  # 保存图片

def plot_heapmap2(args, lr, wd, acc_list):
    plt.figure()
    if wd:
        wds, lrs= np.meshgrid(np.arange(len(wd)), np.arange(len(lr)))
        # wd, lr = np.meshgrid(wd, lr)
    else:
        wds, lrs = np.meshgrid(np.arange(len(lr)), np.arange(len(lr)))
        acc_list,_ = np.meshgrid(acc_list, acc_list)
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(wds, lrs, acc_list, levels=300, cmap=plt.cm.hot)
    cbar = plt.colorbar(contour)
    cbar.set_label('Optimal accuracy', rotation=270, labelpad=20)
    if wd:
        plt.xlabel('weight decay')
        plt.ylabel('learning rate')
    else:
        plt.xlabel('learning rate')
        plt.ylabel('learning rate')
    plt.title('Optimal accuracy for validation')
    plt.xticks(ticks=np.unique(wds), labels=np.unique(wds))
    plt.yticks(ticks=np.unique(lrs), labels=np.unique(lrs))
    # plt.show()
    plt.savefig(f'images/{args.alg}-heat_map2.jpg', format='png', dpi=500)

def FedAvg(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list):
    train_loss, train_accuracy = [], []
    global_model = local_model_list[0]
    avg_acc = 0
    step = 0
    acc_list = []
    acc_mtx = torch.zeros([args.num_users])
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round} |\n')
        print(datetime.now())
        if args.num_users <= 20:
            idxs_users = np.arange(args.num_users)
        else:
            idxs_users = np.random.choice(args.num_users, 20, replace=False)
        for idx in idxs_users:
            local_node = LocalUpdate(args=args, dataset=train_dataset_list[idx],idxs=user_groups[idx])
            w, loss = local_node.update_weights_fedavg(idx, backbone_list=backbone_list, model=copy.deepcopy(global_model), global_round=round)
            params = sum(p.numel() for p in w.values())
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            summary_writer.add_scalar('Train/Loss/user' + str(idx), loss, round)
            local_model_list[idx].load_state_dict(local_weights[idx], strict=True)
        # update global weights
        local_weights_list = average_weights(local_weights)
        global_model = copy.deepcopy(local_model_list[0])
        global_model.load_state_dict(local_weights_list[0], strict=True)

        loss_avg = sum(local_losses) / len(local_losses)
        if round > 0:
            train_loss.append(loss_avg)
        acc_s = []
        print('| Global Round : {} | Avg Loss: {:.3f}'.format(round, loss_avg))
        if round % 5 == 0:
            with torch.no_grad():
                for idx in range(args.num_users):
                    print('Test on user {:d}'.format(idx))
                    local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
                    # local_model_list[idx] = copy.deepcopy(global_model)
                    local_model_list[idx].eval()
                    acc, loss = local_test.test_inference(idx, args, backbone_list, local_model_list[idx])
                    acc_s.append(acc)
                    summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)
            if sum(acc_s) / len(acc_s) > avg_acc:
                step = round
                avg_acc = sum(acc_s) / len(acc_s)
                acc_list = acc_s
                for idx in range(args.num_users):
                    acc_mtx[idx] = acc_list[idx]
    plot_loss(args, train_loss)
    loss_mtx = torch.zeros([args.num_users])
    acc_s = []
    with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
            # local_model_list[idx] = copy.deepcopy(global_model)
            local_model_list[idx].eval()
            acc, loss = local_test.test_inference(idx, args, backbone_list, local_model_list[idx])
            loss_mtx[idx] = loss
            acc_s.append(acc)
    if sum(acc_s) / len(acc_s) > avg_acc:
        step = round
        avg_acc = sum(acc_s) / len(acc_s)
        acc_list = acc_s
        for idx in range(args.num_users):
            acc_mtx[idx] = acc_list[idx]
    print(f"max_acc is {avg_acc}, step is {step}, acc_list is {acc_list}")
    return acc_mtx

def PFedMe(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, model_list):
    train_loss, train_accuracy = [], []
    global_model = model_list[0]
    local_weights, local_losses = [model.state_dict() for model in model_list], [0 for i in range(len(model_list))]
    local_model_list = [copy.deepcopy(list(model_list[i].parameters())) for i in range(len(model_list))]
    optimizer = [pFedMeOptimizer(model_list[i].parameters(), lr=args.lr,  lamda=args.lamda) for i in range(args.num_users)]
    avg_acc = 0
    step = 0
    acc_list = []
    selected_users = np.arange(args.num_users)
    acc_mtx = torch.zeros([args.num_users])
    for round in tqdm(range(args.rounds)):
        print(f'\n | Global Training Round : {round} |\n')
        print(datetime.now())
        for idx in np.arange(args.num_users):
            local_node = LocalUpdate(args=args, dataset=train_dataset_list[idx],idxs=user_groups[idx])
            w, loss = local_node.update_weights_PMe(args, idx, optimizer=optimizer[idx], backbone_list=backbone_list,model=model_list[idx], local_model=local_model_list[idx], global_model=copy.deepcopy(global_model), global_round=round)
            local_weights[idx] = w
            local_losses[idx] = loss
            summary_writer.add_scalar('Train/Loss/user' + str(idx), loss, round)
        # 客户端数量较少，不进行随机选择
        # selected_users = np.random.choice(np.arange(args.num_users), args.number, replace=False)
        
        # update global weights
        total_train = 0
        ratio = []
        for idx in selected_users:
            total_train += len(train_dataset_list[idx])
        for idx in selected_users:
            ratio.append(len(train_dataset_list[idx])/total_train)
        global_model_list = [local_weights[idx] for idx in selected_users]
        average_state_dict = OrderedDict()
        # 计算参数的平均值
        for param_name in global_model_list[0].keys():
            average_state_dict[param_name] = sum(model[param_name]*ratio[idx] for idx, model in enumerate(global_model_list))
        for old_param, new_param in zip(global_model.parameters(), average_state_dict.values()):
            old_param.data = (1-args.beta)*old_param + new_param*args.beta
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        acc_s = []
        print('| Global Round : {} | Avg Loss: {:.3f}'.format(round, loss_avg))
        if round % 5 == 0:
            with torch.no_grad():
                for idx in range(args.num_users):
                    print('Test on user {:d}'.format(idx))
                    local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
                    local_model_for_test = copy.deepcopy(model_list[idx])
                    local_model_for_test.load_state_dict(local_weights[idx], strict=True)
                    local_model_for_test.eval()
                    acc, loss = local_test.test_inference(idx, args, backbone_list, local_model_for_test)
                    acc_s.append(acc)
                    summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)
            if sum(acc_s) / len(acc_s) > avg_acc:
                step = round
                avg_acc = sum(acc_s) / len(acc_s)
                acc_list = acc_s
                for idx in range(args.num_users):
                    acc_mtx[idx] = acc_list[idx]
    plot_loss(args, train_loss)
    loss_mtx = torch.zeros([args.num_users])
    acc_s = []
    with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
            local_model_for_test = copy.deepcopy(model_list[idx])
            local_model_for_test.load_state_dict(local_weights[idx], strict=True)
            local_model_for_test.eval()
            acc, loss = local_test.test_inference(idx, args, backbone_list, local_model_for_test)
            loss_mtx[idx] = loss
            acc_s.append(acc)
            summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)
    if sum(acc_s) / len(acc_s) > avg_acc:
        step = round
        avg_acc = sum(acc_s) / len(acc_s)
        acc_list = acc_s
        for idx in range(args.num_users):
            acc_mtx[idx] = acc_list[idx]
    print(f"max_acc is {avg_acc}, step is {step}, acc_list is {acc_list}")
    print('For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(torch.mean(acc_mtx), torch.std(acc_mtx)))
    return acc_mtx

def perfedavg(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list):
    train_loss, train_accuracy = [], []
    global_model = local_model_list[0]
    selected_users = np.arange(args.num_users)
    local_weights, local_losses = [model.state_dict() for model in local_model_list], [0 for i in range(len(local_model_list))]
    avg_acc = 0
    step = 0
    acc_list = [0,0,0,0,0]
    acc_mtx = torch.zeros([args.num_users])
    for round in tqdm(range(args.rounds)):
        print(f'\n | Global Training Round : {round} |\n')
        print(datetime.now())
        # update global weights
        for idx in selected_users:
            local_node = LocalUpdate(args=args, dataset=train_dataset_list[idx],idxs=user_groups[idx])
            w, loss = local_node.update_weights_peravg(idx, backbone_list=backbone_list, model=copy.deepcopy(local_model_list[idx]), global_model = global_model, global_round=round)
            local_weights[idx] = w
            local_losses[idx] = loss
            summary_writer.add_scalar('Train/Loss/user' + str(idx), loss, round)
        
        
        # update global weights
        # total_train = 0
        # ratio = []
        # for idx in selected_users:
        #     total_train += len(train_dataset_list[idx])
        # for idx in selected_users:
        #     ratio.append(len(train_dataset_list[idx])/total_train)
        global_model_list = [local_weights[idx] for idx in selected_users]
        average_state_dict = OrderedDict()
        # 计算参数的平均值
        for param_name in global_model_list[0].keys():
            average_state_dict[param_name] = sum(model[param_name] for idx, model in enumerate(global_model_list)) / len(global_model_list)
        for old_param, new_param in zip(global_model.parameters(), average_state_dict.values()):
            old_param.data = new_param
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        # 客户端数量较少，不进行随机选择
        # selected_users = np.random.choice(np.arange(args.num_users), args.number, replace=False)
        acc_s = []
        print('| Global Round : {} | Avg Loss: {:.3f}'.format(round, loss_avg))
        if round % 5 == 0:
            with torch.no_grad():
                for idx in range(args.num_users):
                    print('Test on user {:d}'.format(idx))
                    local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
                    local_model_for_test = copy.deepcopy(local_model_list[idx])
                    local_model_for_test.load_state_dict(local_weights[idx], strict=True)
                    local_model_for_test.eval()
                    acc, loss = local_test.test_inference(idx, args, backbone_list, local_model_for_test)
                    acc_s.append(acc)
                    summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)
                if sum(acc_s) / len(acc_s) > avg_acc:
                    avg_acc = sum(acc_s) / len(acc_s)
                    step = round
                    acc_list = acc_s
                    for idx in range(args.num_users):
                        acc_mtx[idx] = acc_list[idx]
    plot_loss(args,train_loss)
    loss_mtx = torch.zeros([args.num_users])
    acc_s = []
    with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
            local_model_for_test = copy.deepcopy(local_model_list[idx])
            local_model_for_test.load_state_dict(local_weights[idx], strict=True)
            local_model_for_test.eval()
            acc, loss = local_test.test_inference(idx, args, backbone_list, local_model_for_test)
            acc_s.append(acc)
            loss_mtx[idx] = loss
            summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)
    if sum(acc_s) / len(acc_s) > avg_acc:
        avg_acc = sum(acc_s) / len(acc_s)
        step = round
        acc_list = acc_s
        for idx in range(args.num_users):
            acc_mtx[idx] = acc_list[idx]
    print(f"max_acc is {avg_acc}, step is {step}, acc_list is {acc_list}")
    print('For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(torch.mean(acc_mtx), torch.std(acc_mtx)))
    return acc_mtx

def fedrep(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list):
    train_loss, train_accuracy = [], []
    global_model = local_model_list[0]
    local_weights, local_losses = [model.state_dict() for model in local_model_list], [0 for i in range(len(local_model_list))]
    local_weights_all = [model.state_dict() for model in local_model_list]
    selected_users = np.arange(args.num_users)
    avg_acc = 0
    step = 0
    acc_list = [0,0,0,0,0]
    acc_mtx = torch.zeros([args.num_users])
    for round in tqdm(range(args.rounds)):
        print(f'\n | Global Training Round : {round} |\n')
        print(datetime.now())
        # update global weights
        for idx in selected_users:
            local_node = LocalUpdate(args=args, dataset=train_dataset_list[idx],idxs=user_groups[idx])
            w, model_params, loss = local_node.update_weights_fedrep(idx, backbone_list=backbone_list, model=copy.deepcopy(local_model_list[idx]), global_model = global_model, global_round=round)
            local_weights[idx] = w
            local_weights_all[idx] = model_params
            local_losses[idx] = loss
            summary_writer.add_scalar('Train/Loss/user' + str(idx), loss, round)
        # # update global weights
        global_model_list = [local_weights[idx] for idx in selected_users]
        # 创建一个新的空字典来存储平均值
        average_state_dict = OrderedDict()
        # 计算参数的平均值
        for param_name in global_model_list[0].keys():
            param_sum = sum(model[param_name] for model in global_model_list)
            average_state_dict[param_name] = param_sum / len(global_model_list)
        global_model.fc1.load_state_dict(average_state_dict, strict=True)
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        # selected_users = np.random.choice(np.arange(args.num_users), args.number, replace=False)
        acc_s = []
        print('| Global Round : {} | Avg Loss: {:.3f}'.format(round, loss_avg))
        if round % 5 == 0:
            with torch.no_grad():
                for idx in range(args.num_users):
                    print('Test on user {:d}'.format(idx))
                    local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
                    local_model_for_test = copy.deepcopy(local_model_list[idx])
                    local_model_for_test.load_state_dict(local_weights_all[idx], strict=True)
                    local_model_for_test.eval()
                    acc, loss = local_test.test_inference(idx, args, backbone_list, local_model_for_test)
                    acc_s.append(acc)
                    summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)
                if sum(acc_s) / len(acc_s) > avg_acc:
                    avg_acc = sum(acc_s) / len(acc_s)
                    step = round
                    acc_list = acc_s
                    for idx in range(args.num_users):
                        acc_mtx[idx] = acc_list[idx]
                    
    plot_loss(args, train_loss)
    acc_s = []
    
    loss_mtx = torch.zeros([args.num_users])
    with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
            local_model_for_test = copy.deepcopy(local_model_list[idx])
            local_model_for_test.load_state_dict(local_weights_all[idx], strict=True)
            local_model_for_test.eval()
            acc, loss = local_test.test_inference(idx, args, backbone_list, local_model_for_test)
            loss_mtx[idx] = loss
            acc_s.append(acc)
            summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)
        if sum(acc_s) / len(acc_s) > avg_acc:
            avg_acc = sum(acc_s) / len(acc_s)
            step = round
            acc_list = acc_s
            for idx in range(args.num_users):
                acc_mtx[idx] = acc_list[idx]
    print(f"max_acc is {avg_acc}, step is {step}, acc_list is {acc_list}")
    print('For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(torch.mean(acc_mtx), torch.std(acc_mtx)))
    return acc_mtx

def fedproto(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list):
    # C_i
    global_protos = {}
    # C^j
    global_avg_protos = {}
    local_protos = {}
    avg_acc = 0
    step = 0
    train_loss = []
    acc_list = [0,0,0,0,0]
    acc_mtx = torch.zeros([args.num_users])
    for round in tqdm(range(args.rounds)):
        print(f'\n | Global Training Round : {round} |\n')
        local_weights, local_loss1, local_loss2, local_loss_total,  = [], [], [], []
        idxs_users = np.arange(args.num_users)
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset_list[idx], idxs=user_groups[idx])
            w, loss, protos = local_model.update_weights_fedproto(args, idx, local_model_list[idx], global_protos,global_avg_protos,  backbone_list=backbone_list, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)
            if args.add_noise_proto:
                agg_protos = add_noise_proto(args.device, agg_protos, args.scale, args.perturb_coe, args.noise_type)
            # params of com is agg_protos
            local_weights.append(copy.deepcopy(w))
            local_loss1.append(copy.deepcopy(loss['1']))
            local_loss2.append(copy.deepcopy(loss['2']))
            local_loss_total.append(copy.deepcopy(loss['total']))
            local_protos[idx] = copy.deepcopy(agg_protos)

            summary_writer.add_scalar('Train/Loss/user' + str(idx), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx), loss['2'], round)

        for idx in idxs_users:
            local_model_list[idx].load_state_dict(local_weights[idx])

        # update global protos
        global_avg_protos = proto_aggregation(local_protos)
        global_protos = copy.deepcopy(local_protos)
        loss_avg = sum(local_loss_total) / len(local_loss_total)
        print('| Global Round : {} | Avg Loss: {:.3f}'.format(round, loss_avg))
        summary_writer.add_scalar('Train/Loss/avg', loss_avg, round)
        train_loss.append(loss_avg)
        acc_s = []
        if round % 5 == 0:
            with torch.no_grad():
                for idx in range(args.num_users):
                    print('Test on user {:d}'.format(idx))
                    local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
                    local_model_for_test = copy.deepcopy(local_model_list[idx])
                    local_model_for_test.load_state_dict(local_weights[idx], strict=True)
                    local_model_for_test.eval()
                    acc, loss = local_test.test_inference_twoway(idx, args, global_avg_protos, local_protos[idx], backbone_list, local_model_for_test)
                    summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)
                    acc_s.append(acc)
            if sum(acc_s) / len(acc_s) > avg_acc:
                avg_acc = sum(acc_s) / len(acc_s)
                step = round
                acc_list = acc_s
                for idx in range(args.num_users):
                    acc_mtx[idx] = acc_list[idx]
                
    plot_loss(args,train_loss)
    
    loss_mtx = torch.zeros([args.num_users])
    acc_s = []
    with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args = args, dataset = test_dataset_list[idx], idxs = user_groups_test[idx])
            local_model_for_test = copy.deepcopy(local_model_list[idx])
            local_model_for_test.load_state_dict(local_weights[idx], strict=True)
            local_model_for_test.eval()
            acc, loss = local_test.test_inference_twoway(idx, args, global_avg_protos, local_protos[idx], backbone_list, local_model_for_test)
            loss_mtx[idx] = loss
            acc_s.append(acc)
            summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)
        if sum(acc_s) / len(acc_s) > avg_acc:
            avg_acc = sum(acc_s) / len(acc_s)
            step = round
            acc_list = acc_s
            for idx in range(args.num_users):
                acc_mtx[idx] = acc_list[idx]
    print(f"max_acc is {avg_acc}, step is {step}, acc_list is {acc_list}")
    return acc_mtx

def Solo(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list):
    idxs_users = np.arange(args.num_users)
    train_loss, train_accuracy = [], []
    avg_acc = 0
    step = 0
    acc_list = [0,0,0,0,0]
    acc_mtx = torch.zeros([args.num_users])
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round} |\n')
        for idx in idxs_users:
            local_node = LocalUpdate(args=args, dataset=train_dataset_list[idx],idxs=user_groups[idx])
            w, loss = local_node.update_weights_solo(idx, backbone_list=backbone_list, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            summary_writer.add_scalar('Train/Loss/user' + str(idx), loss, round)
        # update global weights
        local_weights_list = copy.deepcopy(local_weights)
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        acc_s = []
        print('| Global Round : {} | Avg Loss: {:.3f}'.format(round, loss_avg))
        if round % 5 == 0:
            with torch.no_grad():
                for i in range(args.num_users):
                    print('Test on user {:d}'.format(i))
                    local_test = LocalTest(args=args, dataset=test_dataset_list[i], idxs=user_groups_test[i])
                    local_model_list[i].eval()
                    acc, loss = local_test.test_inference(i, args, backbone_list, local_model_list[i])
                    acc_s.append(acc)
                    summary_writer.add_scalar('Test/Acc/user' + str(i), acc, round)
            if sum(acc_s) / len(acc_s) > avg_acc:
                step = round
                avg_acc = sum(acc_s) / len(acc_s)
                acc_list = acc_s
                for idx in range(args.num_users):
                    acc_mtx[idx] = acc_s[idx]
    plot_loss(args, train_loss)
    loss_mtx = torch.zeros([args.num_users])
    acc_s = []
    with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
            local_model_list[idx].eval()
            acc, loss = local_test.test_inference(idx, args, backbone_list, local_model_list[idx])
            loss_mtx[idx] = loss
            acc_s.append(acc)
        if sum(acc_s) / len(acc_s) > avg_acc:
            step = round
            avg_acc = sum(acc_s) / len(acc_s)
            acc_list = acc_s
            for idx in range(args.num_users):
                acc_mtx[idx] = acc_s[idx]
    print(f"max_acc is {avg_acc}, step is {step}, acc_list is {acc_list}")
    print('For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(torch.mean(acc_mtx), torch.std(acc_mtx)))
    return acc_mtx

def FedPCL(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list):
    global_protos = {}
    global_avg_protos = {}
    local_protos = {}
    avg_acc = 0
    step = 0
    avg_loss = []
    acc_list = [0,0,0,0,0]
    acc_mtx = torch.zeros([args.num_users])
    # print(local_model_list[0].state_dict().keys())
    for round in tqdm(range(args.rounds)):
        print(f'\n | Global Training Round : {round} |\n')
        local_weights, local_loss1, local_loss2, local_loss_total,  = [], [], [], []
        idxs_users = np.arange(args.num_users)
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset_list[idx], idxs=user_groups[idx])
            # gpus
            # if torch.cuda.device_count() > 1:
            #     w, w_urt, loss, protos = local_model.run_training_pcl(args, idx, global_protos, global_avg_protos, backbone_list=backbone_list, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            # else:
            w, w_urt, loss, protos = local_model.update_weights_pcl(args, idx, global_protos, global_avg_protos, backbone_list=backbone_list, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            # if torch.cuda.device_count() > 1:
            #     agg_protos = protos
            # else:
            agg_protos = agg_func(protos)
            if args.add_noise_proto:
                agg_protos = add_noise_proto(args.device, agg_protos, args.scale, args.perturb_coe, args.noise_type)
            local_weights.append(copy.deepcopy(w))
            local_loss1.append(copy.deepcopy(loss['1']))
            local_loss2.append(copy.deepcopy(loss['2']))
            local_loss_total.append(copy.deepcopy(loss['total']))
            local_protos[idx] = copy.deepcopy(agg_protos)
            summary_writer.add_scalar('Train/Loss/user' + str(idx), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx), loss['2'], round)


        for idx in idxs_users:
            # if torch.cuda.device_count() > 1:
            #    state_dict = {k.replace('module.', ''): v for k, v in local_weights[idx].items()}
            #    local_model_list[idx].load_state_dict(state_dict, strict=True)
            # else:
            local_model_list[idx].load_state_dict(local_weights[idx], strict=True)

        # update global protos
        global_avg_protos = proto_aggregation(local_protos)
        global_protos = copy.deepcopy(local_protos)
        loss_avg = sum(local_loss_total) / len(local_loss_total)
        print('| Global Round : {} | Avg Loss: {:.3f}'.format(round, loss_avg))
        if round > 0:
            avg_loss.append(loss_avg)
        summary_writer.add_scalar('Train/Loss/avg', loss_avg, round)
        acc_s = []
        if round % 5 == 0:
            with torch.no_grad():
                for idx in range(args.num_users):
                    print('Test on user {:d}'.format(idx))
                    local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
                    local_model_for_test = copy.deepcopy(local_model_list[idx])
                    # local_model_for_test.load_state_dict(local_weights[idx], strict=True)
                    local_model_for_test.eval()
                    acc, loss = local_test.test_inference_twoway(idx, args, global_avg_protos, local_protos[idx], backbone_list, local_model_for_test)
                    acc_s.append(acc)
                    summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)
                if sum(acc_s) / len(acc_s) > avg_acc:
                    step = round
                    avg_acc = sum(acc_s) / len(acc_s)
                    acc_list = acc_s
                    for idx in range(args.num_users):
                        acc_mtx[idx] = acc_list[idx]
    plot_loss(args, avg_loss)
    
    loss_mtx = torch.zeros([args.num_users])
    acc_s = []
    with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args = args, dataset = test_dataset_list[idx], idxs = user_groups_test[idx])
            local_model_for_test = copy.deepcopy(local_model_list[idx])
            # local_model_for_test.load_state_dict(local_weights[idx], strict=True)
            local_model_for_test.eval()
            acc, loss = local_test.test_inference_twoway(idx, args, global_avg_protos, local_protos[idx], backbone_list, local_model_for_test)
            loss_mtx[idx] = loss
            acc_s.append(acc)
            summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)
    if sum(acc_s) / len(acc_s) > avg_acc:
        step = round
        avg_acc = sum(acc_s) / len(acc_s)
        acc_list = acc_s
        for idx in range(args.num_users):
            acc_mtx[idx] = acc_list[idx]
    print(f"max_acc is {avg_acc}, step is {step}, acc_list is {acc_list}")
    return acc_mtx

def FedSVRG(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list):
    global_protos = {}
    global_avg_protos = {}
    local_protos = {}
    avg_acc = 0
    step = 0
    acc_list = []
    avg_loss = []
    acc_mtx = torch.zeros([args.num_users])
    for round in tqdm(range(args.rounds)):
        print(f'\n | Global Training Round : {round} |\n')
        local_weights, local_loss1, local_loss2, local_loss_total,  = [], [], [], []
        idxs_users = np.arange(args.num_users)
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset_list[idx], idxs=user_groups[idx])
            w, w_urt, loss, protos = local_model.update_weights_svrg(args, idx, global_protos, global_avg_protos, backbone_list=backbone_list, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)
            if args.add_noise_proto:
                agg_protos = add_noise_proto(args.device, agg_protos, args.scale, args.perturb_coe, args.noise_type)

            local_weights.append(copy.deepcopy(w))
            local_loss1.append(copy.deepcopy(loss['1']))
            local_loss2.append(copy.deepcopy(loss['2']))
            local_loss_total.append(copy.deepcopy(loss['total']))
            local_protos[idx] = copy.deepcopy(agg_protos)

            summary_writer.add_scalar('Train/Loss/user' + str(idx), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx), loss['2'], round)

        for idx in idxs_users:
            local_model_list[idx].load_state_dict(local_weights[idx])
        
        # update global protos
        global_avg_protos = proto_aggregation(local_protos)
        global_protos = copy.deepcopy(local_protos)
        loss_avg = sum(local_loss_total) / len(local_loss_total)
        if round > 0:
            avg_loss.append(loss_avg)
        print('| Global Round : {} | Avg Loss: {:.3f}'.format(round, loss_avg))
        summary_writer.add_scalar('Train/Loss/avg', loss_avg, round)
        acc_s = []
        if round % 5 == 0:
            with torch.no_grad():
                for idx in range(args.num_users):
                    print('Test on user {:d}'.format(idx))
                    local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
                    local_model_for_test = copy.deepcopy(local_model_list[idx])
                    local_model_for_test.load_state_dict(local_weights[idx], strict=True)
                    local_model_for_test.eval()
                    acc, loss = local_test.test_inference_twoway(idx, args, global_avg_protos, local_protos[idx], backbone_list, local_model_for_test)
                    acc_s.append(acc)
                    summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)
                if sum(acc_s) / len(acc_s) > avg_acc:
                    step = round
                    avg_acc = sum(acc_s) / len(acc_s)
                    acc_list = acc_s
                    for idx in range(args.num_users):
                        acc_mtx[idx] = acc_list[idx]
                    
    plot_loss(args, avg_loss)
    
    loss_mtx = torch.zeros([args.num_users])
    acc_s = []
    with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args = args, dataset = test_dataset_list[idx], idxs = user_groups_test[idx])
            local_model_for_test = copy.deepcopy(local_model_list[idx])
            local_model_for_test.load_state_dict(local_weights[idx], strict=True)
            local_model_for_test.eval()
            acc, loss = local_test.test_inference_twoway(idx, args, global_avg_protos, local_protos[idx], backbone_list, local_model_for_test)
            loss_mtx[idx] = loss
            acc_s.append(acc)
            summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)
    if sum(acc_s) / len(acc_s) > avg_acc:
        step = round
        avg_acc = sum(acc_s) / len(acc_s)
        acc_list = acc_s
        for idx in range(args.num_users):
            acc_mtx[idx] = acc_list[idx]
    print(f"max_acc is {avg_acc}, step is {step}, acc_list is {acc_list}")
    return acc_mtx

def FedOurs(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list):
    global_protos = {}
    global_avg_protos = {}
    local_protos = {}
    avg_acc = 0
    step = 0
    acc_list = []
    loss_avgs = []
    acc_mtx = torch.zeros([args.num_users])
    for round in tqdm(range(args.rounds)):
        print(f'\n | Global Training Round : {round} |\n')
        local_weights, local_loss1, local_loss2, local_loss_total,  = [], [], [], []
        idxs_users = np.arange(args.num_users)
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset_list[idx], idxs=user_groups[idx])
            w, w_urt, loss, protos = local_model.update_weights_ours(args, idx, global_protos, global_avg_protos, backbone_list=backbone_list, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)
            if args.add_noise_proto:
                agg_protos = add_noise_proto(args.device, agg_protos, args.scale, args.perturb_coe, args.noise_type)

            local_weights.append(copy.deepcopy(w))
            local_loss1.append(copy.deepcopy(loss['1']))
            local_loss2.append(copy.deepcopy(loss['2']))
            local_loss_total.append(copy.deepcopy(loss['total']))
            local_protos[idx] = copy.deepcopy(agg_protos)

            summary_writer.add_scalar('Train/Loss/user' + str(idx), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx), loss['2'], round)

        for idx in idxs_users:
            local_model_list[idx].load_state_dict(local_weights[idx])

        # update global protos
        global_avg_protos = proto_aggregation(local_protos)
        global_protos = copy.deepcopy(local_protos)
        loss_avg = sum(local_loss_total) / len(local_loss_total)
        if round > 0:
            loss_avgs.append(loss_avg)
        print('| Global Round : {} | Avg Loss: {:.3f}'.format(round, loss_avg))
        summary_writer.add_scalar('Train/Loss/avg', loss_avg, round)
        acc_s = []
        if round % 5 == 0:
            with torch.no_grad():
                for idx in range(args.num_users):
                    print('Test on user {:d}'.format(idx))
                    local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
                    local_model_for_test = copy.deepcopy(local_model_list[idx])
                    local_model_for_test.load_state_dict(local_weights[idx], strict=True)
                    local_model_for_test.eval()
                    acc, loss = local_test.test_inference_twoway(idx, args, global_avg_protos, local_protos[idx], backbone_list, local_model_for_test)
                    acc_s.append(acc)
                    summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)
                if sum(acc_s) / len(acc_s) > avg_acc:
                    step = round
                    avg_acc = sum(acc_s) / len(acc_s)
                    acc_list = acc_s
                    for idx in range(args.num_users):
                        acc_mtx[idx] = acc_list[idx]
    plot_loss(args, loss_avgs)
    
    loss_mtx = torch.zeros([args.num_users])
    acc_s = []
    with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args = args, dataset = test_dataset_list[idx], idxs = user_groups_test[idx])
            local_model_for_test = copy.deepcopy(local_model_list[idx])
            local_model_for_test.load_state_dict(local_weights[idx], strict=True)
            local_model_for_test.eval()
            acc, loss = local_test.test_inference_twoway(idx, args, global_avg_protos, local_protos[idx], backbone_list, local_model_for_test)
            loss_mtx[idx] = loss
            acc_s.append(acc)
            summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)
    if sum(acc_s) / len(acc_s) > avg_acc:
        step = round
        avg_acc = sum(acc_s) / len(acc_s)
        acc_list = acc_s
        for idx in range(args.num_users):
            acc_mtx[idx] = acc_list[idx]
    # plot loss curve
    
    print(f"max_acc is {avg_acc}, step is {step}, acc_list is {acc_list}")
    return acc_mtx

def fed_main(args):
    exp_details(args)
    # set random seed
    args.device = args.device if torch.cuda.is_available() else 'cpu'
    
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print(args)
    # dataset initialization
    # feature non-iid, label non-iid
    if args.feature_iid and args.label_iid==0:
        if args.dataset == 'digit':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_mnistm_noniid(args.num_users, args=args)
        elif args.dataset == 'office':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_caltech_noniid(args.num_users, args=args)
        elif args.dataset == 'domainnet':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_real_noniid(args.num_users, args=args)
    # feature non-iid, label non-iid
    elif args.feature_iid==0 and args.label_iid == 0:
        if args.dataset == 'digit':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_digits_noniid(args.num_users, args=args)
        elif args.dataset == 'office':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_office(args.num_users, args=args)
        elif args.dataset == 'domainnet':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_domainnet(args.num_users, args=args)
   
    # load backbone
    if args.model == 'cnn':
        resnet_quickdraw = resnet18(pretrained=True, ds='quickdraw')
        resnet_birds = resnet18(pretrained=True, ds='birds')
        resnet_aircraft = resnet18(pretrained=True, ds='aircraft')
    # model initialization
    local_model_list = []
    for _ in range(args.num_users):
        if args.num_bb == 1:
            if args.model == 'cnn':
                backbone_list = [resnet_quickdraw]
                local_model = ProjandDeci(512, 256, 10)
        elif args.num_bb == 3:
            if args.model == 'cnn':
                backbone_list = [resnet_quickdraw, resnet_aircraft, resnet_birds]
                local_model = ProjandDeci(512*3, 256, 10)
        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)

    for backbone in backbone_list:
        backbone.to(args.device)
        backbone.eval()


    # load backbone version2
    # if args.model == 'cnn':
    #     mlp_imagenet = mlp_m(pretrained=True)
    #     alexnet_imagenet = alexnet(pretrained=True)
    #     vgg_imagenet = vgg11(pretrained=True)
    
    
    # # model initialization
    # local_model_list = []
    # for _ in range(args.num_users):
    #     if args.num_bb == 1:
    #         if args.model == 'cnn':
    #             backbone_list = [mlp_imagenet]
    #             local_model = ProjandDeci(2048, 256, 10)
    #     elif args.num_bb == 3:
    #         if args.model == 'cnn':
    #             backbone_list = [mlp_imagenet, alexnet_imagenet, vgg_imagenet]
    #             local_model = ProjandDeci(4352, 256, 10)
    #     local_model.to(args.device)
    #     local_model.train()
    #     local_model_list.append(local_model)

    # for backbone in backbone_list:
    #     backbone.to(args.device)
    #     backbone.eval()
    
    # # load backbone version3
    # if args.model == 'cnn':
    #     tiny = vit_tiny_patch16_224(pretrained=sTrue)
    #     small = vit_small_patch16_224(pretrained=True)
    #     base = vit_base_patch16_224(pretrained=True)
    
    
    # # model initialization
    # local_model_list = []
    # for _ in range(args.num_users):
    #     if args.num_bb == 1:
    #         if args.model == 'cnn':
    #             backbone_list = [tiny]
    #             local_model = ProjandDeci(2048, 256, 10)
    #     elif args.num_bb == 3:
    #         if args.model == 'cnn':
    #             backbone_list = [tiny, small, base]
    #             local_model = ProjandDeci(4352, 256, 10)
    #     local_model.to(args.device)
    #     local_model.train()
    #     local_model_list.append(local_model)

    # for backbone in backbone_list:
    #     backbone.to(args.device)
    #     backbone.eval()
    summary_writer = SummaryWriter('./tensorboard/' + args.dataset + '_' + args.alg + '_' + str(len(backbone_list)) + 'bb_' + str(args.rounds) + 'r_' + str(args.num_users) + 'u_'+ str(args.train_ep) + 'ep')
    if args.alg == 'fedpcl':
        acc_mtx = FedPCL(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list)
    elif args.alg == 'fedavg':
        acc_mtx = FedAvg(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list)
    elif args.alg == 'solo':
        acc_mtx = Solo(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list)
    elif args.alg == "pfedme":
        acc_mtx = PFedMe(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list)
    elif args.alg == "perfedavg":
        acc_mtx = perfedavg(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list)
    elif args.alg == "fedrep":
        acc_mtx = fedrep(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list)
    elif args.alg == "fedproto":
        acc_mtx = fedproto(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list)
    elif args.alg == "SVRG":
        acc_mtx = FedSVRG(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list)
    elif args.alg == "ours":
        acc_mtx = FedOurs(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list)
    return acc_mtx

if __name__ == '__main__':
    num_trial = 1
    args = args_parser()
    acc_mtx = torch.zeros([num_trial, args.num_users])
    # 网格搜索最佳超参数 adam
    # acc_max = np.zeros((4, 5))
    # lrs = [0.0001,0.001,0.01,0.1]
    # weight_decay = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    # weight_decay = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    # args.seed = 0
    # for idx1, lr in enumerate(lrs):
    #     for idx2, wd in enumerate(weight_decay):
    #         args.weight_decay = wd
    #         args.lr = lr
    #         for i in range(num_trial):
    #             acc_mtx[i,:] = fed_main(args)
    #             acc_max[idx1, idx2] = float(torch.mean(acc_mtx[i,:]) * 100)
    # # acc_max[0, 0] = 
    # plot_heapmap(args, lrs, weight_decay, acc_max) 
    # plot_heapmap2(args, lrs, weight_decay, acc_max)         
    # print(acc_max)
    # 网格搜索最佳超参数 自定义
    # acc_max = []
    # lrs = [0.1,0.01,0.001,0.0001]
    # weight_decay = None
    # for lr in [0.1,0.01,0.001,0.0001]:
    #     args.lr = lr
    #     args.seed = 0
    #     for i in range(num_trial):
    #         acc_mtx[i,:] = fed_main(args)
    #         acc_max.append(torch.mean(acc_mtx[i,:]) * 100)
    # # plot_heapmap(args, lrs, weight_decay, acc_max) 
    # # plot_heapmap2(args, lrs, weight_decay, acc_max)         
    # print(acc_max)
    # 正式运行
    # args.weight_decay = 0.00001
    # args.lr = 0.1
    for i in range(num_trial):
        args.seed = i
        acc_mtx[i,:] = fed_main(args)  
    # acc_max = []
    # weight_decay = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    # for wd in weight_decay:
    #     args.lr = 0.01
    #     args.weight_decay=wd
    #     args.seed = 0
    #     for i in range(num_trial):
    #         acc_mtx[i,:] = fed_main(args)
    #         acc_max.append(torch.mean(acc_mtx[i,:]) * 100)
    # plot_heapmap(args, lrs, weight_decay, acc_max) 
    # plot_heapmap2(args, lrs, weight_decay, acc_max)         
    # print(acc_max)
    print("The avg test acc of all trials are:")
    for j in range(args.num_users):
        print('{:.2f}'.format(torch.mean(acc_mtx[:,j])*100))

    print("The stdev of test acc of all trials are:")
    for j in range(args.num_users):
        print('{:.2f}'.format(torch.std(acc_mtx[:,j])*100))
    acc_avg = torch.zeros([num_trial])
    for i in range(num_trial):
        acc_avg[i] = torch.mean(acc_mtx[i,:]) * 100
    print("The avg and stdev test acc of all clients in the trials:")
    print('{:.2f}'.format(torch.mean(acc_avg)))
    print('{:.2f}'.format(torch.std(acc_avg)))
