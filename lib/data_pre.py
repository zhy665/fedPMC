import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_noniid_lt
from sampling import femnist_iid, femnist_noniid, femnist_noniid_unequal, femnist_noniid_lt
import numpy as np
import data_utils
from data_utils import TwoCropTransform
import seaborn as sns
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt


# feature & label noniid
def prepare_data_digits_noniid(num_users, args):
    # Prepare digit (feature noniid, label iid)
    if args.model == 'cnn':
        transform_mnist = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_svhn = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_usps = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_synth = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_mnistm = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif args.model == 'vit':
        transform_mnist = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_svhn = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_usps = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_synth = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_mnistm = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    data_root = args.data_dir

    # MNIST
    mnist_trainset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/MNIST', channels=1, train=True, transform=TwoCropTransform(transform_mnist))
    mnist_testset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/MNIST', channels=1, train=False, transform=transform_mnist)

    # SVHN
    svhn_trainset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/SVHN', channels=3, train=True, transform=TwoCropTransform(transform_svhn))
    svhn_testset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/SVHN', channels=3, train=False, transform=transform_svhn)

    # USPS
    usps_trainset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/USPS', channels=1, train=True, transform=TwoCropTransform(transform_usps))
    usps_testset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/USPS', channels=1, train=False, transform=transform_usps)

    # Synth Digits
    synth_trainset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/SynthDigits/', channels=3, train=True, transform=TwoCropTransform(transform_synth))
    synth_testset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/SynthDigits/', channels=3, train=False, transform=transform_synth)

    # MNIST-M
    mnistm_trainset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/MNIST_M/', channels=3, train=True, transform=TwoCropTransform(transform_mnistm))
    mnistm_testset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/MNIST_M/', channels=3, train=False, transform=transform_mnistm)

    train_dataset_list = [mnist_trainset, svhn_trainset, usps_trainset, synth_trainset, mnistm_trainset]
    test_dataset_list = [mnist_testset, svhn_testset, usps_testset, synth_testset, mnistm_testset]

    # generate train idx
    idx_batch_train = [[] for _ in range(num_users)]
    user_groups = {}
    K = args.num_classes
    df = np.zeros([num_users, K])
    for k in range(K):
        proportions = np.random.dirichlet(np.repeat(args.alpha, num_users))
        proportions = proportions / proportions.sum()
        proportions = ((proportions) * (num_users*10)).astype(int)
        for i in range(num_users):
            y_train = train_dataset_list[i].labels
            idx_k = np.where(y_train == k)[0]
            idx_batch_train[i].extend(idx_k[0:proportions[i]].tolist())

        j = 0
        for idx_j in idx_batch_train:
            if k != 0:
                df[j, k] = int(len(idx_j))
            else:
                df[j, k] = int(len(idx_j))
            j += 1

    for i in range(num_users):
        user_groups[i] = idx_batch_train[i]

    # generate test idx
    user_groups_test = {}
    idx_batch_test = [[] for _ in range(num_users)]
    for i in range(num_users):
        y_test = test_dataset_list[i].labels
        for k in range(K):
            idx_k = np.where(y_test == k)[0]
            # idx_batch_test[i].extend(idx_k[start : end].tolist())
            idx_batch_test[i].extend(idx_k[0:100].tolist())
        user_groups_test[i] = idx_batch_test[i]

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test

# label noniid
def prepare_data_mnistm_noniid(num_users, args):
    data_root = args.data_dir

    if args.model == 'cnn':
        transform_mnistm = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Synth Digits
        mnistm_trainset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/MNIST_M/', channels=3, train=False, transform=TwoCropTransform(transform_mnistm))
        mnistm_testset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/MNIST_M/', channels=3, train=False, transform=transform_mnistm)

    elif args.model == 'vit':
        transform_mnistm_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        transform_mnistm_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # Synth Digits
        mnistm_trainset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/MNIST_M/', channels=3, train=False, transform=TwoCropTransform(transform_mnistm_train))
        mnistm_testset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/MNIST_M/', channels=3, train=False, transform=transform_mnistm_test)

    train_dataset_list = []
    test_dataset_list = []
    for _ in range(num_users):
        train_dataset_list.append(mnistm_trainset)
        test_dataset_list.append(mnistm_testset)

    # generate train idx and test idx
    K = args.num_classes
    idx_batch = [[] for _ in range(num_users)]
    y = mnistm_trainset.labels
    N = y.shape[0]
    df = np.zeros([num_users, K])
    for k in range(K):
        idx_k = np.where(y == k)[0]
        if num_users ==5 or num_users == 10:
            idx_k = idx_k[0:110*num_users]
        elif num_users == 20:
            idx_k = idx_k[0:55 * num_users]
        elif num_users == 40:
            idx_k = idx_k[0:30 * num_users]
        elif num_users == 80:
            idx_k = idx_k[0:15 * num_users]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(args.alpha, num_users))
        proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

        j = 0
        for idx_j in idx_batch:
            if k != 0:
                df[j, k] = int(len(idx_j))
            else:
                df[j, k] = int(len(idx_j))
            j += 1

    user_groups = {}
    user_groups_test = {}
    for i in range(num_users):
        np.random.shuffle(idx_batch[i])
        num_samples = len(idx_batch[i])
        if num_users == 5 or num_users == 10:
            train_len = int(num_samples/11)
        elif num_users == 20:
            train_len = int(num_samples / 5.5)
        elif num_users == 40:
            train_len = int(num_samples / 3)
        elif num_users == 80:
            train_len = int(num_samples / 1.5)
        user_groups[i] = idx_batch[i][:train_len]
        user_groups_test[i] = idx_batch[i][train_len:]

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test

# feature noniid
def prepare_data_digits(num_users, args):
    if args.model == 'cnn':
        transform_mnist = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_svhn = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_usps = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_synth = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_mnistm = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif args.model == 'vit':
        transform_mnist = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_svhn = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_usps = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_synth = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_mnistm = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    data_root = './data/'

    # MNIST
    mnist_trainset = data_utils.DigitsDataset(args=args, data_path=data_root + "digit/MNIST", channels=1, train=True, transform=TwoCropTransform(transform_mnist))
    mnist_testset = data_utils.DigitsDataset(args=args, data_path=data_root + "digit/MNIST", channels=1, train=False, transform=transform_mnist)

    # SVHN
    svhn_trainset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/SVHN', channels=3, train=True, transform=TwoCropTransform(transform_svhn))
    svhn_testset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/SVHN', channels=3, train=False, transform=transform_svhn)

    # Synth Digits
    synth_trainset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/SynthDigits/', channels=3, train=True, transform=TwoCropTransform(transform_synth))
    synth_testset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/SynthDigits/', channels=3, train=False, transform=transform_synth)

    # USPS
    usps_trainset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/USPS', channels=1, train=True, transform=TwoCropTransform(transform_usps))
    usps_testset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/USPS', channels=1, train=False, transform=transform_usps)

    # MNIST-M
    mnistm_trainset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/MNIST_M/', channels=3, train=True, transform=TwoCropTransform(transform_mnistm))
    mnistm_testset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/MNIST_M/', channels=3, train=False, transform=transform_mnistm)

    train_dataset_list = [mnist_trainset, svhn_trainset, usps_trainset, synth_trainset, mnistm_trainset]
    test_dataset_list = [mnist_testset, svhn_testset, usps_testset, synth_testset, mnistm_testset]

    K = args.num_classes
    idx_batch_train = [[] for _ in range(num_users)]
    user_groups = {}
    for i in range(num_users):
        y_train = train_dataset_list[i].labels
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            idx_batch_train[i].extend(idx_k[0:10].tolist())
        user_groups[i] = idx_batch_train[i]

    # test idx
    user_groups_test = {}
    idx_batch_test = [[] for _ in range(num_users)]
    for i in range(num_users):
        y_test = test_dataset_list[i].labels
        for k in range(K):
            idx_k = np.where(y_test == k)[0]
            # idx_batch_test[i].extend(idx_k[start : end].tolist())
            idx_batch_test[i].extend(idx_k[0:200].tolist())
        user_groups_test[i] = idx_batch_test[i]
    return train_dataset_list, test_dataset_list, user_groups, user_groups_test