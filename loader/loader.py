#!/usr/bin/env python

# torch package
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

# torchattacks toolbox
import torchattacks
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent

def attack_loader(args, net, eps):
    if args.attack == "pgd":
        return torchattacks.PGD(model=net, eps=eps,
                                alpha=eps/args.steps*2.3, steps=args.steps, random_start=True)
    elif args.attack == "tpgd":
        return torchattacks.TPGD(model=net, eps=eps, alpha=eps/args.steps*2.3, steps=args.steps)

    elif args.attack == "eot":
        return torchattacks.EOTPGD(model=net, eps=eps,
                    alpha=eps/args.steps, steps=args.steps, sampling=10)

    elif args.attack == "fgsm":
        return torchattacks.FGSM(model=net, eps=eps)

    elif args.attack == "auto_art":
        classifier = PyTorchClassifier(
            model=net,
            clip_values=(0, 1),
            loss=nn.CrossEntropyLoss(),
            input_shape=(args.channel, args.img_size, args.img_size),
            nb_classes=args.n_classes,
        )
        attack = AutoProjectedGradientDescent(estimator=classifier, eps=eps, eps_step=eps/3,
            max_iter=100, batch_size=args.batch_size, loss_type='cross_entropy', verbose=False)

        def f_attack(input, target):
            return torch.from_numpy(attack.generate(x=input.cpu(), y=target.cpu())).cuda()

        return f_attack

def dataset_loader(args):

    args.mean=0.5
    args.std=0.25

    # Setting Dataset Required Parameters
    if args.data == "stl10":
        args.n_classes = 10
        args.img_size  = 96
        args.channel   = 3
    elif args.data == "cifar10":
        args.n_classes = 10
        args.img_size  = 32
        args.channel   = 3
    elif args.data == "cifar100":
        args.n_classes = 100
        args.img_size  = 32
        args.channel   = 3
    elif args.data == "tiny":
        args.n_classes = 200
        args.img_size  = 64
        args.channel   = 3



    if args.data=="stl10":
        transform_train = transforms.Compose(
        [
         transforms.ToTensor()]
        )

        transform_test = transforms.Compose(
            [
            transforms.ToTensor()]
        )

    else:
        transform_train = transforms.Compose(
            [transforms.Pad(4, padding_mode="reflect"),
             transforms.RandomCrop(args.img_size),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor()]
        )

        transform_test = transforms.Compose(
                [transforms.ToTensor()]
            )


    args.batch_size = 100

    # Full Trainloader/Testloader
    traindataset = dataset(args, True,  transform_train)
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True)
    testloader  = torch.utils.data.DataLoader(dataset(args, False, transform_test),  batch_size=args.batch_size, shuffle=True)

    return trainloader, testloader, len(traindataset)


def dataset(args, train, transform):
        if args.data == "cifar10":
            return torchvision.datasets.CIFAR10(root=args.root, transform=transform, download=True, train=train)

        elif args.data == "cifar100":
            return torchvision.datasets.CIFAR100(root=args.root, transform=transform, download=True, train=train)

        elif args.data == "stl10":
            return torchvision.datasets.STL10(root=args.root, transform=transform, download=True,
                                    split="train" if train else "test")
        elif args.data == "tiny":
            return torchvision.datasets.ImageFolder(root=args.root+'/tiny-imagenet-200/train' if train \
                                    else args.root + '/tiny-imagenet-200/val', transform=transform)