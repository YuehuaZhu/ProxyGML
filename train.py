"""
    PyTorch Package for ProxyGML Loss

    Reference
    NeurIPS'2020: "Less is More: A Deep Graph Metric Learning Perspective Using Few Proxies"

    Copyright@xidian.edu.cn

"""
import argparse
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
from PIL import Image
import loss
import evaluation as eva
import net
from tqdm import tqdm
import numpy as np

#############
import auxiliaries as aux
import time

###############

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', default='cars196', type=str, help='Dataset to use.')
parser.add_argument('--data', default="/home/zqs/simulation/zyh_code_demo/nips2020/ProxyGML/", type=str, help='path to dataset')
parser.add_argument('-j', '--workers', default=8, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size')
parser.add_argument('--modellr', default=0.0001, type=float,
                    help='initial model learning rate')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    help='weight decay', dest='weight_decay')
parser.add_argument('--gpu', default=[2], nargs='+', type=int, help='GPU-id for GPU to use.')  #####I added
parser.add_argument('--eps', default=0.01, type=float,
                    help='epsilon for Adam')

parser.add_argument('--dim', default=512, type=int,
                    help='dimensionality of embeddings')
parser.add_argument('--freeze_BN', action='store_true',
                    help='freeze bn')


parser.add_argument('--r', default=20, type=float, help='ration which determins how many proxies are selected')
parser.add_argument('-C', default=98, type=int,
                    help='C classes')
parser.add_argument('--N', default=10, type=int,
                    help='N trainable proxies')
parser.add_argument('--rate', default=0.1, type=float,
                    help='decay rate')
parser.add_argument('--centerlr', default=0.03, type=float,
                    help='initial center learning rate')
parser.add_argument('--warm', default = 1, type = int,
    help = 'Warmup training epochs'
)

parser.add_argument('--weight_lambda', default=0.3, type=float,
                    help='weight_lambda')
parser.add_argument('--new_epoch_to_decay', nargs='+', default=[20, 40], type=int,
                    help='Recall @ Values.')  # [20,40,70]
parser.add_argument('--epoch_to_test', nargs='+', default=[1,2,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50], type=int,
                    help='Recall @ Values.')  # [20,40,70]



def RGB2BGR(im):
    assert im.mode == 'RGB'
    r, g, b = im.split()
    return Image.merge('RGB', (b, g, r))


def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).lstrip('[').rstrip(']')
    print("torch.cuda.current_device()_{}".format(torch.cuda.current_device()))

    # create model
    model = net.bninception(args.dim)
    # torch.cuda.set_device(args.gpu)
    args.device = "cuda"
    model = model.to(args.device) if not len(args.gpu) > 1 else nn.DataParallel(model).to(args.device)

    # load data
    traindir = os.path.join(args.data, args.dataset, 'train')
    testdir = os.path.join(args.data, args.dataset, 'test')
    normalize = transforms.Normalize(mean=[104., 117., 128.],
                                     std=[1., 1., 1.])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Lambda(RGB2BGR),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
            normalize,
        ]))



    ###################
    if args.dataset == 'cars196':
        args.k_vals = [1, 2, 4, 8]
    elif args.dataset == 'cub200':
        args.k_vals = [1, 2, 4, 8]
    elif args.dataset == 'online_products':
        args.k_vals = [1, 10, 100, 1000]

    args.cN = len(train_dataset.class_to_idx)

    metrics_to_log = aux.metrics_to_examine(args.dataset, args.k_vals)
    args.save_path = os.getcwd() + '/Training_Results'

    args.savename = "ProxyGML_{}/".format(args.dataset) + "dim{}_".format(
        args.dim) + "weight_lambda{}_".format(
        args.weight_lambda) + "N{}_".format(args.N) + "r{}_".format(args.r) + "bs{}_".format(
        args.batch_size) + "graph_lr{}_".format(
        args.centerlr) + "epoch_to_decay{}_".format(
        args.new_epoch_to_decay)

    LOG = aux.LOGGER(args, metrics_to_log, name='Base', start_new=True)

    ##########################

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Lambda(RGB2BGR),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)



    # define loss function (criterion) and optimizer

    criterion = loss.ProxyGML(args).to(args.device)


    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.modellr},
                                  {"params": criterion.parameters(), "lr": args.centerlr}],
                                 eps=args.eps, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):

        args.cur_epoch = epoch
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        start = time.time()
        mean_losss = train(train_loader, model, criterion, optimizer, args)
        LOG.log('train', LOG.metrics_to_log['train'], [epoch, np.round(time.time() - start, 4), mean_losss])

        # Warmup: Train only new params, helps stabilize learning.
        if args.warm > 0:
            unfreeze_model_param = list(model.embedding.parameters()) + list(criterion.parameters())

            if epoch == 0:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = False
            if epoch == args.warm:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = True

        # if (epoch+1)>=0 and (epoch+1) %1==0:
        if (epoch + 1) in args.epoch_to_test:
            start = time.time()
            nmi, recall = validate(test_loader, model, args)
            LOG.log('val', LOG.metrics_to_log['val'], [epoch, np.round(time.time() - start), nmi] + list(recall))
            print("\n")
            print(
                'Recall@ {kval}: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; NMI: {nmi:.3f} \n'
                    .format(kval=args.k_vals, recall=recall, nmi=nmi))

    # evaluate on validation set
    # nmi, recall = validate(test_loader, model, args)
    # print('Recall@{kval}: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; NMI: {nmi:.3f} \n'
    #               .format(kval=args.k_vals,recall=recall, nmi=nmi))


def train(train_loader, model, criterion, optimizer, args):
    # switch to train mode
    loss_collect = []
    loss_samples_collect = []
    model.train()
    if args.freeze_BN:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    trainloader = tqdm(train_loader, desc='Epoch {} Training...'.format(args.cur_epoch))

    for i, (input, target) in enumerate(trainloader):
        if args.gpu is not None:
            # input = input.cuda(args.gpu, non_blocking=True) .to(args.device)
            input = input.to(args.device)
        # target = target.cuda(args.gpu, non_blocking=True)
        target = target.to(args.device)

        # compute output
        output = model(input)
        loss, loss_samples = criterion(output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_collect.append(loss.item())
        loss_samples_collect.append(loss_samples.item())
        if i == len(train_loader) - 1:
            trainloader.set_description(
                'Epoch (Train) {0:}: Mean Loss [{1:.4f}]: Mean Samples Loss [{2:.4f}]'.format(args.cur_epoch,
                                                                                              np.mean(loss_collect),
                                                                                              np.mean(
                                                                                                  loss_samples_collect)))
            print("mem={:.3f}MiB, max_mem={:.0f}MiB\n".format(torch.cuda.memory_allocated() / 1e6,
                                                              torch.cuda.max_memory_allocated() / 1e6))

    return np.mean(loss_collect)


def validate(test_loader, model, args):
    # switch to evaluation mode
    model.eval()
    testdata = torch.Tensor()
    testlabel = torch.LongTensor()

    if  args.dataset == "online_products":
        nmi, recall = aux.eval_metrics_one_dataset(model, test_loader, device=args.device, k_vals=args.k_vals, opt=args)
    else:
        with torch.no_grad():
            testloader = tqdm(test_loader, desc='Epoch {} Testing...'.format(args.cur_epoch))
            for i, (input, target) in enumerate(testloader):
                if args.gpu is not None:
                    # input = input.cuda(args.gpu, non_blocking=True)
                    input = input.to(args.device)

                # compute output
                output = model(input)
                testdata = torch.cat((testdata, output.cpu()), 0)
                testlabel = torch.cat((testlabel, target))
            nmi, recall = eva.evaluation(testdata.numpy(), testlabel.numpy(), [1, 2, 4, 8])
    return nmi, recall


def adjust_learning_rate(optimizer, epoch, args):
    # decayed lr by 10 every 20 epochs
    if (epoch + 1) in args.new_epoch_to_decay:
        print("epoch {} change lr".format(epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.rate


if __name__ == '__main__':
    main()
