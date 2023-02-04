import os
import argparse
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from time import time

import model.vgg_model

parser = argparse.ArgumentParser(description='PyTorch Sharpness Training')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run (default: 200)')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 32)')

parser.add_argument('--lr', '--learning-rate', default=0.00002, type=float, metavar='LR', dest='lr',
                    help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')

parser.add_argument('--wd', '--weight-decay', default=0.0004, type=float, metavar='W', dest='weight_decay',
                    help='weight decay (default: 0.0004)')

parser.add_argument('--resume', '-r', default='',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')

# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')

def image_loader(parent_path, class_name):
    root_path = os.path.abspath(parent_path)
    X_data, y_data = [], []
    for (root, dirs, files) in os.walk(root_path):
        for filename in files:
            if 'jpg' not in filename and 'JPG' not in filename and 'jpeg' not in filename:
                continue
            img = cv2.imread(str(os.path.join(root, filename)), cv2.IMREAD_COLOR) / 255
            img = cv2.resize(img, (224, 224))
            X_data.append(img)
            y_data.append(0) if class_name == 'blur' else y_data.append(1)

    return X_data, y_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

output_path = './0928'

def main():
    args = parser.parse_args()

    epoch_time = AverageMeter()

    global output_path

    # set model
    print(f"=> set model")
    # if args.pretrained:
    #     print(f"=> using pre-trained model")
    #     # size error
    #     net = model.vgg_model.vgg16_bn(pretrained=True, num_classes=2)
    # else:
    #     print(f"=> creating model")
    #     net = model.vgg_model.vgg16_bn(pretrained=False, num_classes=2)
    net = model.vgg_model.vgg16_bn(pretrained=False, num_classes=2)

    net = net.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint {args.resume}")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint {args.resume} (epoch {checkpoint['epoch']})")
    else:
        print(f"=> no checkpoint found at {args.resume}")

    # define learning scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # data loading
    print('=> loading data')

    print('=> loading train set')
    X_train_sharp, y_train_sharp = image_loader('./dataset/train_set/sharp', class_name='sharp')
    X_train_blur, y_train_blur = image_loader('./dataset/train_set/blur', class_name='blur')
    X_train = torch.tensor(np.array(X_train_sharp + X_train_blur))
    y_train = torch.tensor(np.array(y_train_sharp + y_train_blur))
    X_train = X_train.permute(0, 3, 1, 2)
    print(f"=> number of train set: {len(X_train)}")

    print('=> loading validation set')
    X_val_sharp, y_val_sharp = image_loader('./dataset/val_set/sharp', class_name='sharp')
    X_val_blur, y_val_blur = image_loader('./dataset/val_set/blur', class_name='blur')
    X_val = torch.tensor(np.array(X_val_sharp + X_val_blur))
    y_val = torch.tensor(np.array(y_val_sharp + y_val_blur))
    X_val = X_val.permute(0, 3, 1, 2)
    print(f"=> number of validation set: {len(X_val)}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=True)

    # # evaluate on validation set
    # if args.evaluate:
    #     validate(val_loader, net, criterion)
    #     return

    # train for epochs
    for epoch in range(args.start_epoch, args.epochs):
        start = time()
        train(train_loader, net, criterion, optimizer, epoch)
        epoch_time.update(time() - start)

        scheduler.step()

        loss, acc = validate(val_loader, net, criterion, epoch)

        # save checkpoint
        if not (epoch + 1) % 5:
            state = {
                'epoch': epoch,
                'state_dict': net.module.state_dict(),
                'acc': acc,
                'optimizer': optimizer.state_dict()
            }

            if epoch < 10:
                epoch = f"00{epoch}"
            elif epoch < 100:
                epoch = f"0{epoch}"

            torch.save(state, f"{output_path}/vgg16_epoch_{epoch}_loss_{loss:.10f}_acc_{acc:.10f}.pth")

    print(f"Time per epoch: {epoch_time.avg}")

def train(train_loader, net, criterion, optimizer, epoch):
    losses = AverageMeter()
    correct = 0
    total = 0

    global output_path

    net.train(True)

    for i, (input, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)
        input, target = input.float().to(device), target.to(device)

        # compute output
        output = net(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        _, pred = torch.max(output.data, 1)
        total += target.size(0)
        correct += (pred == target).sum().item()

        acc = 100.0 * correct / total
        losses.update(loss.item(), input.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # writing to log file
    print(f"Epoch: {epoch}\tLoss: {losses.avg:.10f}\tAccuracy: {acc:.10f}")
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    with open(f"{output_path}/train_log.txt", 'a') as file:
        file.write(f"Epoch: {epoch}\tLoss: {losses.avg:.10f}\tAccuracy: {acc:.10f}\n")

def validate(val_loader, net, criterion, epoch):
    losses = AverageMeter()
    correct = 0
    total = 0

    global output_path
    
    net.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.type(torch.LongTensor)
            input, target = input.float().to(device), target.to(device)

            # compute output
            output = net(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            _, pred = torch.max(output.data, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()

            acc = 100.0 * correct / total
            losses.update(loss.item(), input.size(0))

    # writing to log file
    if not (epoch + 1) % 5:
        print(f"Validation loss: {losses.avg:.10f}\tValidation accuracy: {acc:.10f}")
    with open(f"{output_path}/validation_log.txt", 'a') as file:
        file.write(f"Epoch: {epoch}\tValidation loss: {losses.avg:.10f}\tValidation accuracy: {acc:.10f}\n")

    return losses.avg, acc

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()