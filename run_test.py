import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from time import time

import model.vgg_model

parser = argparse.ArgumentParser(description='PyTorch Sharpness Testing')

parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 32)')

parser.add_argument('--resume', '-r',
                    default='',
                    type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')

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

neg, pos = [], []
targets = []

def main(hist=False):
    args = parser.parse_args()

    global output_path, neg, pos, targets

    # set and load model
    print(f"=> set model")
    net = model.vgg_model.vgg16_bn(pretrained=False, num_classes=2)

    net = net.to(device)

    # load model
    if os.path.isfile(args.resume):
        print(f"=> loading checkpoint {args.resume}")
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['state_dict'])
        print(f"=> loaded checkpoint {args.resume}")
    else:
        print(f"=> no checkpoint found at {args.resume}")
        return

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # data loading
    print('=> loading test set')
    X_test_sharp, y_test_sharp = image_loader('./dataset/test_set/sharp', class_name='sharp')
    X_test_blur, y_test_blur = image_loader('./dataset/test_set/blur', class_name='blur')
    X_test = torch.tensor(np.array(X_test_sharp + X_test_blur))
    y_test = torch.tensor(np.array(y_test_sharp + y_test_blur))
    X_test = X_test.permute(0, 3, 1, 2)
    print(f"=> number of test set: {len(X_test)}")

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=True)

    # test
    test(test_loader, net)

    # record confusion matrix with threshold
    thd_record = [0.5, 0.9, 0.99]
    thd_lst = [0.99 + 0.0001 * i for i in range(1, 101)]

    for thd in thd_record + thd_lst:
        tn, fp, fn, tp = 0, 0, 0, 0

        for i in range(len(targets)):
            # actual blur
            if targets[i] <= 0:
                if pos[i] <= thd: tn += 1
                else: fp += 1
            # actual sharp
            else:
                if pos[i] <= thd: fn += 1
                else: tp += 1

        if thd in thd_record or fp == 0:
            box_out = open(f"{output_path}/vgg16_confusion_matrix_with_{thd}.txt", mode='w')
            print(f"Threshold: {thd}\n", file=box_out)
            print(f"True Negative: {tn}\nFalse Positive: {fp}\nFalse Negative: {fn}\nTrue Positive: {tp}\n", file=box_out)
            print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.6f}", file=box_out)
            try:
                print(f"Precision: {tp / (tp + fp):.6f}", file=box_out)
            except ZeroDivisionError:
                print(f"Precision: Infinity", file=box_out)
            else:
                break

    # draw histogram
    if hist:
        blur, sharp = [], []
        for i in range(len(targets)):
            # actual blur
            if targets[i] <= 0:
                blur.append(pos[i].item())
            # actual sharp
            else:
                sharp.append(pos[i].item())

        plt.figure()
        plt.hist(blur, label='Blur', bins=100, histtype='step')
        plt.hist(sharp, label='Sharp', bins=100, histtype='step')
        plt.xlabel('Sharp probability')
        plt.ylabel('Count')
        plt.legend()
        plt.title('VGG16')
        plt.savefig(f"{output_path}/vgg16_histogram.jpg")

def test(test_loader, net):
    global neg, pos, targets

    inference_time = AverageMeter()

    net.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            start = time()

            target = target.type(torch.LongTensor)
            input, target = input.float().to(device), target.to(device)

            # compute output
            output = net(input)

            inference_time.update(time() - start)

            # negative, positive, target list
            neg.extend(list(output[:, 0]))
            pos.extend(list(output[:, 1]))
            targets.extend(list(target))

    print(f"Inference time: {inference_time.avg}")
            
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
    main(hist=True)