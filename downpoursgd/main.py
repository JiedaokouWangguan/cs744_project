import os
import logging 
import argparse
import csv
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

import torch.optim as optim
from optim.downpour_sgd import DownpourSGD
from server import ParameterServer

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



def get_dataset(args, transform, rank):
    """
    :param dataset_name:
    :param transform:
    :param batch_size:
    :return: iterators for the dataset
    """
    if args.dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data%d' % rank, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data%d' % rank, train=False, download=True, transform=transform)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
    return trainloader, testloader

def main(args, rank):

    logs = []

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

    trainloader, testloader = get_dataset(args, transform, rank)
    net = Net()

    if args.no_distributed:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)
    else:
        optimizer = DownpourSGD(net.parameters(), lr=args.lr, n_push=args.num_push, n_pull=args.num_pull, model=net)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, min_lr=1e-3)

    # train
    net.train()
    if args.cuda:
        net = net.cuda()

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        print("Training for epoch {}".format(epoch))
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(predicted, labels)

            log_obj = {
                'timestamp': datetime.now(),
                'iteration': i,
                'training_loss': loss.item(),
                'training_accuracy': accuracy,
            }

#             if i % args.log_interval == 0 and i > 0:    # print every n mini-batches
#                 log_obj['test_loss'], log_obj['test_accuracy']= evaluate( net, testloader, args)
#                 print("Timestamp: {timestamp} | "
#                       "Iteration: {iteration:6} | "
#                       "Loss: {training_loss:6.4f} | "
#                       "Accuracy : {training_accuracy:6.4f} | "
#                       "Test Loss: {test_loss:6.4f} | "
#                       "Test Accuracy: {test_accuracy:6.4f}".format(**log_obj))

            logs.append(log_obj)
                
        val_loss, val_accuracy = evaluate(net, testloader, args, verbose=True)
        scheduler.step(val_loss)

    df = pd.DataFrame(logs)
    print(df)
    if args.no_distributed:
        if args.cuda:
            df.to_csv('log/gpu.csv', index_label='index')
        else:
            df.to_csv('log/single.csv', index_label='index')
    else:
        df.to_csv('log/node{}.csv'.format(dist.get_rank()), index_label='index')

    print('Finished Training')


def evaluate(net, testloader, args, verbose=False):
    if args.dataset == 'MNIST':
        classes = [str(i) for i in range(10)]
    else:
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net.eval()
   
    test_loss = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            if args.cuda:
                images, labels = images.cuda(), labels.cuda()

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            test_loss += F.cross_entropy(outputs, labels).item()

    test_accuracy = accuracy_score(predicted, labels)
    if verbose:
        print('Loss: {:.3f}'.format(test_loss))
        print('Accuracy: {:.3f}'.format(test_accuracy))
        print(classification_report(predicted, labels, target_names=classes))
    
    return test_loss, test_accuracy

def init_server():
    model = Net()
    server = ParameterServer(model=model)
    server.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--num-pull', type=int, default=5, metavar='N', help='how often to pull params (default: 5)')
    parser.add_argument('--num-push', type=int, default=5, metavar='N', help='how often to push grads (default: 5)')
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA for training')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N', help='how often to evaluate and print out')
    parser.add_argument('--no-distributed', action='store_true', default=False, help='whether to use DownpourSGD or normal SGD')
    parser.add_argument('--rank', type=int, metavar='N', help='rank of current process (0 is server, 1+ is training node)')
    parser.add_argument('--world-size', type=int, default=3, metavar='N', help='size of the world')
    parser.add_argument('--server', action='store_true', default=False, help='server node?')
    parser.add_argument('--dataset', type=str, default='MNIST', help='which dataset to train on')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:8088', help='url used to init process group')
    args = parser.parse_args()
    print(args)

    if not args.no_distributed:
        """ Initialize the distributed environment.
        Server and clients must call this as an entry point.
        """
        dist.init_process_group('gloo', rank=args.rank, world_size=args.world_size, init_method=args.dist_url)
        # dist.init_process_group('gloo', rank=args.rank, world_size=args.world_size)
        if args.server:
            init_server()
    main(args, args.rank)
