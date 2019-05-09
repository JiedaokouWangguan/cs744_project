import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import datasets, transforms
from parameter_server import ParameterServer
from aeasgd import AEASGD
from utils import MessageCode


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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    # yuanfang added
    parser.add_argument('--quantize-nbits', default=0, type=int,
                        help='quantize')
    parser.add_argument('--tau', default=32, type=int,
                        help='hyperparameter used in AEASGD')
    parser.add_argument('--rho', default=0.01, type=float,
                        help='hyperparameter used in AEASGD')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='does not work', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                        help='distributed backend')
    ps_flag_parser = parser.add_mutually_exclusive_group(required=False)
    ps_flag_parser.add_argument('--flag', dest='ps_flag', action='store_true')
    ps_flag_parser.add_argument('--no-flag', dest='ps_flag', action='store_false')
    parser.set_defaults(ps_flag=False)

    args = parser.parse_args()
    use_cuda = False

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    if args.ps_flag:
        print("before init process group")
        # start a parameter server
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        print("after init process group")
        quantize_num_bits = 8 if args.quantize_nbits == 0 else args.quantize_nbits
        ps = ParameterServer(model, args.world_size, quantize_num_bits=quantize_num_bits)
        print("starting parameter server....")
        ps.start()
    else:
        print("before init process group")
        # start a worker
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

        print("after init process group")
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data%d'%(args.rank), train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data%d'%(args.rank), train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        quantize_num_bits = 8 if args.quantize_nbits == 0 else args.quantize_nbits
        optimizer = AEASGD(model.parameters(), lr=args.lr, tau=args.tau, rho=args.rho, model=model, quantize_num_bits=quantize_num_bits)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)

        optimizer.send_message(MessageCode.WorkerTerminate, torch.randn(optimizer.squash_model(optimizer.model).numel()))

        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
