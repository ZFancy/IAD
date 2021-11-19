from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from tqdm import tqdm
from utils import Logger
from models import *

parser = argparse.ArgumentParser(description='AKD2')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
parser.add_argument('--lr_factor', default=0.1, type=float, help='factor by which to decrease lr')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs for training')
parser.add_argument('--output', default = '', type=str, help='output subdirectory')
parser.add_argument('--model', default = 'ResNet18', type = str, help = 'student model name')
parser.add_argument('--teacher_model', default = 'ResNet18', type = str, help = 'teacher network model')
parser.add_argument('--teacher_path', default = './pre_train/AT_teacher_cifar10/bestpoint.pth.tar', type=str, help='path of AT teacher net being distilled')
parser.add_argument('--teacher_st_path', default = './pre_train/ST_teacher_cifar10/bestpoint.pth.tar', type=str, help='path of ST teacher net being distilled')
parser.add_argument('--temp', default=1.0, type=float, help='temperature for distillation')
parser.add_argument('--val_period', default=1, type=int, help='print every __ epoch')
parser.add_argument('--save_period', default=1, type=int, help='save every __ epoch')
parser.add_argument('--alpha', default=0.5, type=float, help='weight for sum of losses')
parser.add_argument('--dataset', default = 'CIFAR10', type=str, help='name of dataset')
parser.add_argument('--out-dir',type=str,default='./AKD2_CIFAR10',help='dir of output')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--resume',type=str,default=None,help='whether to resume training')
parser.add_argument('--beta',type=float,default=0.0)
parser.add_argument('--alpha1', default=0.5, type=float, help='weight for sum of losses')
parser.add_argument('--alpha2', default=0.25, type=float, help='weight for sum of losses')

args = parser.parse_args()

seed = args.seed
out_dir = args.out_dir
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch in args.lr_schedule:
        lr *= args.lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# Store path
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Save checkpoint
def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

# prepare the dataset
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
if args.dataset == 'CIFAR10':
    trainset = torchvision.datasets.CIFAR10(root='~/data/cifar-10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='~/data/cifar-10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    num_classes = 10
elif args.dataset == 'CIFAR100':
    trainset = torchvision.datasets.CIFAR100(root='~/data/cifar-100', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='~/data/cifar-100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    num_classes = 100



class AttackPGD(nn.Module):
    def __init__(self, basic_net, config):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']

    def forward(self, inputs, targets):
        x = inputs.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(self.basic_net(x), targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0.0, 1.0)
        return self.basic_net(x), x


# build teacher and student models 
# dataparalella

print('==> Building model..'+args.model)
# student
if args.model == 'MobileNetV2':
	basic_net = MobileNetV2(num_classes=num_classes)
elif args.model == 'WideResNet':
	basic_net = WideResNet(num_classes=num_classes)
elif args.model == 'ResNet18':
	basic_net = ResNet18(num_classes=num_classes)
basic_net = basic_net.to(device)
basic_net = torch.nn.DataParallel(basic_net)

# teacher
if args.teacher_path != '':
	if args.teacher_model == 'MobileNetV2':
		teacher_net = MobileNetV2(num_classes=num_classes)
	elif args.teacher_model == 'WideResNet':
		teacher_net = WideResNet(num_classes=num_classes)
	elif args.teacher_model == 'ResNet18':
		teacher_net = ResNet18(num_classes=num_classes)
	teacher_net = teacher_net.to(device)
	for param in teacher_net.parameters():
		param.requires_grad = False

teacher_st_net = WideResNet(num_classes=num_classes)
teacher_st_net = teacher_st_net.to(device)
for param in teacher_st_net.parameters():
    param.requires_grad = False

config_train = {
    'epsilon': 8 / 255,
    'num_steps': 10,
    'step_size': 2 / 255,
}

config_test = {
    'epsilon': 8 / 255,
    'num_steps': 10,
    'step_size': 2 / 255,
}

net = AttackPGD(basic_net, config_train)


if device == 'cuda':
    cudnn.benchmark = True

print('==> Loading at teacher..')
teacher_net = torch.nn.DataParallel(teacher_net)
teacher_net.load_state_dict(torch.load(args.teacher_path)['state_dict'])
teacher_net.eval()

print('==> Loading st teacher..')
teacher_st_net = torch.nn.DataParallel(teacher_st_net)
teacher_st_net.load_state_dict(torch.load(args.teacher_st_path)['state_dict'])
teacher_st_net.eval()


KL_loss = nn.KLDivLoss(reduce=False)
XENT_loss = nn.CrossEntropyLoss()
lr=args.lr

def train(epoch, optimizer, net, basic_net, teacher_net):
    net.train()
    train_loss = 0
    iterator = tqdm(trainloader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(device), targets.to(device)     
        optimizer.zero_grad()
        outputs, pert_inputs = net.forward(inputs, targets)
        teacher_outputs = teacher_net(pert_inputs)
        st_outputs = teacher_st_net(pert_inputs)

        loss = (1-args.alpha1-args.alpha2)*XENT_loss(outputs, targets)+args.alpha1*args.temp*args.temp*(1/len(outputs))*torch.sum(KL_loss(F.log_softmax(outputs/args.temp, dim=1),F.softmax(teacher_outputs/args.temp, dim=1)).sum(dim=1))+args.alpha2*args.temp*args.temp*(1/len(outputs))*torch.sum(KL_loss(F.log_softmax(outputs/args.temp, dim=1),F.softmax(st_outputs/args.temp, dim=1)).sum(dim=1))

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        iterator.set_description(str(loss.item()))

    print('Mean Training Loss:', train_loss/len(iterator))
    return train_loss


def test(epoch, optimizer, net, basic_net, teacher_net):
    net.eval()
    adv_correct = 0
    natural_correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            adv_outputs, pert_inputs = net(inputs, targets)
            natural_outputs = basic_net(inputs)
            _, adv_predicted = adv_outputs.max(1)
            _, natural_predicted = natural_outputs.max(1)
            natural_correct += natural_predicted.eq(targets).sum().item()
            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()
            iterator.set_description(str(adv_predicted.eq(targets).sum().item()/targets.size(0)))
    robust_acc = 100.*adv_correct/total
    natural_acc = 100.*natural_correct/total
    print('Natural acc:', natural_acc)
    print('Robust acc:', robust_acc)
    return natural_acc, robust_acc

def main():
    lr = args.lr
    best_acc = 0
    test_robust = 0

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    logger_test = Logger(os.path.join(out_dir, 'student_results.txt'), title='student')
    logger_test.set_names(['Epoch', 'Natural Test Acc', 'PGD10 Acc'])
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, lr)
        print("teacher >>>> student ")
        train_loss = train(epoch, optimizer, net, basic_net, teacher_net)

        if (epoch+1)%args.val_period == 0:
            natural_val, robust_val = test(epoch, optimizer, net, basic_net, teacher_net)
            logger_test.append([epoch + 1, natural_val, robust_val])

            save_checkpoint({
                        'epoch': epoch + 1,
                        'test_nat_acc': natural_val, 
                        'test_pgd20_acc': robust_val,
                        'state_dict': basic_net.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    })   

            if robust_val > best_acc:
                best_acc = robust_val
                save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': basic_net.state_dict(),
                        'test_nat_acc': natural_val, 
                        'test_pgd20_acc': robust_val,
                        'optimizer' : optimizer.state_dict(),
                    },filename='bestpoint.pth.tar')            
            

if __name__ == '__main__':
    main()
