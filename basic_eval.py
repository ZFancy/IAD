import argparse
import torch.nn as nn
import torchvision
from torchvision import transforms
from models import *
import attack_generator as attack

parser = argparse.ArgumentParser(description='PyTorch White-box Adversarial Attack Test')
parser.add_argument('--net', type=str, default="resnet18", help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn")
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10,help='WRN width factor')
parser.add_argument('--drop_rate', type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--model_path', default="./bestpoint.pth.tar", help='model for white-box attack evaluation')

args = parser.parse_args()

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
print('==> Load Test Data')
if args.dataset == "cifar10":
    testset = torchvision.datasets.CIFAR10(root='~/data/cifar-10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    num_classes = 10
if args.dataset == 'cifar100':
    testset = torchvision.datasets.CIFAR100(root='~/data/cifar-100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    num_classes = 100

print('==> Load Model')
if args.net == "resnet18":
    model = ResNet18(num_classes=num_classes).cuda()
    net = "resnet18"
if args.net == "WRN":
    model = WideResNet(depth=args.depth, num_classes=num_classes, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(args.depth,args.width_factor,args.drop_rate)
model = torch.nn.DataParallel(model)

print(net)
print(args.model_path)
model.load_state_dict(torch.load(args.model_path)['state_dict'])

print('==> Evaluating Performance under White-box Adversarial Attack')

loss, test_nat_acc = attack.eval_clean(model, test_loader)
print('Natural Test Accuracy: {:.2f}%'.format(100. * test_nat_acc))
# Evalutions the same as DAT.
loss, fgsm_wori_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=8/255, step_size=8/255,loss_fn="cent", category="Madry",random=True)
print('FGSM without Random Start Test Accuracy: {:.2f}%'.format(100. * fgsm_wori_acc))
loss, pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=8/255, step_size=2/255,loss_fn="cent", category="Madry", random=True)
print('PGD20 Test Accuracy: {:.2f}%'.format(100. * pgd20_acc))
loss, cw_wori_acc = attack.eval_robust(model,test_loader, perturb_steps=30, epsilon=8/255, step_size=2/255,loss_fn="cw",category="Madry",random=True)
print('CW Test Accuracy: {:.2f}%'.format(100. * cw_wori_acc))
