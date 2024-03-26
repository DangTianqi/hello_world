import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform, download=True)

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
#
# print(test_set[0])

writer = SummaryWriter('log')
for i in range(10):
    img, target = train_set[i]
    writer.add_image('img', img, i)

writer.close()