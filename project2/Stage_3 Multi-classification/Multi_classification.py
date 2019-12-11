import os
import copy
import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from Multi_Network import *


ROOT_DIR = '../Dataset/'
TRAIN_DIR = 'train/'
VAL_DIR = 'val/'
TRAIN_ANNO = 'Multi_train_annotation.csv'
VAL_ANNO = 'Multi_val_annotation.csv'
CLASSES = ['Mammals', 'Birds']  # [0, 1]
SPECIES = ['rabbits', 'rats', 'chickens']  # [0, 1, 2]


class MyDataSet(Dataset):

    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform

        if not os.path.isfile(self.annotations_file):
            print(self.annotations_file + 'does not exist!')
        self.file_info = pd.read_csv(annotations_file, index_col=0)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.file_info['path'][idx]
        if not os.path.isfile(image_path):
            print(image_path + '  does not exist!')
            return None
        image = Image.open(image_path).convert('RGB')
        label_class = int(self.file_info.iloc[idx]['classes'])
        label_species = int(self.file_info.iloc[idx]['species'])

        sample = {'image': image, 'classes': label_class, 'species': label_species}  # 数据格式

        if self.transform:
            sample['image'] = self.transform(image)

        return sample


train_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       ])
val_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                     transforms.ToTensor()
                                     ])


train_dataset = MyDataSet(root_dir=ROOT_DIR + TRAIN_DIR, annotations_file=TRAIN_ANNO, transform=train_transforms)
test_dataset = MyDataSet(root_dir=ROOT_DIR + VAL_DIR, annotations_file=VAL_ANNO, transform=val_transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset)
data_loaders = {'train': train_loader, 'val': test_loader}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def visualize_dataset():
    print(len(train_dataset))
    idx = random.randint(0, len(train_dataset))
    sample = train_loader.dataset[idx]
    print(idx, sample['image'].shape, CLASSES[sample['classes']], SPECIES[sample['species']])
    img = sample['image']
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()
    plt.close('all')


visualize_dataset()


def train_model(model, criterion, optimizer, num_epochs=50):  # scheduler,
    loss_list = {'train': [], 'val': []}
    accuracy_list_classes_and_species = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-*' * 25)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            corrects = 0
            print()
            for idx, data in enumerate(data_loaders[phase]):
                # print(phase+' processing: {}th batch.'.format(idx))
                inputs = data['image'].to(device)
                labels_classes = data['classes'].to(device)
                labels_species = data['species'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    x_classes, x_species = model(inputs)

                    x_classes = x_classes.view(-1, 2)
                    _, predicted_classes = torch.max(x_classes, 1)

                    x_species = x_species.view(-1, 3)
                    _, predicted_species = torch.max(x_species, 1)

                    loss_c = criterion(x_classes, labels_classes)
                    loss_s = criterion(x_species, labels_species)
                    loss = loss_c / 2 + loss_s / 2

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                print('xianshi: ', running_loss)
                print('size is: ', inputs.size(0))
                buffer = abs(predicted_classes - labels_classes) + abs(predicted_species - labels_species)
                corrects += torch.sum(buffer == 0)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            loss_list[phase].append(epoch_loss)

            epoch_acc_classes_and_species = corrects.double() / len(data_loaders[phase].dataset)
            epoch_acc = epoch_acc_classes_and_species
            accuracy_list_classes_and_species[phase].append(100 * epoch_acc_classes_and_species)
            print('{} Loss: {:.4f}  Accuracy: {:.2%}'.format(phase, epoch_loss, epoch_acc_classes_and_species))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc_classes_and_species
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val classes_and_species Acc: {:.2%}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model_classes_and_species.pt')
    print('Best val classes_and_species Acc: {:.2%}'.format(best_acc))
    return model, loss_list, accuracy_list_classes_and_species


epochs = 100
network = Net().to(device)
optimizer_i = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
criterion_i = nn.CrossEntropyLoss()
exp_lr_scheduler_i = lr_scheduler.StepLR(optimizer_i, step_size=1, gamma=0.1)
# Decay LR by a factor of 0.1 every 1 epochs
# exp_lr_scheduler,
model_cs, loss_list_cs, accuracy_list_cs = train_model(model=network, criterion=criterion_i,
                                                       optimizer=optimizer_i, num_epochs=epochs)


# Figure plotting##############################################

draw_key = 'open'  # 控制画图的开关
if draw_key == 'open':
    x = range(0, epochs)
    y1 = loss_list_cs["val"]
    y2 = loss_list_cs["train"]
    plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
    plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
    plt.legend()
    plt.title('train and val loss vs. epochs')
    plt.ylabel('loss')
    plt.savefig("train_and_val_loss_vs_epochs.jpg")
    plt.close('all')
    y5 = accuracy_list_cs["train"]
    y6 = accuracy_list_cs["val"]
    plt.plot(x, y5, color="r", linestyle="-", marker=".", linewidth=1, label="train")
    plt.plot(x, y6, color="b", linestyle="-", marker=".", linewidth=1, label="val")
    plt.legend()
    plt.title('train and val Classes & Species  vs. epochs')
    plt.ylabel('Classes&Species accuracy')
    plt.savefig("train_and_val_Classes_and_Species_acc_vs_epochs.jpg")
    plt.close('all')
elif draw_key == 'close':
    print('There are no figures!')


# Visualization ###############################################

def visualize_model(model):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loaders['val']):
            inputs = data['image']
            labels_classes = data['classes'].to(device)
            labels_species = data['species'].to(device)

            x_classes, x_species = model(inputs.to(device))

            x_classes = x_classes.view(-1, 2)
            _, predicted_classes = torch.max(x_classes, 1)

            x_species = x_species.view(-1, 3)
            _, predicted_species = torch.max(x_species, 1)

            plt.imshow(transforms.ToPILImage()(inputs.squeeze(0)))
            plt.savefig('save/{}_val.jpg'.format(i))
            plt.title('prediction_classes: {}, prediction_species: {}\n gt_classes:{}, gt_species: {}'.format(
                CLASSES[predicted_classes.item()], SPECIES[predicted_species.item()],
                CLASSES[labels_classes], SPECIES[labels_species]))


visualize_model(model_cs)
