from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import shutil
import os
import torch
import numpy as np
import random
import csv
from PIL import ImageFile
import pandas as pd
import matplotlib.pyplot as plt

## copy_images() by Ines Vieira, all else by Romanas Munovas

class config:

    DATASET_PATH = "data/BarkNet"  # specify path to the dataset
    OUTPUT = "output"  # path for saving output
    ## specify the paths to our training and validation set
    TRAIN = "data/train"
    VAL = "data/val"
    # set the input height and width
    CROPSIZE = 224

    # set the batch size and validation data split
    BATCH_SIZE = 8
    VAL_SPLIT = 0.1
    WORKERS = 2  # number of subprocesses in dataLoaders

    EPOCHS = 15
    LEARNING_RATE = 0.01


def copy_images(imagePaths, folder):
    # check if the destination folder exists and if not create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    # loop over the image paths
    for path in imagePaths:
        # grab image name and its label from the path and create
        # a placeholder corresponding to the separate label folder
        imageName = path.split(os.path.sep)[-1]
        label = path.split(os.path.sep)[-2]
        labelFolder = os.path.join(folder, label)

        # check to see if the label folder exists and if not create it
        if not os.path.exists(labelFolder):
            os.makedirs(labelFolder)

        # construct the destination image path and copy the current
        # image to it
        destination = os.path.join(labelFolder, imageName)
        shutil.copy(path, destination)


# Taken from DL in Forestry Jupyter Notebooks provided
def train_one_epoch(model, optimizer, train_loader, epoch, train_losses, train_counter, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * config.BATCH_SIZE) + ((epoch - 1) * len(train_loader.dataset)))


def validation(model, test_loader, test_losses, test_accuracy):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    accuracy = correct / len(test_loader.dataset)
    test_accuracy.append(accuracy)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train(model, optimizer, epochs, train_loader, test_loader, save_path="", model_name=""):
    train_losses = []
    train_counter = []
    test_losses = []
    test_accuracy = []
    validation(model, test_loader, test_losses, test_accuracy)
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, train_loader, epoch, train_losses, train_counter)
        validation(model, test_loader, test_losses, test_accuracy)

    if save_path != "" and model_name != "":
        torch.save(model.state_dict(), save_path + model_name + ".pth")
        torch.save(optimizer.state_dict(), save_path + model_name + "_optimizer.pth")

    return train_losses, test_losses, test_accuracy


def loss_log_to_csv(loss_list, path):
    print("[INFO] Writing to CSV...")
    outfile = open('trainLossRes.csv', 'w')
    out = csv.writer(outfile)
    out.writerows(map(lambda x: [x], loss_list))
    outfile.close()


def val_res_to_csv(acc, mean_loss, path):
    outfile = open('accRes.csv', 'w')
    out = csv.writer(outfile)
    out.writerows(map(lambda x: [x], acc))
    outfile.close()
    outfile = open('valLossRes.csv', 'w')
    out = csv.writer(outfile)
    out.writerows(map(lambda x: [x], mean_loss))
    outfile.close()


def pretrained_net():
    model = torchvision.models.resnet34(pretrained=True)
    model = freeze_layers(1, model)
    # last layer adjusted to match output classes
    model.fc = nn.Linear(512, 21)
    return model


# freeze first layer
def freeze_layers(n, net):
    first_params = [net.conv1.parameters(), net.bn1.parameters()]
    layers = [net.layer1.parameters(), net.layer2.parameters(),
              net.layer3.parameters(), net.layer4.parameters()]
    if n >= 1:
        for params in first_params:
            for param in params:
                param.requires_grad = False

    for i in range(n - 1):
        layer = layers[i]
        for param in layer:
            param.requires_grad = False

    return net


def load_images():
    print("[INFO] Loading Image Paths...")
    imagePaths = []
    tempList = os.listdir(config.DATASET_PATH)
    tempList.remove('ERB')
    tempList.remove('PEG')
    tempList.remove('PID')
    for image_path1 in tempList:
        if not image_path1.startswith('.'):
            image_path2 = os.path.join(config.DATASET_PATH, image_path1)
            stop = 0  # initialize the counter
            tempList2 = os.listdir(image_path2)
            random.shuffle(tempList2)
            for image_path3 in tempList2:
                stop += 1
                # undersampling
                if stop < 597:
                    image_path3 = os.path.join(image_path2, image_path3)
                    imagePaths.append(image_path3)
                else:
                    break

    np.random.shuffle(imagePaths)
    valPathsLen = int(len(imagePaths) * config.VAL_SPLIT)
    trainPathsLen = len(imagePaths) - valPathsLen
    trainPaths = imagePaths[:trainPathsLen]
    valPaths = imagePaths[trainPathsLen:]
    print("[INFO] Copying Images...")
    copy_images(trainPaths, config.TRAIN)
    copy_images(valPaths, config.VAL)

def plot_results():
    accuracy = pd.read_csv (r'accRes.csv')
    accuracy = accuracy.apply(lambda x: x * 100)
    mean_loss = pd.read_csv(r'valLossRes.csv')

    mean_loss = mean_loss.values.tolist()
    accuracy = accuracy.values.tolist()
    epochs = range(16)
    fig, host = plt.subplots(figsize=(8, 5))
    par1 = host.twinx()
    plt.title("ResNet34 Validation Accuracy + Mean Loss per epoch")

    host.set_xlim(0, 15)
    host.set_ylim(0, 100)
    par1.set_ylim(0, 4)

    host.set_xlabel("Epochs")
    host.set_ylabel("Accuracy (%)")
    par1.set_ylabel("Mean Loss")

    color1 = plt.cm.viridis(1)
    color2 = plt.cm.viridis(.5)

    p1, = host.plot(epochs, accuracy, color=color1, label="Accuracy (%)")
    p2, = par1.plot(epochs, mean_loss, color=color2, label="Mean Loss")

    lns = [p1, p2]
    host.legend(handles=lns, loc='lower right')
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    fig.tight_layout()
    plt.show()
    plt.savefig("resnet.png")

if __name__ == '__main__':
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    load_images()  # Make BarkNet Subset using undersampling

    # IMAGE TRANSFORMATION
    hFlip = transforms.RandomHorizontalFlip(p=0.25)
    vFlip = transforms.RandomVerticalFlip(p=0.25)
    rotate = transforms.RandomRotation(degrees=15)
    crop = transforms.RandomCrop(config.CROPSIZE)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    # initialize our training and validation set data augmentation
    trainTransforms = transforms.Compose([crop, hFlip, vFlip, rotate, transforms.ToTensor(), normalize])
    valTransforms = transforms.Compose([crop, transforms.ToTensor(), normalize])

    # initialize the training and validation dataset
    trainDataset = ImageFolder(root=config.TRAIN, transform=trainTransforms)
    valDataset = ImageFolder(root=config.VAL, transform=valTransforms)

    # TRAIN AND VALIDATION SET LOADERS
    trainDataLoader = DataLoader(trainDataset, batch_size=config.BATCH_SIZE, shuffle=True)
    valDataLoader = DataLoader(valDataset, batch_size=config.BATCH_SIZE)

    res_net = pretrained_net()
    optimizer = torch.optim.Adam(res_net.parameters(), lr=config.LEARNING_RATE, weight_decay=0.)
    train_loss, test_loss, test_accuracy = train(res_net, optimizer, config.EPOCHS, trainDataLoader, valDataLoader,
                                                 save_path="./save", model_name="resnet")
    loss_log_to_csv(train_loss, "./trainloss.csv")
    val_res_to_csv(test_accuracy, test_loss, "./valuationRes.csv")
    plot_results()
