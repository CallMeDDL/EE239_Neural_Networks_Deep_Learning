import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


def flatten(x):
    """
    Input:
    - Tensor of shape (N, D1, ..., DM)

    Output:
    - Tensor of shape (N, D1 * ... * DM)
    """
    x_shape = x.size()
    new_shape = 1
    for i in range(len(x_shape) - 1):
        new_shape *= x_shape[i + 1]
    x_flat = x.reshape((x_shape[0],new_shape))
    return x_flat


############################################################################
#                                Architecture                              #
############################################################################

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(1, 40)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm2d(30),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(5, 22, 1)),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.block3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 50)),
        )
        self.block4 = nn.Sequential(
            torch.nn.Linear(26*19,1),
            nn.Softmax(dim=1),
        )

    def forward(self, x, viz=False):
        N, _, _, _ = x.size()
        x = self.block1(x)
        x = x.unsqueeze(1)
        x = self.block2(x)
        x = x.squeeze()
        x = self.block3(x)
        x = x.view(N,4,-1)
        x = self.block4(x)
        return flatten(x)

############################################################################
#                             Hyper parameters                             #
############################################################################
epochs = 60
batch_size = 64 


seed = 12345
device = 0
path = './results/model/'
log = './results/log/'

cuda = torch.cuda.is_available()

if cuda:
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(seed)


X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
person_train_valid = np.load("person_train_valid.npy")
X_train_valid = np.load("X_train_valid.npy")
y_train_valid = np.load("y_train_valid.npy")
person_test = np.load("person_test.npy")
X_train = X_train_valid.reshape((2115, 1, 25, 1000))[:,:,:22,:]
X_test = X_test.reshape((443, 1, 25, 1000))[:,:,:22,:]

trainset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train_valid - 769).long())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test - 769).long())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

model = LeNet()
if cuda:
    model = model.cuda()

############################################################################
#                                Optimization                              #
############################################################################
lr = 1e-3
weight_decay = 1e-3
betas = (0.9, 0.999)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay = weight_decay)

if not os.path.exists(path):
    os.makedirs(path)
if not os.path.exists(log):
    os.makedirs(log)

train_loss = []
train_acc = []
test_loss = []
test_acc = []


def save_model(state, path):
    torch.save(state, os.path.join(path))


def train(epoch):
    model.train()
    training_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (x, y) in enumerate(trainloader):
        if cuda:
            x = x.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        _, predicted = output.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    train_loss.append(training_loss/len(trainloader))
    train_acc.append(100.*correct/total)
    print('Training Epoch:{}, Training Loss:{:.6f}, Acc:{:6f}'.format(epoch, training_loss/len(trainloader), 100.*correct/total))

def test(epoch):
    model.eval()
    testing_loss = 0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(testloader):
        if cuda:
            x = x.cuda()
            y = y.cuda()
        output = model(x)
        loss = criterion(output, y)
        
        testing_loss += loss.item()
        _, predicted = output.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    test_loss.append(testing_loss/len(testloader))
    test_acc.append(100.*correct/total)
    print('Testing Epoch:{}, Testing Loss:{:.6f}, Acc:{:6f}.'.format(epoch, testing_loss/len(testloader), 100.*correct/total))


def plot_loss(train_loss, test_loss):
    plt.plot(test_loss, label = "test")
    plt.plot(train_loss, label = "train")
    plt.xlabel("eopch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("The loss against epoch")
    plt.show()


def plot_acc(train_acc, test_acc):
    plt.plot(test_acc, label = "test")
    plt.plot(train_acc, label = "train")
    plt.xlabel("eopch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("The accuracy against epoch")
    plt.show()

    
def main():
    epoch = 0
    while epoch < epochs:
        epoch += 1
        train(epoch)
        test(epoch)
    
    save_model({
        'network_params': model.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc
    }, path+'model.pth')

    plot_loss(train_loss, test_loss)
    plot_acc(train_acc, test_acc)


if __name__ == "__main__":
    main()
