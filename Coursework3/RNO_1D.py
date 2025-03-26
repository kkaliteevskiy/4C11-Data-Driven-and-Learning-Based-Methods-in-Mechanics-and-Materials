import torch
from torch import tensor
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split, DataLoader
import datetime

import os
import numpy as np
import scipy.io
import h5py
# from tqdm import tqdm

import matplotlib.pyplot as plt

class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

class RNO(nn.Module):
    def __init__(self, size_in = 1, size_out = 1, width = 32, depth = 3, hidden_size = 16): # assume input size is [batch, 1, 1]
        super(RNO, self).__init__()

        # self.h = torch.zeros(batch_size, hidden_size)

        self.g = nn.ModuleList() # d zeta/dt = g(zeta, x)
        self.f = nn.ModuleList() # sigma = f(zeta, x)

        # define f(zeta, x) = sigma
        self.f.append(nn.Linear(size_in + hidden_size, width))
        self.f.append(nn.ReLU())
        for i in range(depth - 1):
            self.f.append(nn.Linear(width, width))
            self.f.append(nn.ReLU())
        self.f.append(nn.Linear(width, size_out))

        # define g(zeta, x) = d zeta/dt 
        self.g.append(nn.Linear(size_in + hidden_size, width))
        self.g.append(nn.ReLU())
        for i in range(depth - 1):
            self.g.append(nn.Linear(width, width))
            self.g.append(nn.ReLU())
        self.g.append(nn.Linear(width, hidden_size))

    def forward(self, x, h, dt):
        # compute g and update hidden state
        h0 = h.detach()
        h_in = torch.cat((x, h), 1) # [batch, channel]
        for _, l in enumerate(self.g):
            h_in = l(h_in)
        
        h = h0 + h_in * dt

        # compute f
        x_in = torch.cat((x, h), 1) # [batch, channel]
        for _, l in enumerate(self.f):
            x_in = l(x_in)

        return x_in, h 

#setup cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# data path
file_path = r'C:\Users\kiril\OneDrive\Documents\University\IIB\4C11 Data-Driven and Learning-Based Methods in Mechanincs and Materials\Coursework3\viscodata_3mat.mat'
data = scipy.io.loadmat(file_path)

sigma = tensor(data['sigma_tol']).unsqueeze(1).float().to(device)
eps = tensor(data['epsi_tol']).unsqueeze(1).float().to(device)

dataset = TensorDataset(eps, sigma)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 280
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size)

# model
hidden_size = 16
net = RNO(size_in=1, size_out=1, width=32, depth=3, hidden_size=hidden_size).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

len_traj = 1001
n_epochs = 5
loss = torch.nn.MSELoss()

training_losses = []
testing_losses = []
t_statrt = datetime.datetime.now()
for epoch in range(n_epochs):
    train_loss = 0.0
    for input, target in train_loader:
        h = torch.zeros(batch_size, hidden_size).to(device)   
        for i in range(len_traj):
            optimizer.zero_grad()
            y, h = net(input[:,:,i], h, 1e-2)
            h = h.detach()
            l = loss(y, target[:,:,i])
            l.backward()
            optimizer.step()
            train_loss += l.item()
    training_losses.append(train_loss)

    test_loss = 0.0
    for input, target in test_loader:
        h = torch.zeros(batch_size, hidden_size).to(device)   
        for i in range(len_traj):
            y, h = net(input[:,:,i], h, 1e-2)
            l = loss(y, target[:,:,i])
            test_loss += l.item()
    testing_losses.append(test_loss)
    t = datetime.datetime.now() - t_statrt
    print(f"Epoch {epoch} training loss: {train_loss}, testing loss: {test_loss}, time: {t}")

t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.mkdir(f"{t}")
torch.save(net.state_dict(), f"{t}/model.pth")
# Save training and testing losses
np.save(f"{t}/training_losses.npy", np.array(training_losses))
np.save(f"{t}/testing_losses.npy", np.array(testing_losses))