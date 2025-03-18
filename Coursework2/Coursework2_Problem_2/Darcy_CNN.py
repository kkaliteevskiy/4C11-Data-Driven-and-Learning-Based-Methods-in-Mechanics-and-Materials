# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from time import time
import datetime
import h5py
from utils import *
import os
import datetime
import optuna
import pdb


# Define Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0] # assume size is (batch_size, x, y) nonono, input size is (batch_size, channels = 1, x, y)

        # Assume uniform mesh
        h = 1.0 / (x.size()[2] - 1.0) # grid of steps [0,1]

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0] # input shape is (batch_size, channels = 1, x, y)

        print('x.shape',x.shape)
        print('y.shape',y.shape)
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def forward(self, x, y):
        return self.abs(x, y)

    def __call__(self, x, y):
        return self.forward(x, y)

# Define data reader
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead).__init__()

        self.file_path = file_path
        self.data = h5py.File(self.file_path)

    def get_a(self):
        a_field = np.array(self.data['a_field']).T
        return torch.tensor(a_field, dtype=torch.float32)

    def get_u(self):
        u_field = np.array(self.data['u_field']).T
        return torch.tensor(u_field, dtype=torch.float32)
    
# Define normalizer, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = torch.mean(x, (0,2,3))#.to(device)
        self.std = torch.std(x, (0,2,3))#.to(device)
        self.eps = eps# .to(device)

    def encode(self, x):
        x = x / (self.std + self.eps)
        return x

    def decode(self, x):
        x = x * (self.std + self.eps)
        return x

# Define network  
class CNN(nn.Module):
    def __init__(self, n_layers = 3, n_channels = 64, kernel_size = 3, activation = nn.ReLU):
        super(CNN, self).__init__()
        
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Conv2d(1, n_channels, kernel_size, padding='same'))
        self.layers.append(activation())

        # Middle layers
        for _ in range(n_layers - 2):
            self.layers.append(nn.Conv2d(n_channels, n_channels, kernel_size, padding='same'))
            self.layers.append(activation())

        # Last layer
        self.layers.append(nn.Conv2d(n_channels, 1, 1, 1, padding='same'))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == '__main__':
    ########################## CUDA SETUP ##########################
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print('CUDA is available!  Training on GPU ...')
        device = torch.device('cuda:0')
    print('Using device:', device)

    ########################### Model directory ###########################
    dirname = f'trained models/{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}'
    os.makedirs(dirname, exist_ok=True)
    
    ############################# Data processing #############################
    # Read data from mat
    train_path = 'Darcy_2D_data_train.mat'
    test_path = 'Darcy_2D_data_test.mat'

    # pdb.set_trace()
    data_reader = MatRead(train_path)
    a_train = data_reader.get_a().unsqueeze(1).to(device)
    u_train = data_reader.get_u().unsqueeze(1).to(device)

    data_reader = MatRead(test_path)
    a_test = data_reader.get_a().unsqueeze(1).to(device)
    u_test = data_reader.get_u().unsqueeze(1).to(device)

    # Normalize data
    a_normalizer = UnitGaussianNormalizer(a_train)
    a_train = a_normalizer.encode(a_train).to(device)
    a_test = a_normalizer.encode(a_test).to(device)

    u_normalizer = UnitGaussianNormalizer(u_train)
    u_train = u_normalizer.encode(u_train).to(device)
    u_test = u_normalizer.encode(u_test).to(device)

    print(a_train.shape)
    print(a_test.shape)
    print(u_train.shape)
    print(u_test.shape)

    # Create data loader
    batch_size = 64
    train_set = Data.TensorDataset(a_train, u_train)
    train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)

    loss_func = torch.nn.MSELoss()

    ############# optuna hyperparameter optimization ############
    def objective(trial):        
        depth = trial.suggest_int('depth', 1, 8)
        width = trial.suggest_int('width', 1, 64)
        kernel_size = trial.suggest_int('kernel_size', 1, 6)
        activation = trial.suggest_categorical('activation', [nn.ReLU, nn.GELU])
        net = CNN(n_layers=depth, n_channels=width, kernel_size=kernel_size, activation=activation).to(device)
        lrs = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        optimizer = torch.optim.Adam(net.parameters(), lr=lrs)

        trainloss = 0
        for epoch in range(20):
            net.train(True)
            trainloss = 0
            loss_arr = []
            for i, data in enumerate(train_loader):
                input, target = data
                output = net(input) # Forwardn
                output = u_normalizer.decode(output).to(device)
                l = loss_func(output, target) # Calculate loss

                optimizer.zero_grad() # Clear gradients
                l.backward() # Backward
                optimizer.step() # Update parameters
                # scheduler.step() # Update learning rate
                loss_arr.append(l.item())

            trainloss = np.mean(np.array(loss_arr))
            print(f'epoch:{epoch}, train loss:{trainloss}', end = '\r')

        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        complexity_penalty = max(0, (5e5-n_params)/5e4)
        trainloss += complexity_penalty
        return trainloss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=25)
    print(study.best_params)
    # Save the best hyperparameters
    with open(f'{dirname}/best_hyperparameters.txt', 'w') as f:
        for key, value in study.best_params.items():
            f.write(f'{key}: {value}\n')
    print(f'Best hyperparameters saved to {dirname}/best_hyperparameters.txt')

    
    ######## actual training ########
    net = CNN(n_layers=study.best_params['depth'], n_channels=study.best_params['width'],\
               kernel_size=study.best_params['kernel_size'], activation=study.best_params['activation']).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=study.best_params['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)

    epochs = 10000 # Number of epochs
    print("Start training CNN for {} epochs...".format(epochs))
    t_start = datetime.datetime.now()
    
    loss_train_list = []
    loss_test_list = []
    x = []
    for epoch in range(epochs):
        net.train(True)
        trainloss = 0
        for i, data in enumerate(train_loader):
            input, target = data
            output = net(input) # Forwardn
            output = u_normalizer.decode(output).to(device)
            l = loss_func(output, target) # Calculate loss

            optimizer.zero_grad() # Clear gradients
            l.backward() # Backward
            optimizer.step() # Update parameters
            scheduler.step() # Update learning rate

            trainloss += l.item()    

        # Test
        net.eval()

        with torch.no_grad():
            test_output = net(a_test)
            test_output = u_normalizer.decode(test_output)
            testloss = loss_func(test_output, u_test).item()

        # Print train loss every 10 epochs
        if epoch % 10 == 0:
            print("Elapsed time {}, epoch:{}, train loss:{}, test loss:{}".format(datetime.datetime.now() -t_start, epoch, trainloss/len(train_loader), testloss), end = '\r')

        loss_train_list.append(trainloss/len(train_loader))
        loss_test_list.append(testloss)
        x.append(epoch)


    # Save the model
    torch.save(net.state_dict(), f'{dirname}/model.pth')
    print(f'Model saved to {dirname}/model.pth')    
    np.save(f'{dirname}/train_losses.npy', loss_train_list)
    np.save(f'{dirname}/test_losses.npy', loss_test_list)
