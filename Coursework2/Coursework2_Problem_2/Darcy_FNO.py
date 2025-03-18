import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import datetime
import h5py
import torch.nn.functional as F
import os
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
        num_examples = x.size()[0] # assume size is (batch_size, x, y)

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0) # grid of steps [0,1]

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        # print('x.shape',x.shape)
        # print('y.shape',y.shape)
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def forward(self, x, y):
        return self.rel(x, y)

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

        self.mean = torch.mean(x, (0, 1, 2))
        self.std = torch.std(x, (0, 1, 2))
        self.eps = eps

    def encode(self, x):
        x = x / (self.std + self.eps)
        return x

    def decode(self, x):
        x = x * (self.std + self.eps)
        return x

# Define network  
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, device = torch.device('cpu')):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2 # must be less than floor(N/2) + 1
        self.device = device

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input.to(weights.device), weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=self.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x.to(self.device)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.act = nn.GELU(approximate='tanh')

    def forward(self, x):
        x = x.to(self.mlp1.weight.device)
        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)
        return x

class FNO(nn.Module):
    def __init__(self, modes1, modes2,  width, device='cpu'):
        super(FNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.device = device

        self.p = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y) - lift to desired channel width
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        # self.act0 = nn.GELU()
        # self.act1 = nn.GELU()
        # self.act2 = nn.GELU()
        # self.act3 = nn.GELU()
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y) - project down to desired dimension

    def forward(self, x):
        grid = self.get_grid(x.shape).to(self.device)
        x = torch.cat((x.unsqueeze(-1), grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2).to(self.device)

        x1 = self.conv0(x) # fft conv invfft layer
        x1 = self.mlp0(x1) # why MLP here? - doesn't seem to do much - dimensions are the same
        x2 = self.w0(x) # regular convolutional layer
        x = x1 + x2
        x = F.gelu(x, approximate='tanh')

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x, approximate='tanh')

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x, approximate='tanh')

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x, approximate='tanh')

        x = self.q(x)
        x = x.squeeze(1)
        return x
    
    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=self.device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=self.device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(self.device)

if __name__ == '__main__':
    ############################# Set up  CUDA device #############################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    ########################### Model directory ###########################
    dirname = f'trained models FNO/{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}'
    os.makedirs(dirname, exist_ok=True)

    ############################# Data processing #############################
    # Read data from mat
    train_path = 'Darcy_2D_data_train.mat'
    test_path = 'Darcy_2D_data_test.mat'

    data_reader = MatRead(train_path)
    a_train = data_reader.get_a().to(device)
    u_train = data_reader.get_u().to(device)

    data_reader = MatRead(test_path)
    a_test = data_reader.get_a().to(device)
    u_test = data_reader.get_u().to(device)

    # Normalize data
    a_normalizer = UnitGaussianNormalizer(a_train)
    a_train = a_normalizer.encode(a_train).to(device)
    a_test = a_normalizer.encode(a_test).to(device)

    u_normalizer = UnitGaussianNormalizer(u_train)
    u_train = u_normalizer.encode(u_train).to(device)
    u_test = u_normalizer.encode(u_test).to(device)

    print(a_train.shape, a_train.device)
    print(a_test.shape, a_test.device)
    print(u_train.shape, u_train.device)
    print(u_test.shape, u_test.device)
    print("normaliser mean and variance", u_normalizer.mean, u_normalizer.std)  

    # pdb.set_trace()

    # Create data loader
    batch_size = 500
    train_set = Data.TensorDataset(a_train, u_train)  
    train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)

    ############################# Define and train network #################################
    loss_func = nn.MSELoss() # LpLoss(reduction='mean')

    #### hyperparameter optimization ####
    def objective(trial):
        modes1 = trial.suggest_int('modes1', 1, 16)
        modes2 = trial.suggest_int('modes2', 1, modes1)
        width = trial.suggest_int('width', 8, 32)
        net = FNO(modes1, modes2, width, device).to(device)
        lrs = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        optimizer = torch.optim.Adam(net.parameters(), lr=lrs)

        loss_train_list = []
        loss_test_list = []
        x = []

        for epoch in range(50):
            net.train(True)
            trainloss = 0
            for i, data in enumerate(train_loader):
                input, target = data
                output = net(input)
                l = loss_func(output, target)

                optimizer.zero_grad()
                l.backward()
                optimizer.step()    

                trainloss += l.item()  
            trainloss = trainloss/len(train_loader)
            print(f'epoch:{epoch}, train loss:{trainloss}', end = '\r')

        return trainloss
        
    study = optuna.create_study(direction='minimize')

    study.optimize(objective, n_trials=50)
    print(study.best_params)

    with open(f'{dirname}/best_hyperparameters.txt', 'w') as f:
        for key, value in study.best_params.items():
            f.write(f'{key}: {value}\n')

    # Train network
    epochs = 2500
    print("Start training FNO for {} epochs...".format(epochs))
    t_start = datetime.datetime.now()

    # create model with optimised hyper parameters
    net =  FNO(13, 10, 21, device).to(device) # FNO(modes1=study.best_params['modes1'], modes2=study.best_params['modes2'], width=study.best_params['width'], device = device).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr= 0.01245690659016353)# study.best_params['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)

    loss_train_list = []
    loss_test_list = []
    x = []

    for epoch in range(epochs):
        net.train(True)
        trainloss = 0
        for i, data in enumerate(train_loader):
            input, target = data
            input = input.to(device)
            output = net(input) # Forwardn
            l = loss_func(output, target) # Calculate loss

            optimizer.zero_grad() # Clear gradients
            l.backward() # Backward
            optimizer.step() # Update parameters
            scheduler.step() # Update learning rate

            trainloss += l.item()    
        trainloss = trainloss/len(train_loader)

        # Test
        net.eval()
        with torch.no_grad():
            test_output = net(a_test)
            testloss = loss_func(test_output, u_test).item()

        # Print train loss every 10 epochs
        # if epoch % 10 == 0:
        print("Elapsed time {}, epoch:{}, train loss:{}, test loss:{}".format(datetime.datetime.now() - t_start, epoch, trainloss/len(train_loader), testloss), end = '\r')

        loss_train_list.append(trainloss)
        loss_test_list.append(testloss)
        x.append(epoch)
    print("")
    # Save model
    torch.save(net.state_dict(), f'{dirname}/model.pth')

    #save training losses
    np.save(f'{dirname}/train_loss.npy', np.array(loss_train_list))
    np.save(f'{dirname}/test_loss.npy', np.array(loss_test_list))




