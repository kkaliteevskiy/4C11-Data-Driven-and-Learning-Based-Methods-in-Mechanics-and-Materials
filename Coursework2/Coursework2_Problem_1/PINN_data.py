import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import scipy
import optuna
import datetime
from utils import *
from tqdm import tqdm
import os

# Define Neural Network
class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if nonlinearity == nn.Tanh:
                    self.layers.append(nn.Tanh())
                elif nonlinearity == nn.ReLU:
                    self.layers.append(nn.ReLU())
                elif nonlinearity == nn.GELU:
                    self.layers.append(nn.GELU())
                else:
                    raise ValueError('Activation function not implemented')
                

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x

### ------------------- CUDA device ---------------- ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

### ------------------- directory for storing info and logs ---------------- ###
dirname = f'trained models/{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}'
os.makedirs(dirname, exist_ok=True)

############################# Data processing #############################
path = 'plate_data.mat'
data = scipy.io.loadmat(path)
torch.set_default_tensor_type(torch.DoubleTensor)
L_boundary = torch.tensor(data['L_boundary'], dtype=torch.float64).to(device)
R_boundary = torch.tensor(data['R_boundary'], dtype=torch.float64).to(device)
T_boundary = torch.tensor(data['T_boundary'], dtype=torch.float64).to(device)
B_boundary = torch.tensor(data['B_boundary'], dtype=torch.float64).to(device)
C_boundary = torch.tensor(data['C_boundary'], dtype=torch.float64).to(device)
Boundary   = torch.tensor(data['Boundary'], dtype=torch.float64, requires_grad=True).to(device)

# truth solution from FEM
disp_truth = torch.tensor(data['disp_data'], dtype=torch.float64).to(device)
# connectivity matrix - this helps you to plot the figure but we do not need it for PINN
t_connect  = torch.tensor(data['t'].astype(float), dtype=torch.float64).to(device)
# all collocation points
x_full = torch.tensor(data['p_full'], dtype=torch.float64,requires_grad=True).to(device)
# collocation points excluding the boundary
x = torch.tensor(data['p'], dtype=torch.float64, requires_grad=True).to(device)
# This chooses 50 fixed points from the truth solution, which we will use for part (e)
rand_index = torch.randint(0, len(x_full), (50,)).to(device)
disp_fix = disp_truth[rand_index,:].to(device)

# We will use two neural networks for the problem:
# NN1: to map the coordinates [x,y] to displacement u
# NN2: to map the coordinates [x,y] to the stresses [sigma_11, sigma_22, sigma_12]
# What we will do later is to first compute strain by differentiate the output of NN1
# And then we compute a augment stress using Hook's law to find an augmented stress sigma_a
# And we will require the output of NN2 to match sigma_a  - we shall do this by adding a term in the loss function
# This will help us to avoid differentiating NN1 twice (why?)
# As it is well known that PINN suffers from higher order derivatives

# Define material properties
E = 10 #N/m^2
mu = 0.3

stiff = E/(1-mu**2)*torch.tensor([[1,mu,0],[mu,1,0],[0,0,(1-mu)/2]]) .to(device)# Hooke's law for plane stress
stiff = stiff.unsqueeze(0).to(device)

# PINN requires super large number of iterations to converge (on the order of 50e^3-100e^3)
#
iterations = 100000 # NOTE: modify for debugging

# Define loss function
loss_func = torch.nn.MSELoss()

# Broadcast stiffness for batch multiplication later
stiff_bc = stiff.to(device)
stiff = torch.broadcast_to(stiff, (len(x),3,3)).to(device)
stiff_bc = torch.broadcast_to(stiff_bc, (len(Boundary),3,3)).to(device)


# Use optuna to optimise hyperparameters:
def objective(trial):

    lrs = trial.suggest_loguniform('lr', 1e-6, 1e-3)
    activation = trial.suggest_categorical('activation', [nn.Tanh, nn.ReLU, nn.GELU])
    n_layers = trial.suggest_int('n_layers', 2, 8)
    n_neurons = trial.suggest_int('n_neurons', 100, 500)

    stress_net = DenseNet([2, *[n_neurons for _ in range(n_layers)], 3], activation).to(device)
    disp_net = DenseNet([2, *[n_neurons for _ in range(n_layers)], 2], activation).to(device)
    params = list(stress_net.parameters()) + list(disp_net.parameters())
    optimizer = torch.optim.Adam(params, lr=lrs)

    loss = 0

    for epoch in range(100): # NOTE: modify for debugging
        optimizer.zero_grad()

        sigma = stress_net(x)
        disp = disp_net(x)

        u = disp[:,0]
        v = disp[:,1]
        dudx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),create_graph=True)[0]
        dvdx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(u),create_graph=True)[0]

        e_11 = dudx[:,0].unsqueeze(1)
        e_22 = dvdx[:,1].unsqueeze(1)
        e_12 = 0.5 * (dvdx[:,0].unsqueeze(1) + dudx[:,1].unsqueeze(1))  
        e = torch.cat((e_11,e_22,e_12), 1)
        e = e.unsqueeze(2)

        sig_aug = torch.bmm(stiff, e).squeeze(2)
        loss_cons = loss_func(sig_aug, sigma)

        disp_bc = disp_net(Boundary)
        sigma_bc = stress_net(Boundary)
        u_bc = disp_bc[:,0]
        v_bc = disp_bc[:,1]

        dudx_bc = torch.autograd.grad(u_bc, Boundary, grad_outputs=torch.ones_like(u_bc),create_graph=True)[0]
        dvdx_bc = torch.autograd.grad(v_bc, Boundary, grad_outputs=torch.ones_like(u_bc),create_graph=True)[0]

        e_11_bc = dudx_bc[:,0].unsqueeze(1)
        e_22_bc = dvdx_bc[:,1].unsqueeze(1)
        e_12_bc = 0.5* (dudx_bc[:,1].unsqueeze(1) + dvdx_bc[:,0].unsqueeze(1))
        e_bc = torch.cat((e_11_bc,e_22_bc,e_12_bc), 1)
        e_bc = e_bc.unsqueeze(2)

        sig_aug_bc = torch.bmm(stiff_bc, e_bc).squeeze(2)
        loss_cons_bc = loss_func(sig_aug_bc, sigma_bc)

        #============= equilibrium ===================#

        sig_11 = sigma[:,0]
        sig_22 = sigma[:,1]
        sig_12 = sigma[:,2]

        dsig11dx = torch.autograd.grad(sig_11, x, grad_outputs=torch.ones_like(sig_11),create_graph=True)[0]
        dsig22dx = torch.autograd.grad(sig_22, x, grad_outputs=torch.ones_like(sig_22),create_graph=True)[0]
        dsig12dx = torch.autograd.grad(sig_12, x, grad_outputs=torch.ones_like(sig_12),create_graph=True)[0]

        eq_x1 = dsig11dx[:,0]+dsig12dx[:,1]
        eq_x2 = dsig22dx[:,1]+dsig12dx[:,0]

        f_x1 = torch.zeros_like(eq_x1)
        f_x2 = torch.zeros_like(eq_x2)

        loss_eq1 = loss_func(eq_x1, f_x1)
        loss_eq2 = loss_func(eq_x2, f_x2)
        #========= boundary ========================#

        tau_R = 0.1 # N/m^2 # torch.zeros_like(R_boundary[:,0]) # NOTE: make sure the dimension is correct
        tau_T = 0 # torch.zeros_like(T_boundary[:,1]) 

        u_L= disp_net(L_boundary)
        u_B = disp_net(B_boundary)

        sig_R = stress_net(R_boundary)
        sig_T = stress_net(T_boundary)
        sig_C = stress_net(C_boundary)

        # Symmetry boundary condition left
        loss_BC_L = loss_func(u_L[:,0], torch.zeros_like(u_L[:,0]))
        # Symmetry boundary condition bottom
        loss_BC_B = loss_func(u_B[:, 1], torch.zeros_like(u_B[:, 1]))
        # Force boundary condition right
        loss_BC_R = loss_func(sig_R[:, 0], tau_R*torch.ones_like(sig_R[:, 0])) \
                    + loss_func(sig_R[:, 2],  torch.zeros_like(sig_R[:, 2]))

        loss_BC_T = + loss_func(sig_T[:, 1], tau_T*torch.ones_like(sig_T[:, 1]))   \
                    + loss_func(sig_T[:, 2],  torch.zeros_like(sig_T[:, 2]))

        # traction free on circle
        loss_BC_C = loss_func(sig_C[:,0]*C_boundary[:,0]+sig_C[:,2]*C_boundary[:,1], torch.zeros_like(sig_C[:, 0]))  \
                    + loss_func(sig_C[:,2]*C_boundary[:,0]+sig_C[:,1]*C_boundary[:,1], torch.zeros_like(sig_C[:, 0]))

    
        # ======= uncomment below for part (e) =======================
        # data_loss_fix
        x_fix = x_full[rand_index, :]
        u_fix = disp_net(x_fix)
        loss_fix = loss_func(u_fix,disp_fix)
        loss = loss_eq1+loss_eq2+loss_cons+loss_BC_L+loss_BC_B+loss_BC_R+loss_BC_T+loss_BC_C+loss_cons_bc + 100*loss_fix
        loss.backward()
        print('loss', loss, 'iter', epoch, end='\r')
        optimizer.step()

    complexity = count_trainable_parameters(stress_net) + count_trainable_parameters(disp_net)
    complexity_penalty = max(0, (complexity - 1e6)/1e6) # punush models with over 1e6 parameters
    return loss.item() + complexity_penalty


# define optuna study
t_start = datetime.datetime.now()
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=25) # NOTE: modify for debugging
t_end = datetime.datetime.now()
print(f"Time taken for hyper-parameter selection: {t_end - t_start}")
print("Best hyperparameters: ", study.best_params)

# Save best parameters to a text file
with open(os.path.join(dirname, 'hyper-parameters.txt'), 'w') as f:
    for key, value in study.best_params.items():
        f.write(f"{key}: {value}\n")
f.close()


# Actual training
# Define the neural network
stress_net = DenseNet([2, *[study.best_params['n_neurons'] for _ in range(study.best_params['n_layers'])], 3], study.best_params['activation']).to(device)
disp_net = DenseNet([2, *[study.best_params['n_neurons'] for _ in range(study.best_params['n_layers'])], 2], study.best_params['activation']).to(device)
optimizer = torch.optim.Adam(list(stress_net.parameters()) + list(disp_net.parameters()), lr=study.best_params['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)


losses = []
t_start = datetime.datetime.now()
for epoch in range(iterations):
    scheduler.step()
    optimizer.zero_grad()

    # To compute stress from stress net
    sigma = stress_net(x)
    # To compute displacement from disp net
    disp = disp_net(x)

    # displacement in x direction
    u = disp[:,0]
    # displacement in y direction
    v = disp[:,1]

    # find the derivatives
    dudx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    dvdx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),create_graph=True)[0]

    # Define strain
    e_11 = dudx[:,0].unsqueeze(1)
    e_22 = dvdx[:,1].unsqueeze(1)
    e_12 = 0.5 * (dvdx[:,0] + dudx[:,1]).unsqueeze(1)  

    e = torch.cat((e_11,e_22,e_12), 1) # i think there should be a 2 here for e_12
    e = e.unsqueeze(2)

    # Define augment stress
    sig_aug = torch.bmm(stiff, e).squeeze(2)

    # Define constitutive loss - forcing the augment stress to be equal to the neural network stress
    loss_cons = loss_func(sig_aug, sigma)

    # find displacement and stress at the boundaries
    disp_bc = disp_net(Boundary)
    sigma_bc = stress_net(Boundary)
    u_bc = disp_bc[:,0]
    v_bc = disp_bc[:,1]

    # Compute the strain and stresses at the boundary
    dudx_bc = torch.autograd.grad(u_bc, Boundary, grad_outputs=torch.ones_like(u_bc),create_graph=True)[0]
    dvdx_bc = torch.autograd.grad(v_bc, Boundary, grad_outputs=torch.ones_like(u_bc),create_graph=True)[0]


    e_11_bc = dudx_bc[:,0].unsqueeze(1)
    e_22_bc = dvdx_bc[:,1].unsqueeze(1)
    e_12_bc = 0.5* (dudx_bc[:,1].unsqueeze(1) + dvdx_bc[:,0].unsqueeze(1))

    e_bc = torch.cat((e_11_bc,e_22_bc,e_12_bc), 1)
    e_bc = e_bc.unsqueeze(2)

    sig_aug_bc = torch.bmm(stiff_bc, e_bc).squeeze(2)

    # force the augment stress to agree with the NN stress at the boundary
    loss_cons_bc = loss_func(sig_aug_bc, sigma_bc)

    #============= equilibrium ===================#

    sig_11 = sigma[:,0]
    sig_22 = sigma[:,1]
    sig_12 = sigma[:,2]

    # stress equilibrium in x and y direction
    dsig11dx = torch.autograd.grad(sig_11, x, grad_outputs=torch.ones_like(sig_11),create_graph=True)[0]
    dsig22dx = torch.autograd.grad(sig_22, x, grad_outputs=torch.ones_like(sig_22),create_graph=True)[0]
    dsig12dx = torch.autograd.grad(sig_12, x, grad_outputs=torch.ones_like(sig_12),create_graph=True)[0]


    eq_x1 = dsig11dx[:,0]+dsig12dx[:,1]
    eq_x2 = eq_x2 = dsig22dx[:,1]+dsig12dx[:,0]

    # zero body forces
    f_x1 = torch.zeros_like(eq_x1)
    f_x2 = torch.zeros_like(eq_x2)

    loss_eq1 = loss_func(eq_x1, f_x1)
    loss_eq2 = loss_func(eq_x2, f_x2)
    #========= boundary ========================#

    # specify the boundary condition
    tau_R = 0.1 # N/m^2 # torch.zeros_like(R_boundary[:,0]) # NOTE: make sure the dimension is correct
    tau_T = 0 # torch.zeros_like(T_boundary[:,1]) 
    #
    u_L= disp_net(L_boundary)
    u_B = disp_net(B_boundary)

    sig_R = stress_net(R_boundary)
    sig_T = stress_net(T_boundary)
    sig_C = stress_net(C_boundary)

    # Symmetry boundary condition left
    loss_BC_L = loss_func(u_L[:,0], torch.zeros_like(u_L[:,0]))
    # Symmetry boundary condition bottom
    loss_BC_B = loss_func(u_B[:, 1], torch.zeros_like(u_B[:, 1]))
    # Force boundary condition right
    loss_BC_R = loss_func(sig_R[:, 0], tau_R*torch.ones_like(sig_R[:, 0])) \
                + loss_func(sig_R[:, 2],  torch.zeros_like(sig_R[:, 2]))

    loss_BC_T = + loss_func(sig_T[:, 1], tau_T*torch.ones_like(sig_T[:, 1]))   \
                + loss_func(sig_T[:, 2],  torch.zeros_like(sig_T[:, 2]))

    # traction free on circle
    loss_BC_C = loss_func(sig_C[:,0]*C_boundary[:,0]+sig_C[:,2]*C_boundary[:,1], torch.zeros_like(sig_C[:, 0]))  \
                + loss_func(sig_C[:,2]*C_boundary[:,0]+sig_C[:,1]*C_boundary[:,1], torch.zeros_like(sig_C[:, 0]))

    # Define loss function:
    loss = loss_eq1+loss_eq2+loss_cons+loss_BC_L+loss_BC_B+loss_BC_R+loss_BC_T+loss_BC_C+loss_cons_bc


    # ======= uncomment below for part (e) =======================
    # data_loss_fix
    x_fix = x_full[rand_index, :]
    u_fix = disp_net(x_fix)
    loss_fix = loss_func(u_fix,disp_fix)
    loss = loss_eq1+loss_eq2+loss_cons+loss_BC_L+loss_BC_B+loss_BC_R+loss_BC_T+loss_BC_C+loss_cons_bc + 100*loss_fix


    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        print('Time elapsed: ', datetime.datetime.now() - t_start, ' loss', loss, 'iter', epoch, end = '\r')


#save the model
torch.save(stress_net.state_dict(), f'{dirname}/stress_net.pth')
torch.save(disp_net.state_dict(), f'{dirname}/disp_net.pth')
np.save(f'{dirname}/losses.npy', losses)


