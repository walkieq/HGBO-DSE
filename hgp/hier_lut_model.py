import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Initialize input channel dimensions
in_channels = 10  
hidden_channels = 32  
num_classes = 3

# Sample features for four nodes
x = torch.randn((4, 10), dtype=torch.float)

# Edge index for constructing graph connectivity
edge_index = torch.tensor([[0, 1, 2, 3, 0, 2], [1, 0, 3, 2, 2, 0]], dtype=torch.long)

# Labels for training
y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

# Construct a graph data object
data = Data(x=x, edge_index=edge_index, y=y)

class BayesianGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(BayesianGNN, self).__init__()
        # Graph convolutional layer
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, out_channels)
        
        # Define priors for Bayesian learning
        self.fc1w_prior = Normal(0., 1.).expand([hidden_channels, hidden_channels]).to_event(2)
        self.fc1b_prior = Normal(0., 1.).expand([hidden_channels]).to_event(1)
        self.outw_prior = Normal(0., 1.).expand([out_channels, hidden_channels]).to_event(2)
        self.outb_prior = Normal(0., 1.).expand([out_channels]).to_event(1)
    
    def forward(self, x, edge_index):
        # Forward pass through layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        return F.log_softmax(x, dim=1)

def model(x_data, edge_index, y_data):
    #print("x_data shape:", x_data.shape)
    # Bayesian inference model
    with pyro.plate("data", size=x_data.size(0)):
        lifted_module = pyro.random_module(
            'module', 
            net, 
            {
                'fc1.weight': net.fc1w_prior, 
                'fc1.bias': net.fc1b_prior,
                'out.weight': net.outw_prior, 
                'out.bias': net.outb_prior
            }
        )
        lifted_reg_model = lifted_module()
        lhat = lifted_reg_model(x_data, edge_index)
        pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

def guide(x_data, edge_index, y_data):
    #print("x_data shape in guide:", x_data.shape)  # 调试形状
    with pyro.plate("data", x_data.size(0)):
        fc1w_mu = torch.randn_like(net.fc1.weight)
        fc1w_sigma = torch.randn_like(net.fc1.weight)
        fc1b_mu = torch.randn_like(net.fc1.bias)
        fc1b_sigma = torch.randn_like(net.fc1.bias)
        outw_mu = torch.randn_like(net.out.weight)
        outw_sigma = torch.randn_like(net.out.weight)
        outb_mu = torch.randn_like(net.out.bias)
        outb_sigma = torch.randn_like(net.out.bias)
        # Register learnable params in the param store
        fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
        fc1w_sigma_param = pyro.param("fc1w_sigma", torch.nn.Softplus()(fc1w_sigma))
        fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
        fc1b_sigma_param = pyro.param("fc1b_sigma", torch.nn.Softplus()(fc1b_sigma))
        outw_mu_param = pyro.param("outw_mu", outw_mu)
        outw_sigma_param = pyro.param("outw_sigma", torch.nn.Softplus()(outw_sigma))
        outb_mu_param = pyro.param("outb_mu", outb_mu)
        outb_sigma_param = pyro.param("outb_sigma", torch.nn.Softplus()(outb_sigma))
        # Use Normal distribution as variational distribution
        fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
        fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
        outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param)
        outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
        priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
                  'out.weight': outw_prior, 'out.bias': outb_prior}
        lifted_module = pyro.random_module("module", net, priors)
        return lifted_module()

# Initialize the neural network and the optimization model
net = BayesianGNN(in_channels, hidden_channels, num_classes)
optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    loss = svi.step(data.x, data.edge_index, data.y)
    print(f'Epoch {epoch}: Loss {loss}')
