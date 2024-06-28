import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv, GCNConv, GATConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn.pool import global_add_pool, global_max_pool, SAGPooling
from torch_geometric.loader import DataLoader
from dataset_utils import *
from torch.distributions import Normal
from torch.distributions import constraints

import pyro
import param
from pyro.distributions import Normal
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoNormal

target = ['lut', 'ff', 'dsp', 'bram', 'uram', 'srl', 'cp', 'power']
tar_idx = 0
jknFlag = 0

class BayesianNet(PyroModule):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = PyroModule[torch.nn.Linear](in_features, hidden_features)
        self.fc1.weight = PyroSample(prior=Normal(0., 1.).expand_by([hidden_features, in_features]).to_event(2))
        self.fc1.bias = PyroSample(prior=Normal(0., 1.).expand_by([hidden_features]).to_event(1))
        
        self.fc2 = PyroModule[torch.nn.Linear](hidden_features, out_features)
        self.fc2.weight = PyroSample(prior=Normal(0., 1.).expand_by([out_features, hidden_features]).to_event(2))
        self.fc2.bias = PyroSample(prior=Normal(0., 1.).expand_by([out_features]).to_event(1))
    
    def forward(self, x, y=None):
        x = x.float()   #Make sure the input x is of the same data type as the model parameters
        x = F.relu(self.fc1(x))
        mean = self.fc2(x)
        sigma = torch.ones_like(mean)  # Make sure sigma has the same shape as mean
    
        '''
        if y is not None:
            if y.shape[0] != mean.shape[0]:
                raise ValueError("The batch size of 'y' does not match 'mean'.")
            if y.ndim == 1:
                y = y.unsqueeze(1)  '''
        

        with pyro.plate("data", size=x.shape[0]):  # Use the correct batch size
            obs = pyro.sample("obs", Normal(mean, sigma), obs=y)
        return mean

def train_svi(model, guide, optimizer, train_loader):
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    model.train()
    num_epochs = 10 
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            x, y = data.x.to(device).float(), data.y[tar_idx].to(device).float()    #Ensure data types are consistent before using them
            #print("x shape:", x.shape, "y shape:", y.shape)
            
            #if x.shape[0] != y.shape[0]:
                #print("Mismatch in batch sizes", x.shape, y.shape)
                #continue  # Skip this batch or do other processing

            loss = svi.step(x, y)
            total_loss += loss
        print(f'Epoch {epoch} : loss = {total_loss / len(train_loader.dataset)}')

'''def train_svi(model, guide, optimizer, train_loader):
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    model.train()
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            x, y = data.x.to(device).float(), data.y.to(device).float()  
            if x.shape[0] != y.shape[0]:
                print(f"Mismatch in batch sizes: {x.shape} vs {y.shape}")
                continue  
        
            y = y[:, tar_idx] if y.ndim > 1 else y  

            loss = svi.step(x, y)
            total_loss += loss
        average_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}: loss = {average_loss}')'''

def test_svi(model, guide, test_loader):
    svi = SVI(model, guide, Adam({"lr": 0.001}), loss=Trace_ELBO())
    model.eval()
    predictive = Predictive(model, guide=guide, num_samples=100, return_sites=("obs", "_RETURN"))
    means = []
    for data in test_loader:
        x = data.x.to(device)
        samples = predictive(x)
        means.append(samples["_RETURN"].mean(dim=0))
    return torch.cat(means, dim=0)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_dir = os.path.abspath('../dataset/std')
    model_dir = os.path.abspath('./model')

    dataset = os.listdir(dataset_dir)
    dataset_list = generate_dataset(dataset_dir, dataset, print_info=False)
    train_ds, test_ds = split_dataset(dataset_list, shuffle=True, seed=128)

    print('train_ds size = {}, test_ds size = {}'.format(len(train_ds), len(test_ds)))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=True, drop_last=True)

    data_ini = next(iter(train_loader))

    if data_ini.x is None or data_ini.x.size(1) is None:
        raise ValueError("data_ini.x is not correctly formatted.")

    in_features = data_ini.x.size(1) 
    out_features = 1  
    hidden_features = 50

    model = BayesianNet(in_features, out_features, hidden_features).to(device)
    guide = AutoNormal(model)
    optimizer = Adam({"lr": 0.01})

    for epoch in range(10):
        train_loss = train_svi(model, guide, optimizer, train_loader)
        predicted_means = test_svi(model, guide, test_loader)
        print(f'Epoch: {epoch}, Predictions: {predicted_means.mean()}')
