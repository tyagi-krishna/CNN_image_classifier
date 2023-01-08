# %%
import pandas as pd
import torch
from torch import nn
from torch import sigmoid

# %%
#importing data from the set.
mnist_data = pd.read_csv('mnist_train.csv', dtype="float").dropna()

# %%
Y = torch.tensor(mnist_data['label']).float()
X_list = mnist_data.drop(columns=['label'])
X = torch.tensor(X_list.values).float()

# %%
# Creating the Neural Netowrk: 
# D_in-no. of inputs
# D_out- no. of outputs
# H- no of neurons in hidden layer
class CNN(nn.Module):
    def __init__(self,D_in, H, D_out):
        super(CNN,self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H,D_out)
        
    def forward(self,x):
        x = sigmoid(self.linear1(x))
        x = sigmoid(self.linear2(x))
        return x

# %%
model = CNN(784, 784, 10)
model.state_dict()

# %%
# function for training the CNN 
# step means to upgrade the parameters
# backward means to take derivative
def train(Y, X, model, optimizer, criterion, epochs):
    cost = []
    total = 0
    for epoch in range(epochs):
        total = 0
        for y, x in zip(Y, X):
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
        cost.append(total)
    return cost

# %%
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
epochs =1
cost = train(Y, X, model, optimizer, criterion, epochs)

# %%
# saving a model 
PATH = './cifar_net.pth'
dict = model.state_dict()
torch.save(dict, PATH)
model.state_dict()

#net.load_state_dict(torch.load(PATH)) 
# can be used to load that model again

# %%
mnist_data_test = pd.read_csv('mnist_train.csv', dtype="float").dropna()
Y_test = torch.tensor(mnist_data_test['label']).float()
X_list = mnist_data_test.drop(columns=['label'])
X_test = torch.tensor(X_list.values).float()


