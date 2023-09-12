import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Network(nn.Module):
    def __init__(self, lr, n_actions):
        super(Network, self).__init__()
        self.lr = lr
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(1,8,kernel_size=3, padding="same")
        self.max1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8,8,kernel_size=3,padding="same")
        self.max2 = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1152,256)
        self.fc2 = nn.Linear(256,124)
        self.pi = nn.Linear(124,n_actions)
        self.v = nn.Linear(124,1)

        self.optim = optim.RMSprop(self.parameters(), lr=self.lr)

        self.device = "cuda:0" if T.cuda.is_available() else 'cpu'
        self.to(self.device)



    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = self.max1(x)
        x = F.relu(self.conv2(x))
        x = self.max2(x)
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        pi = self.pi(x)
        v  = self.v(x)

        return (pi,v)
