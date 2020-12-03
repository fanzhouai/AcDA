import torch.nn as nn
import torch.nn.functional as F
import torch

class Net_fea(nn.Module):
    """
    Feature extractor network

    """

    def __init__(self):
        super(Net_fea, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)

        return  x

class Net_clf(nn.Module):
    """
    Classifier network, also give the latent space and embedding feature
    (we use the embedding feature in the KNN prediction)
    """

    def __init__(self):
        super(Net_clf,self).__init__()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self,x):

        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)

        return x, e1

    def get_embedding_dim(self):

        return 50


class Net_dis(nn.Module):

    """
    Discriminator network, output with [0,1] (sigmoid function)

    """
    def __init__(self):
        super(Net_dis,self).__init__()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self,x):
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x
