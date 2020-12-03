import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms, utils
from mnist_m import MNISTM

# Here for the loaders we implement two kinds of dataset and loaders 
# The first is for single loaders we can use it as a iter(loader) method
# The second is for source_target loader method, such method requires us to take care of the length of each dataset

class single_dataset():
    '''
    Define a datahandler to get both source and target dataset
    '''

    def __init__(self, data_name='SVHN', train=True, download=False):

        #self.source_name = source_name
        self.data_name = data_name

        self.train = train
        self.download = download

        if self.train:
            self.split = 'train'
        else:
            self.split = 'test'
        
    def return_dataset(self):
        
        if self.data_name == 'SVHN':
            X,Y = self.get_SVHN()
        elif self.data_name == 'MNIST':
            X,Y = self.get_MNIST()
        elif self.data_name == 'USPS':
            X,Y = self.get_USPS()
        elif self.data_name == 'MNIST_M':
            X, Y = self.get_MNIST_M()
        else:
            print('Invailaid data name, please check')
        return X,Y


    def get_MNIST(self):
        raw = datasets.MNIST('./mnist/MNIST/processed/', train=self.train, download=self.download)

        if self.train:
            X = raw.train_data
            Y = raw.train_labels
        else:
            X = raw.test_data
            Y = raw.test_labels

        print('MNIST loaded')
        return X,Y

    def get_SVHN(self):

        data = datasets.SVHN('./SVHN', split=self.split, download=self.download)

        if self.train:
            X = torch.from_numpy(data.data)
            Y = torch.from_numpy(data.labels)
        else:
            X = torch.from_numpy(data.data)
            Y = torch.from_numpy(data.labels)

        print('SVHN loaded')

        return X, Y

    def get_USPS(self):

        data = datasets.USPS('USPS',train=self.train,download=self.download)

        if self.train:
            X = torch.from_numpy(data.data)
            Y = torch.from_numpy(np.array(data.targets))
        else:
            X = torch.from_numpy(data.data)
            Y = torch.from_numpy(np.array(data.targets))
        print('USPS loaded')

        return X,Y

    def get_MNIST_M(self):

        data = MNISTM('./mnist_m/processed', train=self.train, download=self.download)

        if self.train:
            X = data.train_data
            Y = data.train_labels
        else:
            X = data.test_data
            Y = data.test_labels


        print('Mnist_M loaded')
        print(X.dtype)
        print(X.size())

        return X,Y

class single_handler(Dataset):

    def __init__(self, X, Y, data_name, transform=None, return_id = False):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.data_name = data_name
        self.return_id = return_id

    def __getitem__(self, index):

        x, y = self.X[index], self.Y[index]

        if self.transform is not None:
            if self.data_name == 'MNIST':
                x = Image.fromarray(x.numpy(), mode='L')
                x = self.transform(x)
            elif self.data_name == 'SVHN':
                x = Image.fromarray(np.transpose(x.numpy(), (1, 2, 0)))
                x = self.transform(x)
            elif self.data_name == 'USPS':
                #x = Image.fromarray(np.transpose(x.numpy(), (1, 2, 0)))
                x = Image.fromarray(x.numpy(), mode='L')
                x = self.transform(x)
            elif self.data_name == 'MNIST_M':
                x = Image.fromarray(x.numpy())
                x = self.transform(x)
        if self.return_id:
            return  x, y, index
        else:
            return x, y

    def __len__(self):
        return len(self.Y)

class source_target_dataset():

    '''
    Define a datahandler to get both source and target dataset
    '''

    def __init__(self, source_name='SVHN', target_name='MNIST', train=True, download=True):

        self.source_name = source_name
        self.target_name = target_name

        self.train = train
        self.download = download

        if self.train:
            self.split = 'train'
        else:
            self.split = 'test'


    def get_MNIST(self):
        raw = datasets.MNIST('mnist', train=self.train, download=self.download)

        if self.train:
            X = raw.train_data
            Y = raw.train_labels
        else:
            X = raw.test_data
            Y = raw.test_labels

        print('MNIST loaded')
        return X,Y

    def get_SVHN(self):

        data = datasets.SVHN('SVHN', split=self.split, download=self.download)

        if self.train:
            X = torch.from_numpy(data.data)
            Y = torch.from_numpy(data.labels)
        else:
            X = torch.from_numpy(data.data)
            Y = torch.from_numpy(data.labels)

        print('SVHN loaded')

        return X, Y

    def get_USPS(self):

        data = datasets.USPS('USPS',train=self.train,download=self.download)

        if self.train:
            X = torch.from_numpy(data.data)
            Y = torch.from_numpy(np.array(data.targets))
        else:
            X = torch.from_numpy(data.data)
            Y = torch.from_numpy(np.array(data.targets))
        print('USPS loaded')

        return X,Y

    def get_MNIST_M(self):

        data = MNISTM('mnist_m', train=self.train, download=self.download)

        if self.train:
            X = data.train_data
            Y = data.train_labels
        else:
            X = data.test_data
            Y = data.test_labels


        print('Mnist_M loaded')
        print(X.dtype)
        print(X.size())

        return X,Y


    def get_dataset(self):

        if self.source_name == 'MNIST':
            X_s, Y_s = self.get_MNIST()
            print('Source Dataset: MNIST')

            if self.target_name == 'USPS':
                X_t, Y_t = self.get_USPS()
                print('Target Dataset: USPS')

            elif self.target_name == 'MNIST_M':
                X_t, Y_t = self.get_MNIST_M()
                print('Target Dataset: MNIST_M')

            else: print(' Target name is not defined ')

        elif self.source_name == 'USPS':
            X_s, Y_s = self.get_USPS()
            print('Source Dataset: USPS')

            if self.target_name == 'MNIST':
                X_t, Y_t = self.get_MNIST()
                print('Target Dataset: MNIST')

            else:
                print(' Target name is not defined ')

        elif self.source_name == 'SVHN':

            X_s, Y_s = self.get_SVHN()
            print('Source Dataset SVHN')
            if self.target_name == 'MNIST':
                X_t, Y_t = self.get_MNIST()
                print('Target Dataset: MNIST')
            else:
                print(' Target name is not defined')

        elif self.source_name == 'MNIST_M':
            X_s, Y_s = self.get_MNIST_M()
            print('Source Dataset: MNIST_M')

            if self.target_name == 'MNIST':
                X_t, Y_t = self.get_MNIST()
                print('Target Dataset: MNIST')

            else:
                print(' Target name is not defined ')



        else:
            print(' Source name is not defined')

        print('~~~~~~source shape is', X_s.size())
        print('==========target shape is', X_t.size())


        return X_s,Y_s,X_t,Y_t



class source_target_handler(Dataset):

    def __init__(self, source_name, target_name,
                X_s, Y_s, X_t, Y_t,
                transform_source = None, transform_target = None):

        self.Xs = X_s
        self.Ys = Y_s
        self.Xt = X_t
        self.Yt = Y_t
        self.transform_source = transform_source
        self.transform_target = transform_target
        self.source_name = source_name
        self.target_name = target_name

    def __len__(self):
        # returning the minimum length of two data-sets
        return min(len(self.Ys),len(self.Yt))

    def __getitem__(self, index):
        Len1 = len(self.Ys)
        Len2 = len(self.Yt)

        # checking the index in the range or not

        if index < Len1:
            x_s = self.Xs[index]
            y_s = self.Ys[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_s = self.Xs[re_index]
            y_s = self.Ys[re_index]

        # checking second datasets
        if index < Len2:

            x_t = self.Xt[index]
            y_t = self.Yt[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_t = self.Xt[re_index]
            y_t = self.Yt[re_index]

        if self.transform_source is not None:

            if self.source_name == 'MNIST':
                x_s = Image.fromarray(x_s.numpy(), mode='L')
                x_s = self.transform_source(x_s)

            elif self.source_name == 'SVHN':
                x_s = Image.fromarray(np.transpose(x_s.numpy(), (1, 2, 0)))
                x_s = self.transform_source(x_s)

            elif self.source_name == 'USPS':
                x_s = Image.fromarray(np.transpose(x_s, (1, 2, 0)))
                x_s = self.transform_source(x_s)

            elif self.source_name == 'MNIST_M':
                x_s = Image.fromarray(x_t.numpy())
                x_s = self.transform_source(x_s)

        if self.transform_target is not None:
            if self.target_name == 'MNIST':
                x_t = Image.fromarray(x_t.numpy(), mode='L')
                x_t = self.transform_target(x_t)
            elif self.target_name == 'SVHN':
                x_t = Image.fromarray(np.transpose(x_t.numpy(), (1, 2, 0)))
                x_t = self.transform_target(x_t)
            elif self.target_name == 'USPS':
                x_t = Image.fromarray(np.transpose(x_t.numpy(), (1, 2, 0)))
                x_t = self.transform_target(x_t)
            elif self.target_name == 'MNIST_M':
                x_t = Image.fromarray(x_t.numpy())
                x_t = self.transform_target(x_t)



        return  index,x_s,y_s,x_t,y_t


class target_te_handler(Dataset):

    def __init__(self, X, Y, target_name, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.target_name = target_name

    def __getitem__(self, index):

        x, y = self.X[index], self.Y[index]

        if self.transform is not None:
            if self.target_name == 'MNIST':
                x = Image.fromarray(x.numpy(), mode='L')
                x = self.transform(x)
            elif self.target_name == 'SVHN':
                x = Image.fromarray(np.transpose(x.numpy(), (1, 2, 0)))
                x = self.transform(x)
            elif self.target_name == 'USPS':
                x = Image.fromarray(np.transpose(x.numpy(), (1, 2, 0)))
                x = self.transform(x)
            elif self.target_name == 'MNIST_M':
                x = Image.fromarray(x.numpy())
                x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.Y)


class update_dataset(): #Dataset This one will not be used since remove function has some errors
    """
    This is version one on how to deal with the queried target data
    
    
    This class serves as update the source and target dataset after query
    That is, push the queried data from target domain into target domain
    then extract those instances from the target domain
    
    Note, we already load X_t_tr and Y_t_tr so we can directly work on them
    """

    def __init__(self, 
                 source,
                 source_label,
                 target,
                 target_label,
                 idxs_lb,
                 return_id=False):
        '''
        parms: source: [idx, image_size, image_size, 1 or 3] 
               source_label: [idx]
               
               
        '''
        
        
        self.return_id = return_id
        
        self.X_s = source
        self.Y_s = source_label
        self.X_t = target
        self.Y_t = target_label
        self.n_pool = len(self.Y_t)
       
        self.idxs_lb = idxs_lb # len(idx_lb) = n_pool, if one image i is queried, idx_lb[i] = True, else:False

        #self.load_query_data()
        self.query_idx = np.where(self.idxs_lb)[0]
        self.return_id = return_id
        
    def load_query_data(self):
        # load the queried data and labels
        self.query_labels = self.Y_t[self.query_idx]
        self.query_datas = self.X_t[self.query_idx]

        return self.query_datas, self.query_labels
        

    def return_new_target_idx(self):
        # Here during training, we used the labeled target together with source, 
        # so we need to remove them from the target dataset
        self.idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        
        return self.idxs_unlabeled





