import numpy as np
import torch
from dataloaders import single_dataset, single_handler, source_target_dataset,source_target_handler,target_te_handler
from model import Net_fea,Net_clf,Net_dis
from torchvision import transforms

from active_da import active_da_digits
import argparse

import collections
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms.transforms import *

parser = argparse.ArgumentParser()

parser.add_argument("lr_fea_cls", type = float, help="learning_rate", default=1e-5)
parser.add_argument("lr_fea_dis", type = float, help="learning_rate", default=1e-5)
parser.add_argument("lr_dis", type = float, help="learning_rate", default=1e-5)


parser.add_argument("gp_param", type = float, default=10, help='trade-off coefficient for gradient penalty') # Used as trade_off *(-wass + gp_coeff*gp)
parser.add_argument('w_d_round', type=int, default=10, help='Used to repeat some times for discriminator training')
parser.add_argument('w_d_param',type=float, default=0.1, help='Used to repeat some times for discriminator training')
parser.add_argument('source',type=str, help='The source domain M F U or S')
parser.add_argument('target',type=str, help='The target domain M F U or S')
#parser.add_argument('active_round',type=int, default = 20 ,help='How many round of active training round')
#parser.add_argument('weight_query',type=float, default = 1 ,help='weight of queried instances in the cls loss in the query train round')
#parser.add_argument('weight_source',type=float, default = 1 ,help='weight of source instances in the cls loss in the query train round')
parser.add_argument('weight_decay',type=float, default = 0 ,help='weight decay in the optimizer')
parser.add_argument('use_gp_in_clf',type=float, default = 0 ,help='use gp in clf round')

parser.add_argument('lr_dis_active',type=float, default = 1e-3 ,help='lr_dis_active round')
parser.add_argument('lr_fea_active',type=float, default = 1e-3 ,help='lr fea active round')
parser.add_argument('lr_clf_active',type=float, default = 1e-3 ,help='lr clf active round')
#parser = argparse.ArgumentParser()
#parser.add_argument("lr_fea", type = float, help="feature extractor learning_rate")
#parser.add_argument("trade_off", type = float, help='trade-off coefficient')
#parser.add_argument("gp_pen", type = float, help='gradient-penality coefficient')

args = parser.parse_args()

#learning_rate = args.lr
#trade_off     = args.trade_off
#gp_pen        = args.gp_pen


# Data and configuration settings
# source_dataset = 'MNIST'
# target_dataset = 'MNIST_M'

print(args)
if args.source == 'S':
    source_domain = 'SVHN'
if args.source == 'M':
    source_domain = 'MNIST'
if args.source == 'F':
    source_domain = 'MNIST_M'
if args.source == 'U':
    source_domain = 'USPS'

if args.target == 'S':
    target_domain = 'SVHN'
if args.target == 'M':
    target_domain = 'MNIST'
if args.target == 'F':
    target_domain = 'MNIST_M'
if args.target == 'U':
    target_domain = 'USPS'





config = {'image_size':28}
# Define some transforms for different dataset

args_pool = {'MNIST':
                {'transform': transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081))
                                                  ]),
                 },
            'SVHN':
                {'transform': transforms.Compose([
                                                  transforms.Resize(config['image_size']),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
                                                ]),
                 },
            'USPS':
                {'transform': transforms.Compose([
                                                  transforms.Resize(config['image_size']),
                                                  transforms.Grayscale(num_output_channels=3),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.2469, 0.2469, 0.2469), (0.2989, 0.2989, 0.2989))
                                                 ]),
                 },
            'MNIST_M':
                {'transform': transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4514, 0.4500, 0.4506), (0.2515,0.2522,0.2530))
                                                ]),
                 }

            }

args_common = { 'target_dataset':target_domain,
               'source_dataset':source_domain,
               'lr_fea_cls':args.lr_fea_cls,
               'lr_fea_dis':args.lr_fea_dis,
               'lr_dis'    : args.lr_dis,
               'lr_fea_active': args. lr_fea_active,
               'lr_clf_active': args.lr_clf_active,
               'lr_dis_active': args.lr_dis_active,
               'source_training_epoch': 3,
               'gradient_penalty_pamtr':args.gp_param,
               'weight_decay': args.weight_decay,
               'w_d_round':args.w_d_round,  
               'w_d_param':args.w_d_param, 
               'query_budget':0.1,
               'adv_epoch':45,
               'adv_clf_round':1,
               'use_trade_off': False,
               'use_gp_clf':args.use_gp_in_clf,
               'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
               'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
               'num_class': 10}



print(args_common)
args_s = args_pool[source_domain]
args_t = args_pool[target_domain]



# loading source target data
#X_s_tr,Y_s_tr, X_t_tr,Y_t_tr = source_target_dataset( source_name=source_dataset,
#                                                      target_name=target_dataset,
#                                                      train=True,download=True).get_dataset()

source_dataset = single_dataset(data_name= source_domain, train=True, download=True)
X_s, Y_s = source_dataset.return_dataset()

#source_dataset_te = single_dataset(data_name= source_domain, train=True, download=True)
#X_s_te, Y_s_te = source_dataset.return_dataset()

target_dataset = single_dataset(data_name= target_domain, train= True, download=True)
X_t, Y_t = target_dataset.return_dataset()
# Note: After query, we should test the accuracy use the test transform


# X_s_te,Y_s_te,_,_ = source_target_dataset(source_name=source_dataset,
#                                                     target_name=target_dataset,
#                                                     train=False,download=True).get_dataset()

# Now we shall define the query budget

n_pool = len(Y_t)
print('number of unlabeled pool in target domain: {}'.format(n_pool))
num_query = int(args_common['query_budget'] * n_pool) 
print('Active query budget number in the target domain: {}'.format(num_query))
args_common.update({'num_query':num_query})

# load the network
net_fea, net_clf, net_dis = Net_fea, Net_clf, Net_dis

# Initialize the strategy
strategy = active_da_digits(X_s_tr = X_s, Y_s_tr = Y_s, X_t_tr= X_t, Y_t_tr=Y_t,  
                              net_fea = net_fea, net_clf=net_clf, net_dis=net_dis, args_s = args_s, args_t = args_t , args=args_common)

strategy.adversarial_train()
#strategy.random_active()
#strategy.train_source()
#strategy.WL_active()
print('Done~')

