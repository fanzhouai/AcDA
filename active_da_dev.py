import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from util import*
from dataloaders import single_dataset, single_handler

import random
import collections
from collections import Counter

# setting gradient values






class active_da_digits():

    def __init__(self, X_s_tr, Y_s_tr, X_t_tr, Y_t_tr, 
                net_fea, net_clf, net_dis, args_s, args_t, args):


        # Firstly load the source and target domain data. We only load them once to aviod potential effects in indexs 
        self.X_s_tr = X_s_tr
        self.Y_s_tr = Y_s_tr
        self.X_t_tr = X_t_tr
        self.Y_t_tr = Y_t_tr
        

        #self.total_epoch = 100
        
        self.args = args
        self.args_s = args_s
        self.args_t = args_t

        self.target_name = self.args['target_dataset']
        self.source_name = self.args['source_dataset']

        
        #self.learning_rate = self.args['optimizer_args']['lr']
        # Pass the learning rate lr_fea_cls is used for supervised training process
        # lr_fea_dis is used for adversarial training process
        # lr_dis is the learning rate for discriminator
        self.lr_fea_cls = self.args['lr_fea_cls']
        self.lr_fea_dis = self.args['lr_fea_dis']
        self.lr_dis     = self.args['lr_dis']

        
        self.source_loader = DataLoader(single_handler(X=X_s_tr, Y=Y_s_tr,data_name = self.source_name ,transform= self.args_s['transform'] ))
        self.target_loader = DataLoader(single_handler(X=X_t_tr, Y=Y_t_tr,data_name = self.target_name ,transform= self.args_t['transform'] ))


        # At begining we don't initialize any labeled idx, that is, idx_lb = empty

        #self.idx_lb  = idx_lb
        # Initialize the network 

        self.net_fea = net_fea()
        self.net_clf = net_clf()
        self.net_dis = net_dis()
        
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.fea = self.net_fea.to(self.device)
        self.clf = self.net_clf.to(self.device)
        self.dis = self.net_dis.to(self.device)


        #self.test_handler = test_handler
        #self.DA_handler = DA_handler
         
        self.source_training_epoch = self.args['source_training_epoch']
        self.adv_epoch = 40 # self.args['adv_training_epoch']     
        self.active_step = 2000   




        #self.selection = 10

        self.gp_param = self.args['gradient_penalty_pamtr']


        self.weight_decay = self.args['weight_decay']

        self.w_d_round = self.args['w_d_round'] # How many time wasserstein distance being computed in the critic round
        self.w_d_param = self.args['w_d_param'] # Weights for wasserstein distance in the classification round
        self.use = self        

        # Initialize the list for query
        self.n_pool  = len(Y_t_tr) # Unlabeled target pool
        
        self.num_query = self.args['num_query']
        self.idxs_lb = np.zeros(self.n_pool, dtype=bool) # Initialize the query list with all zeros
        
        #self.active_round = self.args['active_round']
        #self.weight_q = self.args['weight_of_query']

    def train_source(self):
        '''
        Source pre-training round
        Train on source only and test on target
         '''

        print('[Training on the source dataset]')
        for epoch in range(self.source_training_epoch):

            # setting the training mode in the beginning of EACH epoch
            # (since we need to compute the training accuracy during the epoch, optional)

            self.fea.train()
            self.clf.train()

            opt_fea = optim.Adam(self.fea.parameters(), lr = self.lr_fea_cls)
            opt_clf = optim.Adam(self.clf.parameters(), lr = self.lr_fea_cls,weight_decay =self.weight_decay)
                
            total_acc = []
            total_acc_source = []
            total_loss = []

                #opt_fea = optim.SGD(self.fea.parameters(),**self.args['optimizer_args'])
                #opt_clf = optim.SGD(self.clf.parameters(),**self.args['optimizer_args'])
                #opt_dis = optim.SGD(self.dis.parameters(),**self.args['optimizer_args'])
                #opt_dis_s_t = optim.SGD(self.dis.parameters(),**self.args['optimizer_args']

            batches = zip(self.source_loader, self.target_loader)
            n_batches = min(len(self.source_loader), len(self.target_loader))
                #criterion = nn.CrossEntropyLoss()

            for (x_s, y_s), (x_t, y_t) in tqdm(batches, leave=False, total= n_batches):
                    # Here we train the network only on source data.
                    # We take the target data only for test, that means the row source only
                    # We shall also use it as target only. It will be the same strategy
                  
                    

                x_s, y_s, x_t, y_t = x_s.cuda(), y_s.cuda(), x_t.cuda(), y_t.cuda()
                    
                opt_fea.zero_grad()
                opt_clf.zero_grad()

                z_s = self.fea(x_s)
                z_t = self.fea(x_t)

                pred_class, _ = self.clf(z_s)

                clf_loss      = F.cross_entropy(pred_class,y_s)
                loss_source   = clf_loss.mean()


                total_loss.append(loss_source)
                
                loss_source.backward()
                    
                opt_fea.step()
                opt_clf.step()

                opt_clf.zero_grad()
                opt_fea.zero_grad()



            acc_s = self.compute_accuracy(self.X_s_tr,self.Y_s_tr,self.source_name,source=True)
            acc_t = self.compute_accuracy(self.X_t_tr,self.Y_t_tr,self.target_name,source=False)
            avg_loss = torch.tensor(total_loss).mean()

            print("Classification error in epoch {0:d} is {1:.3f}".format(epoch,avg_loss))
            print("Accuracy for source in epoch {0:d} is {1:.3f}".format(epoch,acc_s*100))
            print("Accuracy for target in epoch {0:d} is {1:.3f}".format(epoch,acc_t*100))

    def adversarial_train(self): 
        
        '''
        Adversarial Training using wasserstein distance
        Train on the source (labeled) and target domain (unlabeled)

        
        '''
        # Firstly, we shall train on source domain for servaral epochs as an initialization

        self.train_source() 

        # Then, we start the adversarial training using WDGRL

            
        # setting the training mode in the beginning of EACH epoch
        # (since we need to compute the training accuracy during the epoch, optional)
        self.use_gp_clf = self.args['use_gp_clf']
        if self.use_gp_clf:
            print('We use the gradient penalty in the clf round')
            
        total_acc = 0
        total_acc_source = 0
        total_loss = 0
        total_acc_s = 0
        total_acc_t = 0

        batches = zip(self.source_loader, self.target_loader)
        
            
            # Epoch greater than the source training epochs, then, we can start the adversarial training

            #batches = zip(self.source_loader, self.target_loader)
        
        #n_batches = 0

        self.fea.train()
        self.clf.train()
        self.dis.train()

        opt_fea = optim.Adam(self.fea.parameters() , lr=self.lr_fea_dis)
        opt_clf = optim.Adam(self.clf.parameters(), lr=self.lr_fea_dis,weight_decay=self.weight_decay)
        opt_dis = optim.Adam(self.dis.parameters(), lr=self.lr_dis)
        n_batches = min(len(self.source_loader), len(self.target_loader))
        clf_criterion = nn.CrossEntropyLoss()
        for epoch in range(self.adv_epoch):
        
            iters = tqdm(zip(self.source_loader, self.target_loader), desc=f'epoch {epoch} ', total=n_batches)
            correct_s = 0
            correct_t = 0

            for i, ((x_s, y_s), (x_t, y_t)) in enumerate(iters):

                # We adopt this coefficient trade-off from DANN setting
                p = float(i + (epoch)* n_batches) / (epoch+1) / n_batches
                trade_off = 2. / (1. + np.exp(-10 * p)) - 1
                    
                opt_fea.zero_grad()
                opt_clf.zero_grad()
                opt_dis.zero_grad()

                x_s, y_s, x_t, y_t = x_s.cuda(), y_s.cuda(), x_t.cuda(), y_t.cuda()
            
                # train the domain critic first

                set_requires_grad(self.fea, requires_grad=False)
                set_requires_grad(self.clf, requires_grad=False)
                set_requires_grad(self.dis, requires_grad=True)

                with torch.no_grad():
                    z_s = self.fea(x_s)
                    z_t = self.fea(x_t)

                for _ in range(self.w_d_round):

                    # gradient ascent for multiple times like GANS training

                    gp_s_t = gradient_penalty(self.dis, z_s, z_t)

                        
                    wassertein_source_and_target = self.dis(z_s).mean() - self.dis(z_t).mean()

                    dis_s_t_loss = (-1 * wassertein_source_and_target + self.gp_param * gp_s_t)
                    #dis_s_t_loss = (-1 * wassertein_source_and_target + self.gp_param * gp_s_t * 2)

                            # Currently we don't apply any weights on w-disstance loss

                            
                    dis_s_t_loss.backward()
                    opt_dis.step() # 

                        
                # Then, we shall train the feature extractor and classifier
                        
                self.fea.train()
                self.clf.train()
                self.dis.eval()

                set_requires_grad(self.fea, requires_grad=True)
                set_requires_grad(self.clf, requires_grad=True)
                set_requires_grad(self.dis, requires_grad=False)

                opt_fea.zero_grad()
                opt_clf.zero_grad()
                opt_dis.zero_grad()
                
                lmbd = torch.tensor(1.)
                l2_reg_fea = torch.tensor(0.)

                for name, param in self.fea.named_parameters():
                    if 'weight' in name:
                        l2_reg_fea += torch.norm(param)
                for name, param in self.clf.named_parameters():
                    if 'weight' in name:
                        l2_reg_clf += torch.norm(param)

                l2_loss += lmbd * (l2_reg_fea+l2_reg_clf)

                # Here we don't use fc1_s but to compute source_z again to avoid potentail variable gradient conflict issue
                for _ in range(self.adv_clf_round):
                    
                    source_z = self.fea(x_s)
                    target_z = self.fea(x_t)
                    
                    pred_source,_  = self.clf(source_z)
                    clf_loss = clf_criterion(pred_source, y_s)
                    
                    wassertein_source_and_target = self.dis(source_z).mean() - self.dis(target_z).mean()
                    #print('wasserstein source and target', wassertein_source_and_target)
                    # Then compute the gradient penalty
                    gp_s_t = gradient_penalty(self.dis, target_z, source_z)

                    # Alpha is a hyper-prmtr for fine-tuning. Typically when network got deeper, the higher 
                    #loss = cls_loss + (-wassertein_source_and_target - self.alpha * gp_s_t * 2)
                    '''
                    if self.use_trade_off:
                        loss = clf_loss.mean() + 1.0*trade_off*(wassertein_source_and_target)
                    else:
                        loss = clf_loss.mean() + self.w_d_param*(wassertein_source_and_target)
                    '''
                    if self.use_trade_off:
                        loss = clf_loss.mean() + 1.0*trade_off*(wassertein_source_and_target)+self.l2_weight*l2_loss
                    if self.use_gp_clf:
                        loss = clf_loss.mean() + self.w_d_param*(wassertein_source_and_target- self.gp_param * gp_s_t * 2)+self.l2_weight*l2_loss
                    else:
                        loss = clf_loss.mean() + self.w_d_param*(wassertein_source_and_target)+self.l2_weight*l2_loss
                    #elif self.use_negative_wa:
                    #    loss = clf_loss.mean() + self.w_d_param*(-wassertein_source_and_target + alpha * gp_s_t * 2)

                    #loss = cls_loss + (alpha * wassertein_source_and_target - alpha * gp_s_t * 2)
                    #total_loss +=loss
                    loss.backward()
                    opt_fea.step()
                    opt_clf.step()

                    opt_fea.zero_grad()
                    opt_clf.zero_grad()
                    opt_dis.zero_grad()                  


                self.fea.eval()
                self.clf.eval()


                with torch.no_grad():
                    latent_s = self.fea(x_s)
                    latent_t = self.fea(x_t)
                    
                    out_t, _ = self.clf(latent_t)
                    out_s, _ = self.clf(latent_s)

                #pred_t = out_t.max(1)[1]
                predt = out_t.max(1)[1]
                preds = out_s.max(1)[1]

                correct_t += (predt == y_t).float().mean().item()
                correct_s += (preds == y_s).float().mean().item()
                
            acc_s = 1.0*correct_s/n_batches
            acc_t = 1.0*correct_t/n_batches
            
            
            #print("Classification error in epoch {0:d} is {1:.3f}".format(epoch,avg_loss))
            print("Accuracy for source in epoch {0:d} is {1:.3f}".format(epoch,acc_s*100))
            print("Accuracy for target in epoch {0:d} is {1:.3f}".format(epoch,acc_t*100))

    def _adversarial_train(self):
        
        '''
        Adversarial Training using wasserstein distance
        Train on the source (labeled) and target domain (unlabeled)

        
        '''
        # Firstly, we shall train on source domain for servaral epochs as an initialization

        self.train_source() 

        # Then, we start the adversarial training using WDGRL

            
        # setting the training mode in the beginning of EACH epoch
        # (since we need to compute the training accuracy during the epoch, optional)


            
        total_acc = 0
        total_acc_source = 0
        total_loss = 0
        total_acc_s = 0
        total_acc_t = 0

        batches = zip(self.source_loader, self.target_loader)
        
            
            # Epoch greater than the source training epochs, then, we can start the adversarial training

            #batches = zip(self.source_loader, self.target_loader)
        
        #n_batches = 0

        self.fea.train()
        self.clf.train()
        self.dis.train()

        opt_fea = optim.Adam(self.fea.parameters() , lr=self.lr_fea_dis)
        opt_clf = optim.Adam(self.clf.parameters(), lr=self.lr_fea_dis,weight_decay=self.weight_decay)
        opt_dis = optim.Adam(self.dis.parameters(), lr=self.lr_dis)
        n_batches = min(len(self.source_loader), len(self.target_loader))

        iters = tqdm(zip(self.source_loader, self.target_loader), desc=f'epoch {epoch} ', total=n_batches)
                #epoch_id += 1
        for epoch in range(self.adv_epoch):

            for i, ((x_s, y_s), (x_t, y_t)) in enumerate(iters):

                # We adopt this coefficient trade-off from DANN setting
                p = float(i + (epoch)* n_batches) / (epoch) / n_batches
                trade_off = 2. / (1. + np.exp(-10 * p)) - 1
                    
                opt_fea.zero_grad()
                opt_clf.zero_grad()
                opt_dis.zero_grad()

                x_s, y_s, x_t, y_t = x_s.cuda(), y_s.cuda(), x_t.cuda(), y_t.cuda()
            
                # train the domain critic first

                set_requires_grad(self.fea, requires_grad=False)
                set_requires_grad(self.clf, requires_grad=False)
                set_requires_grad(self.dis_s_t, requires_grad=True)


                z_s = self.fea(x_s)
                z_t = self.fea(x_t)

                for _ in range(self.w_d_round):

                    # gradient ascent for multiple times like GANS training

                    gp_s_t = gradient_penalty(self.dis_s_t, z_s, z_t)

                        
                    wassertein_source_and_target = self.dis_s_t(z_s).mean() - self.dis_s_t(z_t).mean()

                    dis_s_t_loss = (-1 * wassertein_source_and_target + self.gp_param * gp_s_t * 2)
                            # Currently we don't apply any weights on w-disstance loss

                            
                    dis_s_t_loss.backward()
                    opt_dis_s_t.step() # 

                        
                # Then, we shall train the feature extractor and classifier
                        
                self.fea.train()
                self.clf.train()
                self.dis_s_t.eval()

                set_requires_grad(self.fea, requires_grad=True)
                set_requires_grad(self.clf, requires_grad=True)
                set_requires_grad(self.dis_s_t, requires_grad=False)

                        

                fc1_s       = self.fea(x_s)
                pred_source,_  = self.clf(fc1_s) # second term could be used for TSNE, currently we don;t need it
                
                # Compute CrossEntropy loss for classification

                clf_loss = F.cross_entropy(pred_source, y_s)
                
                
                # Here we don't use fc1_s but to compute source_z again to avoid potentail variable gradient conflict issue
                source_z = self.fea(x_s)
                target_z = self.fea(x_t)

                # Then compute the wasserstein distance between source and target
                wassertein_source_and_target = self.dis(source_z).mean() - self.dis(target_z).mean()
                # Then compute the gradient penalty
                gp_s_t = gradient_penalty(self.dis, target_z, source_z)
                        
                # Alpha is a hyper-prmtr for fine-tuning. Typically when network got deeper, the higher 
                #loss = cls_loss + (-wassertein_source_and_target - self.alpha * gp_s_t * 2)
                if self.use_trade_off:
                    loss = clf_loss.mean() + 1.0*trade_off*(-wassertein_source_and_target + self.gp_param * gp_s_t * 2)
                else:
                    loss = clf_loss.mean() + self.w_d_param*(-wassertein_source_and_target + self.gp_param * gp_s_t * 2)
                #loss = cls_loss + (alpha * wassertein_source_and_target - alpha * gp_s_t * 2)

                loss.backward()
                opt_fea.step()
                opt_clf.step()                   
                    
            ## Then we shall evaluate the accuracy of the
            acc_s = self.compute_accuracy(self.X_s_tr,self.Y_s_tr,self.source_name,source=True)
            acc_t = self.compute_accuracy(self.X_t_tr,self.Y_t_tr,self.target_name,source=False)
            avg_loss = torch.tensor(total_loss).mean()

            print("Classification error in epoch {0:d} is {1:.3f}".format(epoch,avg_loss))
            print("Accuracy for source in epoch {0:d} is {1:.3f}".format(epoch,acc_s*100))
            print("Accuracy for target in epoch {0:d} is {1:.3f}".format(epoch,acc_t*100))
                            
    def random_active(self):
        
        idxs_lb = np.zeros(self.n_pool, dtype=bool)
        idxs_tmp = np.arange(self.n_pool)
        
        np.random.shuffle(idxs_tmp)
        print('[We enter the active training stage]')
        
        self.idxs_lb[idxs_tmp[:self.num_query]] = True # Randomly choose serveral idx 
        
        updated = update_dataset(source = self.X_s_tr, source_label = self.Y_s_tr,
                                 target = self.X_t_tr, target_label = self.Y_t_tr,
                                 idxs_lb= idxs_lb, return_id=False)


        X_q, Y_q = updated.load_query_data() # Load the query data and corresponding labels
        #print('We finish the query process, the original source domain has size '+str(len(self.Y_s_tr)) +' current set size is '+str(len(Y_s_new)))
        queryset = single_handler(X_q,Y_q, data_name = self.target_name, transform=self.args_t['transform']) # We query the data from the target set, so use the target transform

        query_loader = DataLoader(queryset, batch_size=16, shuffle=True, drop_last = True)
        # Then, we should update the new target dataset since we remove some instances from the original target
        new_target_idx = updated.return_new_target_idx()
        target_loader = DataLoader(single_handler(X=X_t_tr[new_target_idx], Y=Y_t_tr[new_target_idx],data_name = self.target_name ,transform= self.args_t['transform'] ))

        # get the length of the loaders, used for iter manuplation
        
        len_q_loader = len(query_loader)
        len_s_loader = len(self.source_loader)
        len_t_loader = len(target_loader)
        print('the new target loader len is:', len_t_loader)
        n_batches = 0
        total_acc_s = 0
        total_acc_t = 0

        for step in range(self.active_step):
            best_acc_s = 0
            best_acc_t = 0

            
            opt_fea = optim.Adam(self.fea.parameters() , lr=1e-5)
            opt_clf = optim.Adam(self.clf.parameters(), lr=1e-5,weight_decay = self.weight_decay)
            opt_dis_s_t = optim.Adam(self.dis.parameters(), lr=1e-5)

            if step % len_s_loader == 0:
                iter_s = iter(self.source_loader)
            if step % len_t_loader == 0:      
                iter_t = iter(target_loader)
            if step % len_q_loader == 0:
                iter_q = iter(query_loader)
            
            opt_fea.zero_grad()
            opt_clf.zero_grad()
            
            x_s, y_s = next(iter_s)
            x_t, y_t = next(iter_t)
            x_q, y_q = next(iter_q)
            n_batches+=1

            x_s, y_s, x_t, y_t, x_q, y_q = x_s.cuda(), y_s.cuda(), x_t.cuda(), y_t.cuda(), x_q.cuda(), y_q.cuda()
            
            set_requires_grad(self.fea, requires_grad=False)
            set_requires_grad(self.clf, requires_grad=False)
            set_requires_grad(self.dis, requires_grad=True)

            z_s = self.fea(x_s)
            z_t = self.fea(x_t)


            for _ in range(self.w_d_round):

                # gradient ascent for multiple times like GANS training

                gp_s_t = gradient_penalty(self.dis, z_s, z_t)

                wassertein_source_and_target = self.dis(z_s).mean() - self.dis(z_t).mean()

                dis_s_t_loss = (-1 * wassertein_source_and_target + self.gp_param * gp_s_t * 2)
                            # Currently we don't apply any weights on w-disstance loss

                            
                dis_s_t_loss.backward()
                opt_dis_s_t.step() # 

                        
            # Then, we shall train the feature extractor and classifier

            self.fea.train()
            self.clf.train()
            self.dis_s_t.eval()

            set_requires_grad(self.fea, requires_grad=True)
            set_requires_grad(self.clf, requires_grad=True)
            set_requires_grad(self.dis_s_t, requires_grad=False)

            s_data = torch.cat((x_s, x_q), 0)
            s_label = torch.cat((y_s, y_q), 0)

            s_feature = self.fea(s_data)
            #fc1_s, feature_source, fc2_s, predict_prob_source = self.clf(s_feature)
            
            fc1_s       = self.fea(x_s)
            pred_source,_  = self.clf(fc1_s)

            ce = F.cross_entropy(pred_source, y_s)

            clf_loss = torch.mean(ce, dim=0, keepdim=True)


            source_z = self.fea(x_s)
            target_z = self.fea(x_t)

            # Then compute the wasserstein distance between source and target
            wassertein_source_and_target = self.dis(source_z).mean() - self.dis(target_z).mean()
             # Then compute the gradient penalty
            gp_s_t = gradient_penalty(self.dis, target_z, source_z)
                        
            # Alpha is a hyper-prmtr for fine-tuning. Typically when network got deeper, the higher 
            #loss = cls_loss + (-wassertein_source_and_target - self.alpha * gp_s_t * 2)
            loss = clf_loss + self.w_d_param*(-wassertein_source_and_target + self.gp_param * gp_s_t * 2)
            #loss = cls_loss + (alpha * wassertein_source_and_target - alpha * gp_s_t * 2)

                #print('loss_source', loss_source)
                
            #total_loss += loss_source
                #loss_source = F.cross_entropy(pred,y_s)
                #loss_source = nn.cross_entropy(pred, y_s)

            loss.backward()
            opt_fea.step()
            opt_clf.step()

                ########### Predict source and target acc
            self.fea.eval()
            self.clf.eval()
            
            with torch.no_grad():
                latent_s = self.fea(x_s)
                latent = self.fea(x_t)
                
            out_s1,_ = self.clf(latent_s)
            out1,_ = self.clf(latent)
            
            total_acc_t    += (out1.max(1)[1] == y_t).float().mean().item()
            total_acc_s  += (out_s1.max(1)[1] == y_s).float().mean().item()
                                
            acc_t = 100.0* total_acc_t/n_batches
            acc_s = 100.0 * total_acc_s/n_batches
            print('total batch num is', n_batches)
            print('==========Inner epoch {:d} ========'.format(epoch))
            print('     Mean acc on target domain is ', acc_t)
            print('     Mean acc on source domain is ', acc_s)


    def query(self,query_num):

        """
        adversarial query strategy

        :param n:
        :return:

        """

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        # prediction output probability
        probs = self.predict_prob(self.X_t_tr, self.Y_t_tr)



        # uncertainly score (three options, single_worst, L2_upper, L1_upper)
        # uncertainly_score = self.single_worst(probs)
        uncertainly_score = 0.5* self.L2_upper(probs) + 0.5* self.L1_upper(probs)

        # print(uncertainly_score)


        # prediction output discriminative score
        dis_score = self.pred_dis_score(self.X_t_tr, self.Y_t_tr)

        # print(dis_score)


        # computing the decision score
        total_score = uncertainly_score - self.selection * dis_score
        # print(total_score)
        b = total_score.sort()[1][:query_num]
        # print(total_score[b])


        # sort the score with minimal query_number examples
        # (expected value outputs from smaller to large)

        return idxs_unlabeled[total_score.sort()[1][:query_num]]
    def query_using_entropy_uncertainty(self,query_num):

        """
        adversarial query strategy

        :param n:
        :return:

        """

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        # prediction output probability
        probs = self.predict_prob(self.X_t_tr, self.Y_t_tr)



        # uncertainly score (three options, single_worst, L2_upper, L1_upper)
        # uncertainly_score = self.single_worst(probs)

        uncertainly_score = self.entropy_uncertainty(probs)

        # print(uncertainly_score)

        # prediction output discriminative score
        dis_score = self.pred_dis_score(self.X_t_tr, self.Y_t_tr)

        # print(dis_score)


        # computing the decision score
        total_score = uncertainly_score - self.selection * dis_score
        # print(total_score)
        b = total_score.sort()[1][:query_num]
        # print(total_score[b])


        # sort the score with minimal query_number examples
        # (expected value outputs from smaller to large)

        return idxs_unlabeled[total_score.sort()[1][:query_num]]


    def WL_active(self,uncertainly_method):
        '''
        This fucntion we adopt the WAAL active learning strategy for query the most informative instances
        '''
        # Initialize a list of query idx

        idxs_lb = np.zeros(self.n_pool, dtype=bool)
        #idxs_tmp = np.arange(self.n_pool)
        
        #np.random.shuffle(idxs_tmp)
        
       
        print('[We enter the active training stage]')
        ########### Query budget of instances
        print('[WL Active Strategy]')

        if uncertainly_method == 'entropy':
            q_idxs = self.query_using_entropy_uncertainty(self.num_query)
        elif uncertainly_method == 'L2':
            q_idxs = self.query(self.num_query)
        else:
            print('invailid uncertainty indicator')
        

        idxs_lb[q_idxs] = True

        # update
        self.update(idxs_lb)
        ################


        updated = update_dataset( source_idx_list = self.X_s_tr,
                 source_label_list= self.Y_s_tr,
                 target_idx_list= self.X_t_tr,
                 target_label_list = self.Y_t_tr,
                 idxs_lb = self.idxs_lb)
        

        X_q, Y_q = updated.load_query_data() # Load the query data and corresponding labels
        #print('We finish the query process, the original source domain has size '+str(len(self.Y_s_tr)) +' current set size is '+str(len(Y_s_new)))
        queryset = query_dataset(X_q,Y_q, transform=self.args['train_transform'])

        query_loader = DataLoader(queryset, batch_size=16, shuffle=True, drop_last = True)
        
        # get the length of the loaders, used for iter manuplation
        len_q_loader = len(query_loader)
        len_s_loader = len(self.source_loader)
        len_t_loader = len(self.target_loader)
        n_batches = 0
        total_acc_s = 0
        total_acc_t = 0

        for step in range(2600):
            best_acc_s = 0
            best_acc_t = 0

            
            opt_fea = optim.Adam(self.fea.parameters() , lr=1e-5)
            opt_clf = optim.Adam(self.clf.parameters(), lr=1e-5,weight_decay =self.weight_decay)
            opt_dis_s_t = optim.Adam(self.dis_s_t.parameters(), lr=1e-5)

            if step % len_s_loader == 0:
                iter_s = iter(self.source_loader)
            if step % len_t_loader == 0:      
                
                iter_t = iter(self.target_loader)
            if step % len_q_loader == 0:
                iter_q = iter(query_loader)
            
            opt_fea.zero_grad()
            opt_clf.zero_grad()
            
            x_s, y_s = next(iter_s)
            x_t, y_t = next(iter_t)
            x_q, y_q = next(iter_q)
            n_batches+=1

            x_s, y_s, x_t, y_t, x_q, y_q = x_s.cuda(), y_s.cuda(), x_t.cuda(), y_t.cuda(), x_q.cuda(), y_q.cuda()
            
            set_requires_grad(self.fea, requires_grad=False)
            set_requires_grad(self.clf, requires_grad=False)
            set_requires_grad(self.dis_s_t, requires_grad=True)

            z_s = self.fea(x_s)
            z_t = self.fea(x_t)


            for _ in range(self.w_d_round):

                # gradient ascent for multiple times like GANS training

                gp_s_t = gradient_penalty(self.dis_s_t, z_s, z_t)

                wassertein_source_and_target = self.dis_s_t(z_s).mean() - self.dis_s_t(z_t).mean()

                dis_s_t_loss = (-1 * wassertein_source_and_target + self.gp_param * gp_s_t * 2)
                            # Currently we don't apply any weights on w-disstance loss

                            
                dis_s_t_loss.backward()
                opt_dis_s_t.step() # 

                        
            # Then, we shall train the feature extractor and classifier

            self.fea.train()
            self.clf.train()
            self.dis_s_t.eval()

            set_requires_grad(self.fea, requires_grad=True)
            set_requires_grad(self.clf, requires_grad=True)
            set_requires_grad(self.dis_s_t, requires_grad=False)

            s_data = torch.cat((x_s, x_q), 0)
            s_label = torch.cat((y_s, y_q), 0)

            s_feature = self.fea(s_data)
            fc1_s, feature_source, fc2_s, predict_prob_source = self.clf(s_feature)

            ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, s_label)
            loss_source = torch.mean(ce, dim=0, keepdim=True)


            source_z = self.fea(x_s)
            target_z = self.fea(x_t)

            # Then compute the wasserstein distance between source and target
            wassertein_source_and_target = self.dis_s_t(source_z).mean() - self.dis_s_t(target_z).mean()
             # Then compute the gradient penalty
            gp_s_t = gradient_penalty(self.dis_s_t, target_z, source_z)
                        
            # Alpha is a hyper-prmtr for fine-tuning. Typically when network got deeper, the higher 
            #loss = cls_loss + (-wassertein_source_and_target - self.alpha * gp_s_t * 2)
            loss = loss_source + self.w_d_param*(-wassertein_source_and_target + self.gp_param * gp_s_t * 2)
            #loss = cls_loss + (alpha * wassertein_source_and_target - alpha * gp_s_t * 2)

                #print('loss_source', loss_source)
                
            #total_loss += loss_source
                #loss_source = F.cross_entropy(pred,y_s)
                #loss_source = nn.cross_entropy(pred, y_s)

            loss.backward()
            opt_fea.step()
            opt_clf.step()

                ########### Predict source and target acc
            self.fea.eval()
            self.clf.eval()
            
            with torch.no_grad():
                latent_s = self.fea(x_s)
                latent = self.fea(x_t)
                
            _,_,_, out_s1 = self.clf(latent_s)
            _,_,_, out1 = self.clf(latent)
            
            total_acc_t  += (out1.max(1)[1] == y_t).float().mean().item()
            total_acc_s  += (out_s1.max(1)[1] == y_s).float().mean().item()
            #acc_t  = (out1.max(1)[1] == y_t).float().mean().item()

            if (step+1) % 100 ==0:          
                
                #acc_s  = (out_s1.max(1)[1] == y_s).float().mean().item()
                acc_s = 100.0* total_acc_s/ n_batches
                acc_t = 100.0* total_acc_t/ n_batches
                if best_acc_s< acc_s:
                    best_acc_s = acc_s
                if best_acc_t< acc_t:
                    best_acc_t = acc_t
                total_acc_t = 0
                total_acc_s = 0
                n_batches = 0
                #print('total batch num is', n_batches)
                print('==========Step {:d} ========'.format(step))
                print('     Mean acc on target domain is ', acc_t)
                print('     Mean acc on source domain is ', acc_s)
                # print('Training Loss {:.3f}'.format(Total_loss))
                # print('Training accuracy {:.3f}'.format(acc*100))



    def uncertainty_score(self, uncertainty_method):
        '''
        compute the uncertainty score in active query round
        uncertainty_method: str, different kind of uncertainty score computation
        '''
        
        pass
        



'''

##################################################3

        
    def eval(self, x, y):
        # An eval step
        '''
        Input x, y 
        return evaluation acc
        '''
        self.fea.eval()

        ########### Predict source and target acc
        self.fea.eval()
        self.clf.eval()

        latent = self.fea(x)
        out    = self.clf(latent)
        
        pred = out.max(1)[1]
        probs[idxs] = pred.cpu()


        acc = 1.0 * (Y==probs).sum().item() / len(Y)




                


    def compute_accuracy(self,X,Y,data_name,source = False):

        """
        Computing the prediction accuracy
        :param X:
        :param Y:
        :param data_name: the given data name, then computing the corresponding accuracy (either for source or target domain)
        :param source: indicator to measure it is source or target
        :return:
        """
        if source:
            loader_te = DataLoader(
                single_handler(X, Y, data_name=data_name, transform=self.args_s['transform'], return_id= True),
                shuffle=False, **self.args['loader_te_args'])
        else:
            loader_te = DataLoader(
                single_handler(X, Y, data_name=data_name, transform=self.args_t['transform'], return_id= True),
                shuffle=False, **self.args['loader_te_args'])


        self.fea.eval()
        self.clf.eval()

        probs = torch.zeros(len(Y),dtype=Y.dtype)

        with torch.no_grad():

            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                latent = self.fea(x)
                out, _ = self.clf(latent)
                pred = out.max(1)[1]
                probs[idxs] = pred.cpu()


        acc = 1.0 * (Y==probs).sum().item() / len(Y)

        return acc
