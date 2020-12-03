import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn

from torchvision import transforms
from office_dataloader import Target_labeled_dataset, BaseImageDataset , FileListDataset, BaseImageDataset_all, FileListDataset_all, test_loader, update_dataset, load_updated_set,query_and_target_data,  all_dataset_in_active_loop, query_dataset
from tqdm import tqdm
from torch.distributions import Categorical

import random
import collections
from collections import Counter

# setting gradient values
def set_requires_grad(model, requires_grad=True):
    """
    Used in training adversarial approach
    :param model:
    :param requires_grad:
    :return:
    """

    for param in model.parameters():
        param.requires_grad = requires_grad


# setting gradient penalty for sure the lipschitiz property
def gradient_penalty(critic, h_s, h_t):
    ''' Gradeitnt penalty for Wasserstein GAN'''
    alpha = torch.rand(h_s.size(0), 1).cuda()
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
    # interpolates.requires_grad_()
    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()

    return gradient_penalty

#X_s_tr, Y_s_tr, X_t_tr, Y_t_tr
class active_da_office():

    def __init__(self, X_s_tr, Y_s_tr, X_t_tr, Y_t_tr, 
                net_fea, net_clf, net_dis, net_dis_s_t,
                source_loader, target_loader,args):

        """

        :param X: instances from the target pool
        :param Y: real label of the target pool
        :param idx_lb: The idx of the label need to be queried target domain
        :param net_fea: feature extractor
        :param net_clf: classifier
        :param net_dis: Used for the query round that to distinguish whether the instances is far from that has been seen or not
        :param net_dis_s_t: Used to distinguish instances from source domain or target domain
        :param train_handler: generate a dataset in the training procedure, since training requires two datasets, the returning value
                                looks like a (index, x_dis1, y_dis1, x_dis2, y_dis2)
        :param test_handler: generate a dataset for the prediction, only requires one dataset
        :param args:
        """
        # Load the source and target dataset both from train and test


        #labeled_target_dataset = Target_labeled_dataset(target_idx_list = X_t_tr,
        #                                                target_label_list=Y_s_tr,
        #                                                idxs_lb=idxs_lb, transform = config['train_transform'])
        #self.X_s_tr = torch.tensor(X_s_tr)
        #self.Y_s_tr = torch.tensor(Y_s_tr)
        #self.X_t_tr = torch.tensor(X_t_tr)
        #self.Y_t_tr = torch.tensor(Y_t_tr)


        self.X_s_tr = X_s_tr
        self.Y_s_tr = Y_s_tr
        self.X_t_tr = X_t_tr
        self.Y_t_tr = Y_t_tr
        
        #self.query_dataloader = DataLoader(labeled_target_dataset, batch_size=16,shuffle = True)

        #self.query_loader = query_loader
        #self.test_handler = test_handler

        #self.X = X
        #self.Y = Y
        self.source_loader = source_loader
        self.target_loader = target_loader

        
        #self.idx_lb  = idx_lb
        self.net_fea = net_fea
        self.net_clf = net_clf
        self.net_dis = net_dis
        self.net_dis_s_t = net_dis_s_t

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.fea = net_fea.to(self.device)
        self.clf = net_clf.to(self.device)
        self.dis = net_dis.to(self.device)
        self.dis_s_t = net_dis_s_t.to(self.device)


        #self.train_handler = train_handler
        #self.test_handler  = test_handler
        self.args    = args
        #self.args_s  = args_s
        #self.args_t  = args_t
        #self.st_handler = st_handler


        self.num_class = self.args['num_class']


        self.selection = 10

        self.source_training_epoch = self.args['source_training_epoch']    # Define how many epochs to train on source labeled data  
        self.gp_param = self.args['gradient_penalty_pamtr']


        self.lr_fea_cls = self.args['lr_fea_cls']
        self.lr_fea_dis = self.args['lr_fea_dis']
        self.lr_dis = self.args['lr_dis']
        self.weight_decay = self.args['weight_decay']

        self.w_d_round = self.args['w_d_round'] # How many time wasserstein distance being computed in the critic round
        self.w_d_param = self.args['w_d_param'] # Weights for wasserstein distance in the classification round
        

        # Initialize the list for query
        self.n_pool  = len(Y_t_tr) # Unlabeled target pool
        
        self.num_query = self.args['num_query']
        self.idxs_lb = np.zeros(self.n_pool, dtype=bool) # Initialize the query list with all zeros
        
        self.active_round = self.args['active_round']
        self.weight_q = self.args['weight_of_query']
        # Generate the initial labeled pool 
        #idxs_lb = np.zeros(n_pool, dtype=bool)
        #idxs_tmp = np.arange(n_pool)
        #np.random.shuffle(idxs_tmp)
        #idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

    def random_active(self):
        '''
        This function will try to first randomly select some budget of instances in the target domain
        and train them under a semi-supervised DA setting
        Before start with this random active, we should finish the adversarail training first
        And this function directly union the source with the labeled target, which will have noise
        '''

        idxs_lb = np.zeros(self.n_pool, dtype=bool)
        idxs_tmp = np.arange(self.n_pool)
        
        np.random.shuffle(idxs_tmp)
        print('[We enter the active training stage]')
        self.idxs_lb[idxs_tmp[:self.num_query]] = True # Randomly choose serveral idx 

        updated = update_dataset( source_idx_list = self.X_s_tr,
                 source_label_list= self.Y_s_tr,
                 target_idx_list= self.X_t_tr,
                 target_label_list = self.Y_t_tr,
                 idxs_lb = self.idxs_lb)
        

        X_s_new, Y_s_new, X_t_new, Y_t_new = updated.return_new_source_new_target()
        X_q, Y_q = updated.load_query_data() # Load the query data and corresponding labels
        print('We finish the query process, the original source domain has size '+str(len(self.Y_s_tr)) +' current set size is '+str(len(Y_s_new)))


        new_source = load_updated_set(imgs = X_s_new, labels = Y_s_new,transform=self.args['train_transform'])
        new_target = load_updated_set(imgs = X_t_new, labels = Y_t_new,transform=self.args['train_transform'])

        # Here we define the sampler to balance each class
        new_source_classes = Y_s_new
        new_source_freq = Counter(new_source_classes)
        new_source_class_weight = {x : 1.0 / new_source_freq[x] if self.args['new_need_balance'] else 1.0 for x in new_source_freq}

        new_source_weights = [new_source_class_weight[x] for x in Y_s_new]
        new_source_sampler = WeightedRandomSampler(new_source_weights, len(Y_s_new))

        new_target_classes = Y_t_new
        new_target_freq = Counter(new_target_classes)
        new_target_class_weight = {x : 1.0 / new_target_freq[x] if self.args['new_need_balance'] else 1.0 for x in new_target_freq}

        new_target_weights = [new_target_class_weight[x] for x in Y_t_new]
        new_target_sampler = WeightedRandomSampler(new_target_weights, len(Y_t_new))

        #Then we shall have the loaders

        new_source_loader = DataLoader(new_source, batch_size=16,sampler = new_source_sampler, drop_last=True)
        new_target_loader = DataLoader(new_target, batch_size=16,sampler = new_target_sampler, drop_last=True)
        # Then we shall use the unsupervised DA (actually semi since we already have the labeled target) training on the new datasets
        #all_data = all_dataset_in_active_loop(X_s= X_s_new, Y_s = Y_s_new, X_t = X_t_new, Y_t= Y_t_new, X_q=X_q, Y_q=Y_q,
        #                transform=self.args['train_transform'])

        #loaderall = DataLoader(all_data, batch_size=16, sampler = new_target_sampler , drop_last=True)
        
        #epoch_id += 1
        for epoch in range(self.active_round):
            total_acc_t = 0
            total_acc_s = 0
            #for (x_s, y_s), (x_t, y_t) in tqdm(batches, leave=False, total=n_batches-2):
            # During the adversarial training
            #batches = zip(self.source_loader, self.target_loader)
            #n_batches = min(len(self.source_loader), len(self.target_loader))
            # Here we follow the WDGRL method for the domain level critic

            self.fea.train()
            self.clf.train()
            # Here we use same discrimator
            #self.dis.train() # dis used for instance level critic
            self.dis_s_t.train() # dis_s_t is used for domain level critic

            opt_fea = optim.Adam(self.fea.parameters() , lr=self.lr_fea_dis)
            opt_clf = optim.Adam(self.clf.parameters(), lr=self.lr_fea_dis,weight_decay=self.weight_decay)
            #opt_dis = optim.Adam(self.dis.parameters() , lr=self.lr_dis)
            opt_dis_s_t = optim.Adam(self.dis_s_t.parameters(), lr=self.lr_dis)
            n_batches = min(len(new_source_loader), len(new_target_loader))
            iters = tqdm(zip(new_source_loader, new_target_loader), desc=f'epoch {epoch} ', total=n_batches)

        
            #all_data = all_dataset_in_active_loop(X_s= X_s_new, Y_s = Y_s_new, X_t = X_t_new, Y_t= Y_t_new, X_q=X_q, Y_q=Y_q,
            #                transform=self.args['train_transform'])

            #loaderall = DataLoader(all_data, batch_size=16, drop_last=True)

            #iters = tqdm(loaderall, desc=f'epoch {epoch} ', total=len(loaderall))
            #n_batches = 0
            #for x_s,y_s,x_t,y_t,x_q,y_q in loaderall:
            #    n_batches+=1
            

            for i, ((x_s, y_s), (x_t, y_t)) in enumerate(iters):
            #for x_s,y_s, x_t,y_t, x_q, y_q in loaderall:
                n_batches+=1

                # We adopt this coefficient trade-off from DANN setting
                #p = float(i + (epoch -self.source_training_epoch)* n_batches) / (epoch -self.source_training_epoch) / n_batches
                #alpha = 2. / (1. + np.exp(-10 * p)) - 1

                opt_fea.zero_grad()
                opt_clf.zero_grad()
                #opt_dis.zero_grad()
                opt_dis_s_t.zero_grad()

                x_s, y_s, x_t, y_t = x_s.cuda(), y_s.cuda(), x_t.cuda(), y_t.cuda()
                # train the domain critic first

                set_requires_grad(self.fea, requires_grad=False)
                set_requires_grad(self.clf, requires_grad=False)
                set_requires_grad(self.dis_s_t, requires_grad=True)
                #set_requires_grad(self.dis, requires_grad = True)


                z_s = self.fea(x_s)
                z_t = self.fea(x_t)
                print('z_s is ', z_s)


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

                            

                fc1_s = self.fea(x_s)
                fc1_s, feature_source, fc2_s, predict_prob_source = self.clf(fc1_s)
                # feature_source could be used for TSNE


                # Compute CrossEntropy loss for classification

                ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, y_s)
                cls_loss = torch.mean(ce, dim=0, keepdim=True)

                #fc1_q = self.fea(x_q)
                #fc1_q, feature_q, fc2_q, predict_prob_q = self.clf(fc1_q)
                #ce_q = nn.CrossEntropyLoss(reduction='none')(fc2_q, y_q)
                #cls_loss_q = torch.mean(ce_q, dim=0, keepdim=True)
                
                ##print('loss_source', loss_source)
                            
                            #total_loss += cls_loss

                            # Here we don't use fc1_s but to compute source_z again to avoid potentail variable gradient conflict issue
                source_z = self.fea(x_s)
                target_z = self.fea(x_t)

                # Then compute the wasserstein distance between source and target
                wassertein_source_and_target = self.dis_s_t(source_z).mean() - self.dis_s_t(target_z).mean()
                    # Then compute the gradient penalty
                gp_s_t = gradient_penalty(self.dis_s_t, target_z, source_z)
                            
                    # Alpha is a hyper-prmtr for fine-tuning. Typically when network got deeper, the higher 
                    #loss = cls_loss + (-wassertein_source_and_target - self.alpha * gp_s_t * 2)
                #loss = cls_loss+ self.weight_q * cls_loss_q + self.w_d_param*(-wassertein_source_and_target + self.gp_param * gp_s_t * 2)
                    #loss = cls_loss + (alpha * wassertein_source_and_target - alpha * gp_s_t * 2)
                loss = cls_loss+ self.w_d_param*(-wassertein_source_and_target + self.gp_param * gp_s_t * 2)

                loss.backward()
                opt_fea.step()
                opt_clf.step()                   
                        

                    # Then we shall test the test results on the target domain
                self.fea.eval()
                self.clf.eval()

                #P = torch.zeros(len(Y), dtype=torch.long)
                            
                with torch.no_grad():
                    latent_s = self.fea(x_s)
                    latent = self.fea(x_t)

                _,_,_, out_s1 = self.clf(latent_s)
                _,_,_, out1 = self.clf(latent)
                #print('n batches is', n_batches)

                            
                total_acc_t    += (out1.max(1)[1] == y_t).float().mean().item()
                total_acc_s  += (out_s1.max(1)[1] == y_s).float().mean().item()
                                
            acc_t = 100.0* total_acc_t/n_batches
            acc_s = 100.0 * total_acc_s/n_batches
            print('total batch num is', n_batches)
            print('==========Inner epoch {:d} ========'.format(epoch))
            print('     Mean acc on target domain is ', acc_t)
            print('     Mean acc on source domain is ', acc_s)


    def random_active_entropy(self):
        '''
        This function will try to first randomly select some budget of instances in the target domain
        and train them under a semi-supervised DA setting
        Before start with this random active, we should finish the adversarail training first
        '''

        idxs_lb = np.zeros(self.n_pool, dtype=bool)
        idxs_tmp = np.arange(self.n_pool)
        
        np.random.shuffle(idxs_tmp)
        print('[We enter the active training stage]')
        self.idxs_lb[idxs_tmp[:self.num_query]] = True # Randomly choose serveral idx 

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

        for step in range(2000):
            best_acc_s = 0
            best_acc_t = 0

            
            opt_fea = optim.Adam(self.fea.parameters() , lr=1e-5)
            opt_clf = optim.Adam(self.clf.parameters(), lr=1e-5,weight_decay = self.weight_decay)
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


    def train_source(self, source_training_epoch):
        
        #self.source_loader
        #self.target_loader



        print('[Training on the source dataset]')
        for epoch in range(source_training_epoch):

            # setting the training mode in the beginning of EACH epoch
            # (since we need to compute the training accuracy during the epoch, optional)

            self.fea.train()
            self.clf.train()
            #self.dis.train()
            #self.dis_s_t.train()
            opt_fea = optim.Adam(self.fea.parameters() , lr=1e-5)
            opt_clf = optim.Adam(self.clf.parameters(), lr=1e-5,weight_decay =self.weight_decay)
            
            total_acc = 0
            total_acc_source = 0
            total_loss = 0

            #opt_fea = optim.SGD(self.fea.parameters(),**self.args['optimizer_args'])
            #opt_clf = optim.SGD(self.clf.parameters(),**self.args['optimizer_args'])
            #opt_dis = optim.SGD(self.dis.parameters(),**self.args['optimizer_args'])
            #opt_dis_s_t = optim.SGD(self.dis.parameters(),**self.args['optimizer_args']

            batches = zip(self.source_loader, self.target_loader)
            n_batches = 0
            #criterion = nn.CrossEntropyLoss()

            for (x_s, y_s), (x_t, y_t) in tqdm(batches, leave=False, total= min(len(self.source_loader), len(self.target_loader))):
                # Here we train the network only on source data.
                # We take the target data only for test, that means the row source only
                # We shall also use it as target only. It will be the same strategy
                n_batches+=1
                

                x_s, y_s, x_t, y_t = x_s.cuda(), y_s.cuda(), x_t.cuda(), y_t.cuda()

                source_feature = self.fea(x_s)
                fc1_s, feature_source, fc2_s, predict_prob_source = self.clf(source_feature)

                
                #pred    = source_pred.max(1)[1]
                #pred    = source_pred.max(1)[1]
                opt_fea.zero_grad()
                opt_clf.zero_grad()

                #print('pred size', pred.size())
                #print('y_s size', y_s.size())
                #print('pred is', pred)
                #print('y_s is', y_s)
                #print('fc_2 is', fc2_s.size())
                #loss_source = criterion(source_pred, y_s)
                #loss_source = F.cross_entropy(fc2_s, y_s)
                #loss_source = torch.mean(ce, dim=0, keepdim=True)
                ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, y_s)
                loss_source = torch.mean(ce, dim=0, keepdim=True)
                #print('loss_source', loss_source)
                
                total_loss += loss_source
                #loss_source = F.cross_entropy(pred,y_s)
                #loss_source = nn.cross_entropy(pred, y_s)

                loss_source.backward()
                opt_fea.step()
                opt_clf.step()


                ########### Predict source and target acc
                self.fea.eval()
                self.clf.eval()

                #P = torch.zeros(len(Y), dtype=torch.long)
               
                latent = self.fea(x_t)
                _,_,_, out = self.clf(latent)
                
                latent_s = self.fea(x_s)
                _,_,_,out_s  = self.clf(latent_s)
                #print('out is ', out.size())
                #print('out_source is ',out_s.size())
                total_acc_source += (out_s.max(1)[1] == y_s).float().mean().item()
                total_acc += (out.max(1)[1] == y_t).float().mean().item()
        


            

            #total_acc    += (out.max(1)[1] == y_t).float().mean().item()
            #total_acc_source += (out_s.max(1)[1] == y_s).float().mean().item()
            #_, predict = torch.max(out, 1)
                # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
            #total_acc += torch.sum(torch.squeeze(predict).float() == y_t) / float(y_t.size()[0])
                #
            #_, predict_s = torch.max(out_s, 1)

            
            acc = 100*total_acc/n_batches
            acc_s = 100 * total_acc_source/n_batches
            total_loss = total_loss/n_batches


            print('Round', epoch)
            print('[Source only], test acc on the target domain is', acc)
            print('[Source only], test acc on the source domain is', acc_s)
            print('[Source only], total loss is', total_loss)

        #torch.save({
        #    'fea_net': self.fea(),
        #    'clf_net': self.clf(),
        #    'fea_state_dict': self.fea.state_dict(),
        #    'clf_state_dict': self.clf.state_dict(),
        #    'opt_fea_state_dict': opt_fea.state_dict(),
        #    'opt_clf_state_dict': opt_clf.state_dict(),
        #    }, './basemodel/fea_clf_a2d.pt')
        #print('=========     Pretrained source only model saved!     =========')


    def adversarial_train(self, total_epoch):

        '''
        Train on the source (labeled) and target domain (unlabeled)
        '''

        print('[Training on the source dataset]')
        for epoch in range(total_epoch):

            
            # setting the training mode in the beginning of EACH epoch
            # (since we need to compute the training accuracy during the epoch, optional)

            self.fea.train()
            self.clf.train()
            self.dis.train()
            self.dis_s_t.train()

            
            total_acc = 0
            total_acc_source = 0
            total_loss = 0
            total_acc_s = 0
            total_acc_t = 0

            batches = zip(self.source_loader, self.target_loader)
            
            
            #criterion = nn.CrossEntropyLoss()
            if epoch <= self.source_training_epoch:
                print('[current step: Train on source only] Epoch ', epoch)        
                # The learning rate at source training epochs can be different with the ones in adversarail epochs

                opt_fea = optim.Adam(self.fea.parameters() , lr=self.lr_fea_cls)
                opt_clf = optim.Adam(self.clf.parameters(), lr=self.lr_fea_cls,weight_decay=self.weight_decay)
                n_batches = 0

                for (x_s, y_s), (x_t, y_t) in tqdm(batches, leave=False, total=min(len(self.source_loader), len(self.target_loader))):
                    # Here we train the network only on source data for a good initialization

                    x_s, y_s, x_t, y_t = x_s.cuda(), y_s.cuda(), x_t.cuda(), y_t.cuda()
                    n_batches+=1

                    source_feature = self.fea(x_s)
                    fc1_s, feature_source, fc2_s, predict_prob_source = self.clf(source_feature)


                    opt_fea.zero_grad()
                    opt_clf.zero_grad()


                    ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, y_s)
                    loss_source = torch.mean(ce, dim=0, keepdim=True)
                    #print('loss_source', loss_source)
                    
                    total_loss += loss_source
                    #loss_source = F.cross_entropy(pred,y_s)
                    #loss_source = nn.cross_entropy(pred, y_s)

                    loss_source.backward()
                    opt_fea.step()
                    opt_clf.step()


                    ########### Predict source and target acc
                    self.fea.eval()
                    self.clf.eval()
                
                    latent = self.fea(x_t)
                    _,_,_, out = self.clf(latent)
                    
                    latent_s = self.fea(x_s)
                    _,_,_,out_s  = self.clf(latent_s)
                    #print('out is ', out.size())
                    #print('out_source is ',out_s.size())
                    total_acc_source += (out_s.max(1)[1] == y_s).float().mean().item()
                    total_acc += (out.max(1)[1] == y_t).float().mean().item()


                acc = 100*total_acc/n_batches
                acc_s = 100 * total_acc_source/n_batches
                total_loss = total_loss/n_batches


                print('Round', epoch)
                print('[Source only], test acc on the target domain is', acc)
                print('[Source only], test acc on the source domain is', acc_s)
                print('[Source only], total loss is', total_loss)
            
            else: # Epoch greater than the source training epochs, then, we can start the adversarial training

                #batches = zip(self.source_loader, self.target_loader)
                n_batches = 0
                # Here we follow the WDGRL method for the domain level critic

                self.fea.train()
                self.clf.train()
                #self.dis.train() # dis used for instance level critic
                self.dis_s_t.train() # dis_s_t is used for domain level critic

                opt_fea = optim.Adam(self.fea.parameters() , lr=self.lr_fea_dis)
                opt_clf = optim.Adam(self.clf.parameters(), lr=self.lr_fea_dis,weight_decay=self.weight_decay)
                #opt_dis = optim.Adam(self.dis.parameters() , lr=self.lr_dis)
                opt_dis_s_t = optim.Adam(self.dis_s_t.parameters(), lr=self.lr_dis)

                #for (x_s, y_s), (x_t, y_t) in tqdm(batches, leave=False, total=n_batches-2):
                    # During the adversarial training

                iters = tqdm(zip(self.source_loader, self.target_loader), desc=f'epoch {epoch} ', total=min(len(self.source_loader), len(self.target_loader)))
                #epoch_id += 1

                for i, ((x_s, y_s), (x_t, y_t)) in enumerate(iters):

                    # We adopt this coefficient trade-off from DANN setting
                    #p = float(i + (epoch -self.source_training_epoch)* n_batches) / (epoch -self.source_training_epoch) / n_batches
                    #alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    n_batches+=1

                    opt_fea.zero_grad()
                    opt_clf.zero_grad()
                    #opt_dis.zero_grad()
                    opt_dis_s_t.zero_grad()


                    if len(y_s) == len(y_t):

                        x_s, y_s, x_t, y_t = x_s.cuda(), y_s.cuda(), x_t.cuda(), y_t.cuda()
            
                        # train the domain critic first

                        set_requires_grad(self.fea, requires_grad=False)
                        set_requires_grad(self.clf, requires_grad=False)
                        set_requires_grad(self.dis_s_t, requires_grad=True)
                        #set_requires_grad(self.dis, requires_grad = True)


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

                        

                        fc1_s = self.fea(x_s)
                        fc1_s, feature_source, fc2_s, predict_prob_source = self.clf(fc1_s)
                        # feature_source could be used for TSNE


                        # Compute CrossEntropy loss for classification

                        ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, y_s)
                        cls_loss = torch.mean(ce, dim=0, keepdim=True)
                        ##print('loss_source', loss_source)
                        
                        #total_loss += cls_loss

                        # Here we don't use fc1_s but to compute source_z again to avoid potentail variable gradient conflict issue
                        source_z = self.fea(x_s)
                        target_z = self.fea(x_t)

                        # Then compute the wasserstein distance between source and target
                        wassertein_source_and_target = self.dis_s_t(source_z).mean() - self.dis_s_t(target_z).mean()
                        # Then compute the gradient penalty
                        gp_s_t = gradient_penalty(self.dis_s_t, target_z, source_z)
                        
                        # Alpha is a hyper-prmtr for fine-tuning. Typically when network got deeper, the higher 
                        #loss = cls_loss + (-wassertein_source_and_target - self.alpha * gp_s_t * 2)
                        loss = cls_loss + self.w_d_param*(-wassertein_source_and_target + self.gp_param * gp_s_t * 2)
                        #loss = cls_loss + (alpha * wassertein_source_and_target - alpha * gp_s_t * 2)

                        loss.backward()
                        opt_fea.step()
                        opt_clf.step()                   
                    

                        # Then we shall test the test results on the target domain
                        self.fea.eval()
                        self.clf.eval()

                      #P = torch.zeros(len(Y), dtype=torch.long)
                        
                        with torch.no_grad():
                            latent_s = self.fea(x_s)
                            latent = self.fea(x_t)

                            _,_,_, out_s1 = self.clf(latent_s)
                            _,_,_, out1 = self.clf(latent)

                        
                        total_acc_t    += (out1.max(1)[1] == y_t).float().mean().item()
                        total_acc_s  += (out_s1.max(1)[1] == y_s).float().mean().item()
                            
                acc_t = 100.0* total_acc_t/n_batches
                acc_s = 100.0 * total_acc_s/n_batches
                print('total batch num is', n_batches)
                print('==========Inner epoch {:d} ========'.format(epoch))
                print('     Mean acc on target domain is ', acc_t)
                print('     Mean acc on source domain is ', acc_s)
                # print('Training Loss {:.3f}'.format(Total_loss))
                # print('Training accuracy {:.3f}'.format(acc*100))





    def train(self, total_epoch, n_iter):
    #def train(self, alpha_s, alpha, total_epoch, n_iter):

        """
        Only training samples with labeled and unlabeled data-set
        alpha is the trade-off between the empirical loss and error, the more interaction, the smaller \alpha
        alpha_s is used for source domain training
        :return:
        """

        print("[====training started====]")
        # n_epoch = self.args['n_epoch']
        n_epoch = total_epoch




        self.fea = self.net_fea.to(self.device)
        self.clf = self.net_clf.to(self.device)
        #self.dis = self.net_dis.to(self.device)
        self.dis_s_t = self.net_dis_s_t.to(self.device)



        # setting  optimizers

        opt_fea = optim.SGD(self.fea.parameters(),**self.args['optimizer_args'])
        opt_clf = optim.SGD(self.clf.parameters(),**self.args['optimizer_args'])
        opt_dis = optim.SGD(self.dis.parameters(),**self.args['optimizer_args'])
        opt_dis_s_t = optim.SGD(self.dis.parameters(),**self.args['optimizer_args'])

        # setting idx_lb and idx_ulb
        idx_lb_train = np.arange(self.n_pool)[self.idx_lb]
        idx_ulb_train = np.arange(self.n_pool)[~self.idx_lb]

        # computing the unbalancing ratio, a value betwwen [0,1], generally 0.1 - 0.5
        gamma_ratio = len(idx_lb_train)/len(idx_ulb_train)
        # gamma_ratio = 1

        # Data-loading (Redundant Trick)
        '''

        loader_tr = DataLoader(self.train_handler(self.X[idx_lb_train],self.Y[idx_lb_train],self.X[idx_ulb_train],self.Y[idx_ulb_train],
                                            transform = self.args['transform']), shuffle= True, **self.args['loader_tr_args'])
        

        loader_tr = DataLoader(self.train_handler(self.source_name, self.target_name, self.X[idx_lb_train], self.Y[idx_lb_train], self.X[idx_ulb_train], self.Y[idx_ulb_train],
                                            transform_source = self.args_s['transform'], transform_target= self.args_t['transform']), 
                                            shuffle= True, **self.args['loader_tr_args'])
        '''

        #loder_all = DataLoader(self.st_handler(self.source_name, self.target_name, self.X_s_tr, self.Y_s_tr , self.X_t_tr[idx_lb_train], self.Y_t_tr[idx_lb_train], self.X_t_tr[idx_ulb_train], self.Y_t_tr[idx_ulb_train],
        #                                    transform_source = self.args_s['transform'], transform_target= self.args_t['transform']), 
        #                                    shuffle= True, **self.args['loader_tr_args'])
        
        #loader_tr = self.train_handler
        #loader_te = self.test_handler



        # training like 10 epochs
        i = 0

        for epoch in range(n_epoch):

            # setting the training mode in the beginning of EACH epoch
            # (since we need to compute the training accuracy during the epoch, optional)

            self.fea.train()
            self.clf.train()
            self.dis.train()
            self.dis_s_t.train()
            total_acc = 0
            total_acc_source = 0


            batches = zip(source_loader_tr, target_loader_tr)
            n_batches = min(len(source_loader_tr), len(target_loader_tr))

            for (x_s, y_s), (x_t, y_t) in tqdm(batches, leave=False, total=n_batches):
                # Target label is always unkonwn during the training time
                # Here we load it to tackle the training process

                x_s, y_s, x_t = x_s.cuda(), y_s.cuda(), x_t.cuda()
                
                # Implement the alpha trade-off same with the DANN setting

                #p = float(i + epoch * n_batches) / n_epoch / n_batches    
                #alpha = 2. / (1. + np.exp(-10 * p)) - 1

                set_requires_grad(self.fea,requires_grad=True)
                set_requires_grad(self.clf,requires_grad=True)
                set_requires_grad(self.dis,requires_grad=False)
                set_requires_grad(self.dis_s_t,requires_grad=False)

                # Firstly train the feature extractor and predictor on Source domain, use z to mention the latent feature space
                source_z = self.fea(x_s)
                
                _,_,_,pred_source = self.clf(source_z)
                    
                opt_fea.zero_grad()
                opt_clf.zero_grad()
                loss_source = F.cross_entropy(pred_source,y_s)
                target_z = self.fea(x_t)

                wassertein_source_and_target = self.dis_s_t(source_z).mean() - self.dis(target_z).mean()


                #with torch.no_grad():

                #    source_z = self.fea(x_s)
                #    target_z = self.fea(x_t)

                gp_s_t = gradient_penalty(self.dis_s_t, target_z, source_z)

                loss = loss_source + alpha * wassertein_source_and_target - alpha * gp_s_t * 2

                loss.backward()
                opt_fea.step()
                opt_clf.step()

                    # Then train the discrinator 

                set_requires_grad(self.fea, requires_grad=False)
                set_requires_grad(self.clf, requires_grad=False)
                set_requires_grad(self.dis_s_t, requires_grad=True)


                #with torch.no_grad():

                #    source_z = self.fea(x_s)
                #    target_z = self.fea(x_t)


                for _ in range(1):

                    # gradient ascent for multiple times like GANS training

                    gp_s_t = gradient_penalty(self.dis_s_t, target_z, source_z)

                    wassertein_source_and_target = self.dis_s_t(source_z).mean() - self.dis(target_z).mean()

                    dis_s_t_loss = -1 * alpha * wassertein_source_and_target + alpha * gp_s_t * 2

                    opt_dis_s_t.zero_grad()
                    dis_s_t_loss.backward()
                    opt_dis_s_t.step()
                
                # Then we shall test the test results on the target domain
                self.fea.eval()
                self.clf.eval()

#                P = torch.zeros(len(Y), dtype=torch.long)
                
                with torch.no_grad():
                    latent = self.fea(x_t)
                    _,_,_, out = self.clf(latent)
                
                total_acc    += (out.max(1)[1] == y_t).float().mean().item()
            
                        
            acc = 100.0* total_acc/total_batchs
            print('total batch num is', batch_num)
            print('==========Inner epoch {:d} ========'.format(epoch))
            print('     Mean acc on target domain is ', acc)
            # print('Training Loss {:.3f}'.format(Total_loss))
            # print('Training accuracy {:.3f}'.format(acc*100))
        

###################################################################################################
    def update(self, idx_lb):

        self.idxs_lb = idx_lb
        
        #new_labeled_target_dataset = Target_labeled_dataset(target_idx_list = self.X_t_tr,
        #                                                target_label_list= self.Y_s_tr,
        #                                                idxs_lb=idx_lb, transform = config['train_transform'])
        #self.query_dataloader = DataLoader(labeled_target_dataset, batch_size=16,shuffle = True)



    def predict(self,X,Y):

        #loader_te = DataLoader(self.test_handler(X, Y, target_name = self.target_name, transform=self.args_t['transform']),
        #                       shuffle=False, **self.args['loader_te_args'])

        loader_te = DataLoader(test_loader( dataset_list = X, label_list = Y, transform = self.args['train_transform'],return_id = True), 
                                batch_size = 16, shuffle = True )
                                

        self.fea.eval()
        self.clf.eval()

        P = torch.zeros(len(Y), dtype=torch.long)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                latent  = self.fea(x)
                _,_,_, out   = self.clf(latent)
                _,  pred   = torch.max(out,1)  #out.max(1)[1]
                P[idxs] = pred.cpu()

        return P


    def predict_prob(self,X,Y):

        """
        prediction output score probability
        :param X:
        :param Y: NEVER USE the Y information for direct prediction
        :return:
        """

        loader_te = DataLoader(test_loader( dataset_list = X, label_list = Y, 
                                transform = self.args['train_transform'],return_id = True), 
                                batch_size = 16, shuffle = True )
        self.fea.eval()
        self.clf.eval()

        probs = torch.zeros([len(Y), self.num_class])
        with torch.no_grad():

            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                latent = self.fea(x)
                _,_,out,_  = self.clf(latent)
                prob = F.softmax(out, dim=-1)
                #print(prob)
                probs[idxs] = prob.cpu()

        return probs


    def pred_dis_score(self,X,Y):

        """
        prediction discrimnator score
        :param X:
        :param Y:  FOR numerical simplification, NEVER USE Y information for prediction
        :return:

        """

        loader_te = DataLoader(test_loader( dataset_list = X, label_list = Y, transform = self.args['train_transform'],return_id = True), 
                                batch_size = 16, shuffle = True )

        self.fea.eval()
        self.dis_s_t.eval()

        scores = torch.zeros(len(Y))

        with torch.no_grad():
            for x, y, idxs in loader_te:

                x, y = x.to(self.device), y.to(self.device)
                latent = self.fea(x)
                out = self.dis_s_t(latent).cpu()
                scores[idxs] = out.view(-1)

        print('scores size ', scores.size())
        return scores


    def single_worst(self, probas):

        """
        The single worst will return the max_{k} -log(proba[k]) for each sample

        :param probas:
        :return:  # unlabeled \times 1 (tensor float)

        """

        value,_ = torch.max(-1*torch.log(probas),1)

        return value


    def L2_upper(self, probas):

        """
        Return the /|-log(proba)/|_2

        :param probas:
        :return:  # unlabeled \times 1 (float tensor)

        """

        value = torch.norm(torch.log(probas),dim=1)

        return value


    def L1_upper(self, probas):

        """
        Return the /|-log(proba)/|_1
        :param probas:
        :return:  # unlabeled \times 1

        """
        value = torch.sum(-1*torch.log(probas),dim=1)

        return value
    
    
    def entropy_uncertainty (self, probas):
        """
        Compute the entropy uncertainty score using the same metric from Jun Wen IJCAI 19 (Bayes UDA)
        """


        value = Categorical(probs = probas).entropy()
        print('value size', value.size())

        return value

