import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import logging

from model.builder import AEBuilder
from model.loss import ScaledL2Trans, ScaledL2Recons
from model.initialization import LSUVinit
from utils.conf import Configuration
from utils.data import load_UCR, embedData, load_HKAIR_50


class TS2Vec:
    def __init__(self, conf: Configuration) -> None:
        ''' Initialize a TS2Vec Model '''
        super().__init__()
        self.__conf = conf
        self.has_setup = False
        self.epoch = 0

        self.device = conf.getHP('device')
        self.max_epoch = conf.getHP('num_epoch')

    def setup(self) -> None:
        self.has_setup = True

        torch.manual_seed(1998)

        if self.device == 'cuda':
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.manual_seed_all(1998)
            else:
                raise ValueError('cuda is not available.')

        
        logging.basicConfig(filename=self.__conf.getHP('log_path'), 
                            filemode='a+', 
                            format='%(asctime)s,%(msecs)d %(levelname).3s [%(filename)s:%(lineno)d] %(message)s', 
                            level=logging.DEBUG,
                            datefmt='%m/%d/%Y:%I:%M:%S')
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        batch_size = self.__conf.getHP('batch_size')
        
        if self.__conf.getHP('name') == 'hkair':
            train_samples, _ = load_HKAIR_50('pollutant', self.__conf.getHP('dataset_name')) # TODO: params
            val_samples = np.empty([50, 61])
            train_samples = torch.Tensor(train_samples).view([-1, 1, self.__conf.getHP('dim_series')]).to(self.device)
            val_samples = torch.Tensor(val_samples).view([-1, 1, self.__conf.getHP('dim_series')]).to(self.device)
        else:
            train_samples, train_labels, val_samples, val_labels = load_UCR(self.__conf.getHP('dataset_name'))
            train_samples = torch.tensor(train_samples).view([-1, 1, self.__conf.getHP('dim_series')]).to(self.device)
            val_samples = torch.tensor(val_samples).view([-1, 1, self.__conf.getHP('dim_series')]).to(self.device)

        self.train_db_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True)
        self.train_query_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True)
        self.val_db_loader = DataLoader(val_samples, batch_size=batch_size, shuffle=True)
        self.val_query_loader = DataLoader(val_samples, batch_size=batch_size, shuffle=True)

        dim_series = self.__conf.getHP('dim_series')
        dim_embedding = self.__conf.getHP('dim_embedding')
        
        self.recons_weight = self.__conf.getHP('recons_weight')

        self.trans_loss = ScaledL2Trans(dim_series, dim_embedding, to_scale=True).to(self.device)
        self.recons_reg = ScaledL2Recons(dim_series, to_scale=True).to(self.device)

        self.model = AEBuilder(self.__conf).to('cuda')

        self.optimizer = self.__getOptimizer()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def __init_model(self, model: nn.Module, samples: torch.Tensor = None) -> nn.Module:
        return LSUVinit(model, samples[torch.randperm(samples.shape[0])][: self.__conf.getHP('lsuv_size')], 
                            needed_mean=self.__conf.getHP('lsuv_mean'), needed_std=self.__conf.getHP('lsuv_std'), 
                            std_tol=self.__conf.getHP('lsuv_std_tol'), max_attempts=self.__conf.getHP('lsuv_maxiter'), 
                            do_orthonorm=self.__conf.getHP('lsuv_ortho'))

        
    def __getOptimizer(self) -> optim.Optimizer:
        if self.__conf.getHP('lr_mode') == 'fix':
            initial_lr = self.__conf.getHP('lr_cons')
        else:
            initial_lr = self.__conf.getHP('lr_max')

        
        if self.__conf.getHP('wd_mode') == 'fix':
            initial_wd = self.__conf.getHP('wd_cons')
        else:
            initial_wd = self.__conf.getHP('wd_min')
        
        momentum = self.__conf.getHP('momentum')
        return optim.SGD(self.model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=initial_wd)
    
    def __train(self) -> None:
        recons_errors = []
        orth_terms = []
        trans_errors = []
        total_losses = []

        model_name = self.__conf.getHP('model_name')

        if model_name == 'seanet':
            for db_batch, query_batch in zip(self.train_db_loader, self.train_query_loader):
                self.optimizer.zero_grad()

                with torch.no_grad():
                    query_embedding = self.model.encode(query_batch).detach()
                    query_batch = query_batch.detach()
                
                db_embedding = self.model.encode(db_batch)
                
                db_reconstructed = self.model.decode(db_embedding)
                recons_term = self.recons_weight * self.recons_reg(db_batch, db_reconstructed)

                trans_error = self.trans_loss(db_batch, query_batch, db_embedding, query_embedding)
                orth_term = self.__orth_reg()

                loss = trans_error + recons_term + orth_term

                loss.backward()
                self.optimizer.step()

                total_losses.append(loss.detach().item())
                recons_errors.append(recons_term.detach().item())
                orth_terms.append(orth_term.detach().item())
                trans_errors.append(trans_error.detach().item())
            
            if self.epoch % 10 == 0:
                print('Epoch: {}/{}, Loss: {:.4f}'.format(self.epoch, self.max_epoch, loss.detach().item()))
            self.logger.info('e{:d} loss = {:4f} recons = {:.4f} orth = {:.4f} trans = {:.4f}'.format(self.epoch, np.mean(total_losses), np.mean(recons_errors), np.mean(orth_terms), np.mean(trans_errors)))
        
        elif model_name == 'tsrnet':
            for db_batch, query_batch in zip(self.train_db_loader, self.train_query_loader):
                self.optimizer.zero_grad()

                with torch.no_grad():
                    query_recons1, query_recons2, query_embedding1, query_new_embedding = self.model.forward(query_batch)
                    query_batch = query_batch

                db_recons1, db_recons2, db_embedding1, db_new_embedding = self.model.forward(db_batch)

                recons_term = self.recons_weight * self.recons_reg(db_batch, db_recons1) + self.recons_weight * self.recons_reg(db_batch, db_recons2)
                trans_error = self.trans_loss(db_batch, query_batch, db_embedding1, query_embedding1) + self.trans_loss(db_batch, query_batch, db_new_embedding, query_new_embedding)
                orth_term = self.__orth_reg()

                loss = trans_error + recons_term + orth_term

                loss.backward()
                self.optimizer.step()

                total_losses.append(loss.detach().item())
                recons_errors.append(recons_term.detach().item())
                orth_terms.append(orth_term.detach().item())
                trans_errors.append(trans_error.detach().item())
                
            if self.epoch % 10 == 0:
               print('Epoch: {}/{}, Loss: {:.4f}'.format(self.epoch, self.max_epoch, loss.detach().item()))
            # print('Epoch: {}/{}, Loss: {:.4f}'.format(self.epoch, self.max_epoch, loss.detach().item()))
            self.logger.info('e{:d} loss = {:4f} recons = {:.4f} orth = {:.4f} trans = {:.4f}'.format(self.epoch, np.mean(total_losses), np.mean(recons_errors), np.mean(orth_terms), np.mean(trans_errors)))
        

    def __validate(self) -> None:
        trans_errors = []

        with torch.no_grad():
            for db_batch, query_batch in zip(self.val_db_loader, self.val_query_loader): 
                db_embedding = self.model.encode(db_batch)
                query_embedding = self.model.encode(query_batch)
                
                trans_error = self.trans_loss(db_batch, query_batch, db_embedding, query_embedding)
                trans_errors.append(trans_error.detach().item())                

        self.logger.info('v{:d} trans = {:.4f}'.format(self.epoch, np.mean(trans_errors)))

    def run(self) -> None:
        if not self.has_setup:
            self.setup()

        while self.epoch < self.max_epoch:
            self.__adjust_lr()
            self.__adjust_wd()

            self.epoch += 1

            self.__train()
            self.__validate()
        
        torch.save(self.model, self.__conf.getHP('model_path'))
        
        if self.__conf.getHP('is_embed'):
            embedData(self.__conf.getHP('name'), self.model, self.__conf.getHP('model_name'), self.__conf.getHP('dataset_name'), self.__conf.getHP('db_embedding_path'),  self.__conf.getHP('data_size'), self.__conf.getHP('batch_size'), self.__conf.getHP('dim_series'))            
        

    def __orth_reg(self) -> torch.Tensor:
        return torch.zeros(1).to(self.device)

    def __adjust_lr(self) -> None:
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
            break
        
        new_lr = current_lr
        if self.__conf.getHP('lr_mode') == 'linear':
            lr_max = self.__conf.getHP('lr_max')
            lr_min = self.__conf.getHP('lr_min')

            new_lr = lr_max - self.epoch * (lr_max - lr_min) / self.max_epoch
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def __adjust_wd(self) -> None:
        for param_group in self.optimizer.param_groups:
            current_wd = param_group['weight_decay']
            break

        new_wd = current_wd
        if self.__conf.getHP('wd_mode') == 'linear':
            wd_max = self.__conf.getHP('wd_max')
            wd_min = self.__conf.getHP('wd_min')

            new_wd = wd_min + self.epoch * (wd_max - wd_min) / self.max_epoch

        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = new_wd
    
        