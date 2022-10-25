from os import environ
# Select GPU 0 only
environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
environ['CUDA_VISIBLE_DEVICES']='0'
environ['MKL_THREADING_LAYER']='GNU'

import numpy as np
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pickle

class MLPDiscriminator:
    def __init__(self, env_spec, expert_paths, obs_dim=None, learn_rate=1e-3, 
                 reg_coef=0.0, batch_size=64, epochs=5, use_gpu=False):
        #self.n = obs_dim if obs_dim is not None else env_spec.observation_dim
        #print(f"observation dim: {self.n}")
        #print(f"action dim: {env_spec.action_dim}")
        self.n = env_spec.observation_dim + env_spec.action_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg_coef = reg_coef
        self.use_gpu = use_gpu
        self.expert_paths = expert_paths

        self.model = nn.Sequential()
        self.model.add_module('fc_0', nn.Linear(self.n, 128))
        self.model.add_module('tanh_0', nn.Tanh())
        self.model.add_module('fc_1', nn.Linear(128, 128))
        self.model.add_module('tanh_1', nn.Tanh())
        self.model.add_module('fc_2', nn.Linear(128, 1))

        if self.use_gpu:
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate, weight_decay=reg_coef)
        self.loss_function = torch.nn.BCEWithLogitsLoss()

    def _features(self, paths):
        o = np.concatenate([path["observations"] for path in paths])
        a = np.concatenate([path["actions"] for path in paths])
        o = np.clip(o, -10, 10)/10.0
        if o.ndim > 2:
            o = o.reshape(o.shape[0], -1)
        oa = np.concatenate((o, a), axis=1)
        #print(f"feature shape: {oa.shape}")
        return oa

    def fit(self, paths, return_errors=False):

        featmat = self._features(paths)
        expert_feat = self._features(self.expert_paths)
        featmat = featmat.astype('float32')
        expert_feat = expert_feat.astype('float32')
        #num_samples = returns.shape[0]

        # Make variables with the above data
        if self.use_gpu:
            featmat_var = Variable(torch.from_numpy(featmat).cuda(), requires_grad=False)
            expert_var = Variable(torch.from_numpy(expert_feat).cuda(), requires_grad=False)
        else:
            featmat_var = Variable(torch.from_numpy(featmat), requires_grad=False)
            expert_var = Variable(torch.from_numpy(expert_feat), requires_grad=False)

        for ep in range(self.epochs):
            fake = self.model(featmat_var)
            real = self.model(expert_var)
            self.optimizer.zero_grad()
            
            if self.use_gpu:
                disc_loss = self.loss_function(fake, torch.ones(fake.shape[0], 1).cuda()) + self.loss_function(real, torch.zeros(real.shape[0], 1).cuda())
            else:
                disc_loss = self.loss_function(fake, torch.ones(fake.shape[0], 1)) +\
                            self.loss_function(real, torch.zeros(real.shape[0], 1))

            disc_loss.backward()
            self.optimizer.step()
            """
            rand_idx = np.random.permutation(num_samples)
            for mb in range(int(num_samples / self.batch_size) - 1):
                if self.use_gpu:
                    data_idx = torch.LongTensor(rand_idx[mb*self.batch_size:(mb+1)*self.batch_size]).cuda()
                else:
                    data_idx = torch.LongTensor(rand_idx[mb*self.batch_size:(mb+1)*self.batch_size])
                batch_x = featmat_var[data_idx]
                #batch_y = returns_var[data_idx]
                self.optimizer.zero_grad()
                yhat = self.model(batch_x)
                loss = self.loss_function(yhat, batch_y)
                loss.backward()
                self.optimizer.step()
            """

    def compute_reward(self, path):
        featmat = self._features([path]).astype('float32')
        if self.use_gpu:
            feat_var = Variable(torch.from_numpy(featmat).float().cuda(), requires_grad=False)
            reward = -F.logsigmoid(self.model(feat_var)).cpu().data.numpy().ravel()
        else:
            feat_var = Variable(torch.from_numpy(featmat).float(), requires_grad=False)
            reward = -F.logsigmoid(self.model(feat_var)).data.numpy().ravel()
        #print(f"{featmat.shape}\t{reward.shape}")
        return reward
