import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class CNNEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        if CNN_MODE == "1D":
            self.conv1 = nn.Conv1d(input_channels, 32, 7, padding=2)
            self.conv2 = nn.Conv1d(32, 64, 5, padding=1)
            self.pool  = nn.MaxPool1d(2)
            

        else:
            self.conv1 = nn.Conv2d(1, 32, 7, stride=(1,1), padding=2)
            self.conv2 = nn.Conv2d(32, 64, 5, stride=(1,1), padding=1)
            self.pool  = nn.MaxPool2d((1,2))

    def forward(self, x):

        if CNN_MODE == "1D":
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            #x = self.pool(x)
        else:
            x = x.unsqueeze(1)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            #x = self.pool(x)

        return torch.flatten(x, 1)
    

  
class FusionNet(nn.Module):
    def __init__(self, weighted_shape, dna2vec_shape, decision_dim):
        super().__init__()

        self.use_weighted = USE_WEIGHTED and weighted_shape is not None
        
        self.use_dna2vec  = USE_DNA2VEC and dna2vec_shape is not None
      
        self.use_decision = USE_DECISION is not None and decision_dim is not None
        

        if self.use_weighted:
            self.weighted_encoder = CNNEncoder(weighted_shape[0])

        if self.use_dna2vec:
            if CNN_MODE == "1D":
                self.dna_encoder = CNNEncoder(dna2vec_shape[0])  # embedding dim
            else:
                self.dna_encoder = CNNEncoder(1)

        fusion_dim = 0

        if self.use_weighted:
            fusion_dim += self._get_dim(self.weighted_encoder, weighted_shape)

        if self.use_dna2vec:
            fusion_dim += self._get_dim(self.dna_encoder, dna2vec_shape)

        if self.use_decision:
            fusion_dim += decision_dim
        

        self.fc1 = nn.Linear(fusion_dim, 48)
        self.fc2 = nn.Linear(48, 16)
        self.out = nn.Linear(16, 2)

        self.dropout = nn.Dropout(DROPOUT)

    def _get_dim(self, encoder, shape):
        dummy = torch.zeros(1, *shape)
        return encoder(dummy).shape[1]

    def forward(self, weighted, dna2vec, decision):
        

        features = []

        if self.use_weighted:
            w = self.weighted_encoder(weighted)
            w = F.normalize(w, p=2, dim=1) 
            #w = F.layer_norm(w, w.shape[1:])
            features.append(w)
        
        if self.use_dna2vec:
            d = self.dna_encoder(dna2vec)
            d = F.normalize(d, p=2, dim=1)
            features.append(d)

        if self.use_decision and decision is not None:
            #decision = F.layer_norm(decision, decision.shape[1:])
            decision = F.normalize(decision, p=2, dim=1)
            features.append(decision)
        

        if len(features) == 0:
            raise ValueError("No active input branches.")
        x = torch.cat(features, dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))   

        return self.out(x)