from torch import nn, Tensor
import torch

from utils.conf import Configuration
from model.ResidualAE import TSEncoder, TSDecoder


class AEBuilder(nn.Module):
    def __init__(self, conf: Configuration) -> None:
        super(AEBuilder, self).__init__()

        self.__encoder = TSEncoder(conf)
        self.__decoder = TSDecoder(conf)

        self.fc_layer = nn.Linear(conf.getHP(
            'dim_embedding')*2, conf.getHP('dim_embedding'))

    def encode(self, input: Tensor) -> Tensor:
        return self.__encoder(input)

    def decode(self, input: Tensor) -> Tensor:
        return self.__decoder(input)

    # explicit model.encode/decode is preferred as decoder might not exist
    # forward is mostly for examining no. parameters
    def forward(self, input: Tensor) -> Tensor:
        embedding1 = self.encode(input)
        recons1 = self.decode(embedding1)

        embedding2 = self.encode(recons1)
        cat_embedding = torch.cat([embedding1, embedding2], dim=1)
        new_embedding = self.fc_layer(cat_embedding)
        recons2 = self.decode(new_embedding)

        return recons1, recons2, embedding1, new_embedding
