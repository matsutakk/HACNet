"""

Model manager handles type of actor and critic

Model Class contains actor and critic.
Actor generate images from tabular data.
Critic classify images created by actor.

"""

import torch
from torch import nn

from actors.propose import HACNet
from critics.vanillacnn import VanillaCNN
from critics.resnet import ResNet18


class ModelManager:
    def __init__(self, cfg, in_dim, out_dim,):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.actor_type = cfg.actor_type
        self.critic_type = cfg.critic_type

    def _get_actor(self, hparam):
        if self.actor_type == 'HACNet':
            return HACNet(hparam, self.in_dim,)

        else:
            print('Invalid actor type')
            exit(1)

    def _get_critic(self, image_scale):
        if self.critic_type == 'VanillaCNN':
            return VanillaCNN(image_scale, self.out_dim)

        elif self.critic_type == 'ResNet18':
            return ResNet18(self.out_dim)

        else:
            print('Invalid critic type')
            exit(1)

    def get_model(self, hparam):
        actor = self._get_actor(hparam)
        critic = self._get_critic(actor.image_scale)
        return Model(actor=actor,
                     critic=critic,
                     image_scale=actor.image_scale,
                    )

class Model(nn.Module):
    def __init__(self, actor, critic, image_scale):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.image_scale = image_scale

    def forward(self, x):
        pixels, _ = self.actor(x)
        img = pixels.reshape(-1, self.image_scale, self.image_scale)
        logits = self.critic(img.unsqueeze(1))
        return logits, img
