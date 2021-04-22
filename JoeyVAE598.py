from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


def hello_vae():
    print("Hello from vae.py!")


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 5, padding=2, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   
    

class VAE(nn.Module):
    def __init__(self, input_size, latent_size=15, hidden_size=None):
        super(VAE, self).__init__()
        self.input_size = input_size # 28*28
        self.latent_size = latent_size # Z
        self.hidden_dim = hidden_size # H
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ############################################################################################
        # TODO: Implement the fully-connected encoder architecture described in the notebook.      #
        # Specifically, self.encoder should be a network that inputs a batch of input images of    #
        # shape (N, 1, H, W) into a batch of hidden features of shape (N, H). Set up self.mu_layer #
        # and self.logvar_layer to be a pair of linear layers that map the hidden features into    #
        # estimates of the mean and variance of the posterior over the latent vectors; the mean    #
        # and variance estimates will both be tensors of shape (N, Z).                             #
        ############################################################################################
        # Replace "pass" statement with your code
        if self.hidden_dim is None:
          self.hidden_dim = int(input_size ** 0.5)
        
        layers = []

        # FCN
        # layers.append(nn.Flatten())
        # layers.append(nn.Linear(3 * self.input_size, self.hidden_dim))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        # layers.append(nn.ReLU())

        # CNN
        layers.append(conv_block(3, 16))
        layers.append(nn.MaxPool2d(2))
        layers.append(conv_block(16, 32))
        layers.append(nn.MaxPool2d(2))
        layers.append(conv_block(32, 64))
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(576, self.hidden_dim))
        layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

        self.mu_layer = nn.Linear(self.hidden_dim, self.latent_size)
        self.logvar_layer = nn.Linear(self.hidden_dim, self.latent_size)
        ############################################################################################
        # TODO: Implement the fully-connected decoder architecture described in the notebook.      #
        # Specifically, self.decoder should be a network that inputs a batch of latent vectors of  #
        # shape (N, Z) and outputs a tensor of estimated images of shape (N, 1, H, W).             #
        ############################################################################################
        # Replace "pass" statement with your code
        layers2 = []
        H = int(np.sqrt(self.input_size))

        # FCN
        layers2.append(nn.Linear(self.latent_size, self.hidden_dim))
        layers2.append(nn.ReLU())
        layers2.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers2.append(nn.ReLU())
        layers2.append(nn.Linear(self.hidden_dim, 3 * self.input_size))
        layers2.append(nn.Unflatten(1, (3, H, H)))
    
        layers2.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers2)
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################


    def forward(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ############################################################################################
        # TODO: Implement the forward pass by following these steps                                #
        # (1) Pass the input batch through the encoder model to get posterior mu and logvariance   #
        # (2) Reparametrize to compute  the latent vector z                                        #
        # (3) Pass z through the decoder to resconstruct x                                         #
        ############################################################################################
        # Replace "pass" statement with your code
        encoding = self.encoder(x)
        mu = self.mu_layer(encoding)
        logvar = self.logvar_layer(encoding)
        z = reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar


class CVAE(nn.Module):
    def __init__(self, input_size, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size # # 28*28
        self.latent_size = latent_size # Z
        self.num_classes = num_classes # C
        self.hidden_dim = None # H
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ############################################################################################
        # TODO: Define a FC encoder as described in the notebook that transforms the image         #
        # (N, 1, H, W) into a hidden_dimension (N, H) feature space, and a final two layers that   #
        # project that feature map (after flattening and now adding our one-hot class vector) to   #
        # posterior mu and posterior variance estimates of the latent space (N, Z)                 #
        ############################################################################################
        # Replace "pass" statement with your code
        if self.hidden_dim is None:
          self.hidden_dim = input_size // 2
        
        layers = []
        layers.append(nn.Linear(self.input_size + self.num_classes, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

        self.mu_layer = nn.Linear(self.hidden_dim + self.num_classes, self.latent_size)
        self.logvar_layer = nn.Linear(self.hidden_dim + self.num_classes, self.latent_size)

        ############################################################################################
        # TODO: Define a fully-connected decoder as described in the notebook that transforms the  #
        # latent space (N, Z + C) to the D input dimension                                         #
        ############################################################################################
        # Replace "pass" statement with your code
        layers2 = []
        layers2.append(nn.Linear(self.latent_size + self.num_classes, self.hidden_dim))
        layers2.append(nn.ReLU())
        layers2.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers2.append(nn.ReLU())
        layers2.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers2.append(nn.ReLU())
        layers2.append(nn.Linear(self.hidden_dim, self.input_size))
        layers2.append(nn.Sigmoid())
        H = int(np.sqrt(self.input_size))
        layers2.append(nn.Unflatten(1, (1, H, H)))
        self.decoder = nn.Sequential(*layers2)
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################

    def forward(self, x, c):
        """
        Performs forward pass through FC-CVAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Input data for this timestep of shape (N,1,H,W)
        - c: One hot vector representing the input class (0-9) (N, C)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z),  with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ############################################################################################
        # TODO: Implement the forward pass by following these steps                                #
        # (1) Pass the input batch through the encoder model to get posterior mu and logvariance   #
        # (2) Reparametrize to compute  the latent vector z                                        #
        # (3) Pass z through the decoder to resconstruct x                                         #
        ############################################################################################
        # Replace "pass" statement with your code
        x_flat = torch.flatten(x, start_dim=1)
        x_concat = torch.cat((x_flat, c), 1)
        encoding = self.encoder(x_concat)

        encoding_c = torch.cat((encoding, c), 1)
        mu = self.mu_layer(encoding_c)
        logvar = self.logvar_layer(encoding_c)

        z = reparametrize(mu, logvar)
        z_c = torch.cat((z, c), 1)
        x_hat = self.decoder(z_c)
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar



def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance using the
    reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with mean mu and
    standard deviation sigma, such that we can backpropagate from the z back to mu and sigma.
    We can achieve this by first sampling a random value epsilon from a standard Gaussian
    distribution with zero mean and unit variance, then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network, it helps to
    pass this function the log of the variance of the distribution from which to sample, rather
    than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns: 
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
    """
    z = None
    ################################################################################################
    # TODO: Reparametrize by initializing epsilon as a normal distribution and scaling by          #
    # posterior mu and sigma to estimate z                                                         #
    ################################################################################################
    # Replace "pass" statement with your code
    N, Z = mu.shape
    samples = torch.normal(0, 1, size=(N, Z))
    samples = samples.to(mu.device)
    z = mu + torch.sqrt(torch.exp(logvar)) * samples
    ################################################################################################
    #                              END OF YOUR CODE                                                #
    ################################################################################################
    return z


def loss_function(x_hat, x, mu, logvar):
    """
    Computes the variational lower bound loss term of the VAE (refer to formulation in notebook).

    Inputs:
    - x_hat: Reconstruced input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
    - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
    
    Returns:
    - loss: Tensor containing the scalar loss for the variational lowerbound
    """
    loss = None
    ################################################################################################
    # TODO: Compute variational lowerbound loss as described in the notebook                       #
    ################################################################################################
    # Replace "pass" statement with your code
    N, __, H, W = x_hat.shape
    N, Z = mu.shape

    ce_loss = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kl_loss = 1 + logvar - mu**2 - torch.exp(logvar)
    kl_loss = 0.5 * torch.sum(kl_loss)

    # Why is it 2 * ce_loss, and divided by 2 not 1
    loss = (ce_loss - kl_loss) / N
    # print(loss, N, ce_loss, kl_loss)
    ################################################################################################
    #                            END OF YOUR CODE                                                  #
    ################################################################################################
    return loss

