import torch
from torch import Tensor
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer

class OT_GAN(nn.Module):
    """Wrapper for OT GAN model."""
    
    def __init__(
            self,
            generator: Module,
            critic: Module,
            sinkhorn: Module,
            opt_generator: Optimizer,
            opt_critic: Optimizer,
        ) -> None:
        """
        Args:
            generator: (nn.Module) Generator network.
            critic: (nn.Module) Critic network.
            sinkhorn: (nn.Module) Sinkhorn algorithm with autodifferentiation.
            opt_generator: (nn.Module) Optimizer for generator network.
            opt_critic: (nn.Module) Optimizer for critic network.
        """
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.sinkhorn = sinkhorn
        self.opt_generator = opt_generator
        self.opt_critic = opt_critic
    
    def forward(self, noise: Tensor, labels: Tensor) -> Tensor:
        """
        Generate images from noise.
        
        Args:
            noise: (Tensor) Batch of noise vectors.
            labels: (Tensor) Labels for the generated images (used for
            conditional generation).
        Returns:
            images: (Tensor) Batch of generated images.
        """
        return self.generator(noise, labels)
    
    def wasserstein(self, latent1: Tensor, latent2: Tensor) -> Tensor:
        """
        Compute the entropy-regularised wasserstein distance between
        uniform distributions for cost 1 - dot product.
        
        Args:
            latent1: (Tensor) Set of latent vectors 1, dimension NxD.
            latent2: (Tensor) Set of latent vectors 2, dimension MxD.
        Returns:
            W: (Tensor) Wasserstein distance, dimension 0.
        """
        C = 1 - latent1.matmul(latent2.T)
        W = self.sinkhorn(C)
        return W
    
    def compute_loss(self, images: Tensor, labels: Tensor, noise: Tensor) -> Tensor:
        """
        Compute the OT GAN loss:
        
        loss = W_c(X,Y) + W_c(X,Y') + W_c(X',Y) + W_c(X',Y')
               - 2 W_c(X, X') - 2 W_c(Y, Y')

        Where X, X' independent real images, and Y, Y' independent generated images.
        
        Args:
            images: (Tensor) Real images.
            labels: (Tensor) Real labels (used for conditional generation).
            noise: (Tensor) Noise input to the generator network.
        Returns:
            loss: (Tensor) OT GAN loss.
        """
        half_batch_size = images.shape[0] // 2
    
        true_latent = self.critic(images, labels)
        fake_latent = self.critic(self.generator(noise, labels), labels)
    
        X1 = true_latent[:half_batch_size]
        X2 = true_latent[half_batch_size:]
        Y1 = fake_latent[:half_batch_size]
        Y2 = fake_latent[half_batch_size:]
    
        loss = (
            self.wasserstein(X1, Y1)
            + self.wasserstein(X1, Y2)
            + self.wasserstein(X2, Y1)
            + self.wasserstein(X2, Y2)
            - 2 * self.wasserstein(X1, X2)
            - 2 * self.wasserstein(Y1, Y2)
        )
        return loss    
            
    def train_generator_step(self, *data) -> Tensor:
        """Train step for generator network."""
        self.opt_generator.zero_grad()
        loss = self.compute_loss(*data)
        loss.backward()
        self.opt_generator.step()
        return loss
        
    def train_critic_step(self, *data) -> Tensor:
        """Train step for critic network."""
        self.critic.zero_grad()
        loss = self.compute_loss(*data)
        (- loss).backward()
        self.opt_critic.step()
        return loss