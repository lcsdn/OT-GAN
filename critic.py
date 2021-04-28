# Credit to AKASHKADEL

from torch import nn, Tensor

class Critic(nn.Module):
    def __init__(self, num_channels: int, num_features: int, dim_critic: int) -> None:
        """
        Args:
            num_channels: (int) number of channels in the images.
            num_features: (int) number of features in the hidden layers.
            dim_critic: (int) dimension of latent critic vector.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(num_channels, num_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 4, dim_critic, 4, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, images: Tensor, labels: Tensor) -> Tensor:
        x = self.network(images)
        x = x.view(x.shape[0], -1)
        return x
