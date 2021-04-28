from torch import nn, Tensor

class Generator(nn.Module):
    def __init__(self, num_channels: int, dim_noise: int, num_features: int) -> None:
        """
        Args:
            num_channels: (int) number of channels in the images.
            dim_noise: (int) dimension of input noise.
            num_features: (int) number of features in the hidden layers.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(dim_noise, num_features*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_features*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_features*4, num_features*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(num_features*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_features*2, num_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_features, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
  
    def forward(self, noise: Tensor, labels: Tensor) -> Tensor:
        x = self.network(noise)
        return x