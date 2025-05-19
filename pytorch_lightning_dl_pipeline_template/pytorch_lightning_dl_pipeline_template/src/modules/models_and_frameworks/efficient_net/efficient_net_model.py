import torch


class EfficientNetModel(torch.nn.Module):
    def __init__(self, config):
        super(EfficientNetModel, self).__init__()
        self.model = efficientnet_b0()
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features,
            config.number_of_classes
        )

    def forward(self, model_input):
        model_output = self.model(model_input.repeat(1, 3, 1, 1))
        return model_output


class ConvolutionalBlock(torch.nn.Module):
    """
        The blue square named 'Conv Block' in Figure 2
        of article https://arxiv.org/pdf/2311.15719v1
    """
    def __init__(self, **kwargs):
        super(ConvolutionalBlock, self).__init__()
        self.convolutional_block = None

        self._setup(**kwargs)

    def forward(self, input_feature_map):
        output_feature_map = self.convolutional_block(input_feature_map)
        return output_feature_map

    def _setup(self, **kwargs):
        self.convolutional_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=kwargs['in_channels'],
                out_channels=kwargs['out_channels'],
                kernel_size=kwargs['kernel_size'],
                stride=kwargs['stride'],
                padding=kwargs['padding'],
                bias=False
            ),
            torch.nn.GELU(),
            torch.nn.BatchNorm2d(
                num_features=kwargs['out_channels']
            )
        )


class ConvolutionalTransposeBlock(torch.nn.Module):
    """
        The yellow square named 'Conv Transpose Block' in Figure 2
        of article https://arxiv.org/pdf/2311.15719v1
    """
    def __init__(self, **kwargs):
        super(ConvolutionalTransposeBlock, self).__init__()
        self.convolutional_transpose_block = None

        self._setup(**kwargs)

    def forward(self, input_feature_map):
        output_feature_map = \
            self.convolutional_transpose_block(input_feature_map)
        return output_feature_map

    def _setup(self, **kwargs):
        self.convolutional_transpose_block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=kwargs['in_channels'],
                out_channels=kwargs['out_channels'],
                kernel_size=kwargs['kernel_size'],
                stride=kwargs['stride'],
                padding=kwargs['padding'],
                bias=False
            ),
            torch.nn.GELU(),
            torch.nn.BatchNorm2d(
                num_features=kwargs['out_channels']
            )
        )


class ConvolutionalUpsamplingBlock(torch.nn.Module):
    """
        The purple square named 'Conv Upsampling Block' in Figure 2
        of article https://arxiv.org/pdf/2311.15719v1
    """
    def __init__(self, **kwargs):
        super(ConvolutionalUpsamplingBlock, self).__init__()
        self.convolutional_upsampling_block = None
        self.interpolation_scale_factor = None

        self._setup(**kwargs)

    def forward(self, input_feature_map):
        output_feature_map = self.convolutional_upsampling_block(
            torch.nn.functional.interpolate(
                input=input_feature_map,
                scale_factor=self.interpolation_scale_factor,
                mode='bilinear'
            )
        )
        return output_feature_map

    def _setup(self, **kwargs):
        self.convolutional_upsampling_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=kwargs['in_channels'],
                out_channels=kwargs['out_channels'],
                kernel_size=kwargs['kernel_size'],
                stride=kwargs['stride'],
                padding=kwargs['padding'],
                bias=False
            ),
            torch.nn.GELU(),
            torch.nn.BatchNorm2d(
                num_features=kwargs['out_channels']
            )
        )
        self.interpolation_scale_factor = kwargs['kernel_size']

class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = None
        self.decoder = None

        self._setup(**kwargs)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = self.encode(x)
        batch_size = x.shape[0]
        # print(x.shape[0])
        alpha = self.alpha_fc(x)
        resampler = ResampleDir(latent_size, batch_size)
        dirichlet_sample = resampler.sample(
            alpha)  # This variable that follows a Dirichlet distribution
        recon_x = self.decoder(
            dirichlet_sample)  # can be interpreted as a probability that the sum is 1)
        return recon_x, alpha, dirichlet_sample

    def _setup(self, **kwargs):
        self.encoder = torch.nn.Sequential(
            ConvolutionalBlock(
                in_channels=1,
                out_channels=base,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            ConvolutionalBlock(
                in_channels=base,
                out_channels=2 * base,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            ConvolutionalBlock(
                in_channels=2 * base,
                out_channels=2 * base,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            ConvolutionalBlock(
                in_channels=2 * base,
                out_channels=2 * base,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            ConvolutionalBlock(
                in_channels=2 * base,
                out_channels=4 * base,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            ConvolutionalBlock(
                in_channels=4 * base,
                out_channels=4 * base,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            ConvolutionalBlock(
                in_channels=4 * base,
                out_channels=4 * base,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            torch.nn.Conv2d(
                in_channels=4 * base,
                out_channels=32 * base,
                kernel_size=8
            ),
            torch.nn.GELU(),
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=32 * base,
                out_features=latent_size * base,
                bias=False
            ),
            torch.nn.BatchNorm1d(
                num_features=latent_size * base,
                momentum=0.9
            ),
            torch.nn.GELU()
        )

        self.alpha_fc = torch.nn.Linear(
            in_features=latent_size * base,
            out_features=latent_size * base
        )

        self.decoder = nn.Sequential(
            torch.nn.Linear(
                in_features=latent_size * base,
                out_features=32 * base,
                bias=False
            ),
            torch.nn.BatchNorm1d(
                num_features=32 * base,
                momentum=0.9
            ),
            torch.nn.GELU(),
            torch.nn.Unflatten(
                dim=1,
                unflattened_size=(32 * base, 1, 1)
            ),
            torch.nn.Conv2d(
                32 * base,
                32 * base,
                1
            ),
            ConvTranspose(32 * base, 4 * base, 8),
            Conv(4 * base, 4 * base, 3, padding=1),
            ConvUpsampling(4 * base, 4 * base, 4, stride=2, padding=1),
            Conv(
                4 * base, 2 * base, 3, padding=1),
            ConvUpsampling(
                2 * base, 2 * base, 4, stride=2, padding=1),
            Conv(
                2 * base, base, 3, padding=1),
            ConvUpsampling(
                base, base, 4, stride=2, padding=1),
            torch.nn.Conv2d(
                base,
                1,
                3,
                padding=1
            ),
            torch.nn.Sigmoid()
        )