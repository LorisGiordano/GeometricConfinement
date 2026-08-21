""" IMPORTS """

from monai.networks.blocks import Convolution
from torch import nn

""" Local appearance """

class LocalAppearance(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 4, hidden_channels: int = 128, kernel_size: int = 3):
        super().__init__()

        self.conv_in = Convolution(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            adn_ordering="ADN",
            act=("leakyrelu", {"negative_slope": 0.1}),
            dropout=0.5,
            norm=None,
        )

        self.average_pool = nn.AvgPool3d(kernel_size=2, stride=2)

        self.tricubic_interpolation = nn.Upsample(
            scale_factor=2,
            mode="trilinear",
            align_corners=True,
        )

        self.conv11 = Convolution(
                spatial_dims=3,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                adn_ordering="ADN",
                act=("leakyrelu", {"negative_slope": 0.1}),
                dropout=None,
                norm=None,
            )

        self.conv12 = Convolution(
                spatial_dims=3,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                adn_ordering="ADN",
                act=("leakyrelu", {"negative_slope": 0.1}),
                dropout=0.5,
                norm=None,
            )

        self.conv21 = Convolution(
                spatial_dims=3,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                adn_ordering="ADN",
                act=("leakyrelu", {"negative_slope": 0.1}),
                dropout=None,
                norm=None,
            )
        self.conv22 = Convolution(
                spatial_dims=3,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                adn_ordering="ADN",
                act=("leakyrelu", {"negative_slope": 0.1}),
                dropout=0.5,
                norm=None,
            )
        self.conv31 = Convolution(
                spatial_dims=3,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                adn_ordering="ADN",
                act=("leakyrelu", {"negative_slope": 0.1}),
                dropout=None,
                norm=None,
            )
        self.conv32 = Convolution(
                spatial_dims=3,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                adn_ordering="ADN",
                act=("leakyrelu", {"negative_slope": 0.1}),
                dropout=0.5,
                norm=None,
            )
        self.conv41 = Convolution(
                spatial_dims=3,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                adn_ordering="ADN",
                act=("leakyrelu", {"negative_slope": 0.1}),
                dropout=None,
                norm=None,
            )
        self.conv42 = Convolution(
                spatial_dims=3,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                adn_ordering="ADN",
                act=("leakyrelu", {"negative_slope": 0.1}),
                dropout=0.5,
                norm=None,
            )

        self.conv_out = Convolution(
            spatial_dims=3,
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            adn_ordering="ADN",
            act=None,
            dropout=None,
            norm=None,
        )

    def forward(self, x):
        x_in = self.conv_in(x)
        x_11 = self.conv11(x_in)
        x_12 = self.conv12(x_11)
        x_20 = self.average_pool(x_11)

        x_21 = self.conv21(x_20)
        x_22 = self.conv22(x_21)
        x_30 = self.average_pool(x_21)

        x_31 = self.conv31(x_30)
        x_32 = self.conv32(x_31)
        x_40 = self.average_pool(x_31)

        x_41 = self.conv41(x_40)
        x_42 = self.conv42(x_41)

        x_33 = self.tricubic_interpolation(x_42) + x_32
        x_23 = self.tricubic_interpolation(x_33) + x_22
        x_13 = self.tricubic_interpolation(x_23) + x_12
        x_out = self.conv_out(x_13)
        
        return x_out


""" Spatial configuration """

class SpatialConfiguration(nn.Module):

    def __init__(self, num_channels: int = 4, hidden_channels: int = 128, kernel_size: int = 7, downsample: int = 4):
        super().__init__()

        self.downsample = nn.AvgPool3d(kernel_size=downsample)

        self.conv_in = Convolution(
            spatial_dims=3,
            in_channels=num_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            adn_ordering="ADN",
            act=("leakyrelu", {"negative_slope": 0.1}),
            dropout=None,
            norm=None,
        )

        self.conv1 = Convolution(
            spatial_dims=3,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            adn_ordering="ADN",
            act=("leakyrelu", {"negative_slope": 0.1}),
            dropout=None,
            norm=None,
        )

        self.conv2 = Convolution(
            spatial_dims=3,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            adn_ordering="ADN",
            act=("leakyrelu", {"negative_slope": 0.1}),
            dropout=None,
            norm=None,
        )

        self.conv_out = Convolution(
            spatial_dims=3,
            in_channels=hidden_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            adn_ordering="ADN",
            act=None,
            dropout=None,
            norm=None,
        )

        self.activation = nn.Tanh()

        self.upsample = nn.Upsample(
            scale_factor=downsample,
            mode="trilinear",
            align_corners=True,
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv_in(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv_out(x)
        x = self.activation(x)
        x = self.upsample(x)

        return x

        
""" SCN3D """

class SCN(nn.Module):
    def __init__(self,      
                in_channels, 
                out_channels, 
                learnable_sigma: bool = False, 
                init_sigma: float = 2.0, 
                gamma: float = 1000.0, 
                la_channels: int = 128, 
                sc_channels: int = 128, 
                la_kernel: int = 3, 
                sc_kernel: int = 7, 
                downsample: int = 4):
                
        super().__init__()

        self.la = LocalAppearance(in_channels=in_channels, out_channels=out_channels, hidden_channels=la_channels, kernel_size=la_kernel)

        self.sc = SpatialConfiguration(num_channels=out_channels, hidden_channels=sc_channels, kernel_size=sc_kernel, downsample=downsample)

        self.learnable_sigma = learnable_sigma

        if self.learnable_sigma:
            self.sigma = nn.Parameter(torch.ones(out_channels) * init_sigma)
            self.gamma = gamma / 15.74961 # gamma / sqrt(2 * pi) ** 3


        self.input_shape = None

    def forward(self, x, gt_coords=None):
        x_la_out = self.la(x)
        x_sc_out = self.sc(x_la_out)

        x_out = x_la_out * x_sc_out

        if self.learnable_sigma:
            if gt_coords is None:
                raise ValueError("Ground truth coordinates (gt_coords) must be provided for sigma calculation.")
            
            if self.input_shape is None or self.input_shape != x.shape:
                self.input_shape = x.shape
                D, H, W = self.input_shape[-3:]

                self.zs = torch.arange(D, device=x.device, dtype=float).view(1,1,D,1,1)
                self.ys = torch.arange(H, device=x.device, dtype=float).view(1,1,1,H,1)
                self.xs = torch.arange(W, device=x.device, dtype=float).view(1,1,1,1,W)

            B, N = x.shape[-5], x.shape[-4]
            z0 = gt_coords[...,0].view(B,N,1,1,1)
            y0 = gt_coords[...,1].view(B,N,1,1,1)
            x0 = gt_coords[...,2].view(B,N,1,1,1)

            # broadcast per-landmark sigma
            sig = self.sigma.view(1,N,1,1,1)
            sqd = (self.zs - z0)**2 + (self.ys - y0)**2 + (self.xs - x0)**2
            gt_vols = ( self.gamma/(sig**3) ) * torch.exp(-sqd / (2 * sig**2))

            return x_out, gt_vols

        return x_out