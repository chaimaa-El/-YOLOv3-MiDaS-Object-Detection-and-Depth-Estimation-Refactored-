import torch
import torch.nn as nn

# Function to create a ResNet backbone for the model
def _make_resnet_backbone(resnet):
    """
    Creates a custom ResNet backbone by extracting relevant layers from a pre-trained ResNet model.

    Args:
        resnet (torch.nn.Module): Pre-trained ResNet model.

    Returns:
        nn.Module: Modified ResNet model with selected layers.
    """
    # Create a new module to hold the modified layers
    pretrained = nn.Module()
    
    # Modify the first layer (conv1, bn1, relu, maxpool, layer1)
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    # Use layers 2, 3, and 4 from the original ResNet model
    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained

# Function to load a pre-trained ResNext101 model and create the ResNet backbone
def _make_pretrained_resnext101_wsl(use_pretrained):
    """
    Loads a pre-trained ResNext101 model with WSL (Weakly Supervised Learning) and returns the modified ResNet backbone.

    Args:
        use_pretrained (bool): Whether to load the pre-trained weights.

    Returns:
        nn.Module: Modified ResNet backbone using ResNext101.
    """
    # Load the pre-trained ResNext101 model from the sarvan0506 repository
    resnet = torch.hub.load("sarvan0506/WSL-Images", "resnext101_32x8d_wsl", pretrained=use_pretrained)
    
    # Return the modified ResNet backbone
    return _make_resnet_backbone(resnet)

# Function to create a scratch backbone with custom convolution layers
def _make_scratch(in_shape, out_shape):
    """
    Creates a scratch (non-pretrained) backbone with custom convolution layers.

    Args:
        in_shape (list): List of input feature map shapes for each layer.
        out_shape (int): Number of output features.

    Returns:
        nn.Module: Custom scratch backbone with convolutional layers.
    """
    scratch = nn.Module()

    # Define custom convolution layers for each input shape
    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    return scratch

# Function to create both a pretrained and scratch encoder
def _make_encoder(features, use_pretrained):
    """
    Creates both a pretrained and scratch encoder.

    Args:
        features (int): Number of features for the scratch backbone.
        use_pretrained (bool): Whether to use pretrained weights for the ResNet backbone.

    Returns:
        tuple: Tuple containing the pretrained ResNet backbone and the scratch backbone.
    """
    pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
    scratch = _make_scratch([256, 512, 1024, 2048], features)

    return pretrained, scratch

# Interpolation module to scale tensors
class Interpolate(nn.Module):
    """
    Interpolation module to scale input tensors.
    """
    def __init__(self, scale_factor, mode):
        """
        Initializes the Interpolate module.

        Args:
            scale_factor (float): Factor by which to scale the input.
            mode (str): Mode of interpolation ('bilinear', 'nearest', etc.)
        """
        super(Interpolate, self).__init__()

        # Use PyTorch's built-in interpolate function
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """
        Forward pass to apply the interpolation.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Scaled tensor.
        """
        # Perform the interpolation
        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False
        )

        return x

# Residual Convolution Unit (RCU) for feature refinement
class ResidualConvUnit(nn.Module):
    """
    Residual convolution module that refines features using two convolution layers.
    """
    def __init__(self, features):
        """
        Initializes the ResidualConvUnit module.

        Args:
            features (int): Number of input and output features.
        """
        super().__init__()

        # Define two convolutional layers
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        # ReLU activation for non-linearity
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass to apply the convolutions and residual connection.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor with added residual.
        """
        # Apply the first convolution, followed by ReLU, then the second convolution
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        # Add the residual connection (skip connection)
        return out + x

# Feature Fusion Block to combine multiple feature maps
class FeatureFusionBlock(nn.Module):
    """
    Feature fusion block that combines multiple feature maps using residual convolutions.
    """
    def __init__(self, features):
        """
        Initializes the FeatureFusionBlock module.

        Args:
            features (int): Number of features to be processed.
        """
        super(FeatureFusionBlock, self).__init__()

        # Two ResidualConvUnit layers for processing input features
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """
        Forward pass to fuse multiple feature maps.

        Args:
            *xs (tensor): Input feature maps to be fused.

        Returns:
            tensor: Fused feature map.
        """
        # Start with the first input feature map
        output = xs[0]

        # If there are two inputs, apply the first ResidualConvUnit
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        # Apply the second ResidualConvUnit
        output = self.resConfUnit2(output)

        # Upsample the output using bilinear interpolation
        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output
