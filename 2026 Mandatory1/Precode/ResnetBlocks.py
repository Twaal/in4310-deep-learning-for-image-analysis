import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    '''
    Basic blocks are used for Resnet-18 and Resnet-34 architectures.
    Each block has two convolutional layers followed by batch normalization 
    and ReLU activation. It also includes a skip connection (identity) 
    that bypasses the block.
    '''
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 1, downsample: nn.Module = None): 
        # Initialize the parent class
        super(BasicBlock, self).__init__()

        # Store the expansion and downsample module
        self.expansion = expansion
        self.downsample = downsample

        # Define the first convolutional layer
        # - Kernel size: 3x3
        # - Stride: As specified (1 by default)
        # - Padding: 1 to maintain the spatial dimension
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Define the second convolutional layer
        # - Kernel size: 3x3
        # - Stride: 1 (no downsampling)
        # - Padding: 1 to maintain the spatial dimensions
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)


    def forward(self, x):
        # Store the input as the identity
        identity = x

        # Pass through the first convolution, batch normalization, and ReLU activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Pass through the second convolution and batch normalization
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsampling if necessary
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the identity (skip connection) to the output
        out += identity

        # Apply ReLU activation
        out = self.relu(out)

        # Return the final output
        return out


class BottleneckBlock(nn.Module):
    '''
    The Bottleneck block is used for ResNet-50, ResNet-101, and ResNet-152 architectures.
    This block has three convolutional layers with batch normalization and ReLU activation.
    The bottleneck design reduces the computational load by using a 1x1 convolution to 
    reduce the dimensionality before applying a 3x3 convolution.
    '''
    def __init__(self, in_channels: int, out_channels: int, stride: int=1, expansion: int = 4, downsample: nn.Module=None):
        super(BottleneckBlock, self).__init__()
        # First convolution layer (1x1) reduces the number of channels (dimensionality reduction)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution layer (3x3) processes the features
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Third convolution layer (1x1) expands the number of channels back to the original size (expansion factor)
        self.conv3 = nn.Conv2d(out_channels, out_channels * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * expansion)

        # ReLU activation after each convolutional layer
        self.relu = nn.ReLU(inplace=True)

        # Store the downsample module (if provided) to match dimensions between identity and the output
        self.downsample = downsample

    def forward(self, x):
        # Store the input tensor as the identity (skip connection)
        identity = x

        # Pass through the first convolution (1x1), batch normalization, and ReLU activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Pass through the second convolution (3x3), batch normalization, and ReLU activation
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Pass through the third convolution (1x1) and batch normalization
        out = self.conv3(out)
        out = self.bn3(out)

        # Apply downsampling if necessary to match dimensions between the identity and the output
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the identity (skip connection) to the output (element-wise addition)
        out += identity

        # Apply ReLU activation again after the addition
        out = self.relu(out)

        # Return the final output tensor
        return out