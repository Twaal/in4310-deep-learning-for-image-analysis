import torch
import torch.nn as nn
from torch import Tensor
from ResnetBlocks import BasicBlock, BottleneckBlock
from typing import Type


class ResNet(nn.Module):
    '''
    Some basic definitions: 

    - Layer: The basic building unit in neural networks, such as convolutional layers, batch normalization layers, and ReLU activation layers.

    - Block: A group of layers that together form a residual block in ResNet. 
            Each block typically consists of a few convolutional layers and a shortcut (residual) connection.

    - Stage: A sequence of blocks that operate at the same spatial resolution in the network. 
            For example, in ResNet, the network downsamples (reduces spatial dimensions) at the beginning of a new stage.
    '''
    def __init__(self, img_channels: int, num_layers: int=18, num_classes: int  = 1000):
        super(ResNet, self).__init__()
        if num_layers == 18:
            # Number of blocks per stage for ResNet-18 is 2,2,2,2 per stage.
            block_counts  = [2, 2, 2, 2]

            # The expansion number is 1
            self.expansion = 1

            # The ResNet architecture utilizes the BasicBlock structure for its blocks
            blockType = BasicBlock

        elif num_layers == 34:
            # TODO: 
            # 1) Define the number of blocks per stage. Hint: ResNet-34 has 3, 4, 6, and 3 blocks per stage.
            # 2) Set the expansion factor. Hint: ResNet-34 uses BasicBlock, which has an expansion factor of 1.
            # 3) Set the block type to BasicBlock.

        
        elif num_layers == 50:
            # TODO:
            # 1) Define the number of blocks per stage. Hint: ResNet-50 uses bottleneck blocks with 3, 4, 6, and 3 blocks per stage.
            # 2) Set the expansion factor. Hint: Bottleneck blocks have an expansion factor of 4.
            # 3) Set the block type to BottleneckBlock.
        
        elif num_layers == 101:
            # TODO:
            # 1) Define the number of blocks per stage. Hint: ResNet-101 uses bottleneck blocks with 3, 4, 23, and 3 blocks per stage.
            # 2) Set the expansion factor. Hint: Bottleneck blocks have an expansion factor of 4.
            # 3) Set the block type to BottleneckBlock.
        
        elif num_layers == 152:
            # TODO:
            # 1) Define the number of blocks per stage. Hint: ResNet-152 uses bottleneck blocks with 3, 8, 36, and 3 blocks per stage.
            # 2) Set the expansion factor. Hint: Bottleneck blocks have an expansion factor of 4.
            # 3) Set the block type to BottleneckBlock.
            
        else:
            raise ValueError("Only ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152 are supported in this implementation.")
        
        self.in_channels = 64
        
        # Initial convolutional layer with a large kernel (7x7), stride of 2, and padding of 3
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Create the stages of the network using the build_stage method
        self.stage1 = self.build_stage(64, block_counts[0], stride=1, block=blockType)
        self.stage2 = self.build_stage(128, block_counts[1], stride=2, block=blockType)
        self.stage3 = self.build_stage(256, block_counts[2], stride=2, block=blockType)
        self.stage4 = self.build_stage(512, block_counts[3], stride=2, block=blockType)
        
        # Global average pooling to reduce the spatial dimensions to 1x1 before the fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer to produce the final class predictions
        self.fc = nn.Linear(512*self.expansion, num_classes)
    

    def build_stage(self, out_channels: int, num_blocks: int, stride: int, block) -> nn.Sequential:
        '''
        Builds a stage which consists of several residual blocks.

        Args:
            out_channels (int): The number of output channels for the blocks in this stage.
            num_blocks (int): The number of blocks to include in this stage.
            stride (int): The stride for the first block in this stage (used for downsampling).

        Returns:
            nn.Sequential: A stage composed of multiple residual blocks.
        '''

        blocks = []
        downsample = None

        # Downsampling if the spatial resolution needs to be reduced or if the channels change
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        
        # Create the first block, This block might need downsampling and thats why we have to add it seperately
        blocks.append(block(self.in_channels, out_channels, stride, self.expansion, downsample))

        # Update the number of input channels for the next blocks
        self.in_channels = out_channels * self.expansion

        # Add the remaining blocks (which do not need downsampling)
        for i in range(1, num_blocks):
            blocks.append(block(self.in_channels, out_channels, expansion=self.expansion))

        # Return the sequential container holding all blocks in this stage
        return nn.Sequential(*blocks)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Pass through each of the stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Global average pooling
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x