import torch
import torch.nn as nn
import torchvision.models as models

class GazeTrakModel(nn.Module):
    def __init__(self, in_channels=16, output_dim=2, pretrained=True):
        """
        GazeTrak Neural Network Model based on ResNet-18.
        
        Args:
            in_channels (int): Number of input channels (default: 16).
            output_dim (int): Number of output regression targets (default: 2, for x and y).
            pretrained (bool): Whether to load ImageNet pre-trained weights for the backbone.
        """
        super(GazeTrakModel, self).__init__()
        
        # 1. Load ResNet-18 backbone
        # Using weights='DEFAULT' loads the best available weights (IMAGENET1K_V1)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # 2. Modify first layer to accept 'in_channels' (e.g., 8)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 
            original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=original_conv1.bias
        )
        
        # Initialize the new layer weights using Kaiming He initialization
        # Because acoustic echo maps are statistically different from RGB images, 
        # random initialization is preferred over copying/averaging RGB weights.
        nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # 3. Modify the Fully Connected (FC) head for regression (2 outputs: x, y)
        # Original fc layer: Linear(in_features=512, out_features=1000)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, output_dim)

    def forward(self, x):
        """
        Forward pass.
        Input x: (Batch, 8, Height, Width) - Stacked Echo Profiles
        Output: (Batch, 2) - Normalized (x, y) coordinates
        """
        return self.backbone(x)

def finetune(model, dataloader, epochs=5, learning_rate=1e-2, device='cpu'):
    """
    Fine-tunes the model on a small dataset (e.g., 15s of user calibration data).
    Uses MSELoss to match the paper training objective.
    
    Strategy: 
    - Freeze the intermediate layers of the backbone to preserve pre-trained features.
    - Train ONLY the input layer (adapter) and the output head (regressor).
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze:
    # 1. Input Conv Layer (to adapt to acoustic data distribution)
    for param in model.backbone.conv1.parameters():
        param.requires_grad = True
        
    # 2. Output Step (to adapt to user-specific geometry/calibration)
    for param in model.backbone.fc.parameters():
        param.requires_grad = True
        
    # Setup Optimizer only for the unfrozen parameters
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate
    )
    
    print("Starting Fine-tuning (Frozen Backbone)...")
    model.train()
    
    for epoch in range(epochs):
        train_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_loss = train_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    print("Fine-tuning complete.")
    return model
