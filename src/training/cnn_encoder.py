import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    CNN architecture for extracting features from GAF images.
    Based on the architecture from the paper (Nature-style DQNs/PPOs).
    """
    def __init__(self, input_shape=(1, 64, 64), features_dim=512):
        super(CNNEncoder, self).__init__()
        
        # input_shape: (C, H, W)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4) # Out: (32, 15, 15)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)            # Out: (64, 6, 6)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)            # Out: (64, 4, 4)
        
        # Calculate flatten size
        with torch.no_grad():
            sample = torch.zeros(1, *input_shape)
            out = self.conv3(self.conv2(self.conv1(sample)))
            self.flatten_size = out.numel()

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, features_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: (Batch, 1, 64, 64)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == "__main__":
    # Test CNN
    model = CNNEncoder()
    mock_img = torch.randn(1, 1, 64, 64)
    features = model(mock_img)
    print(f"Flatten size: {model.flatten_size}")
    print(f"Output features shape: {features.shape}")
