# sleep_federated/models/cnn_gru.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        # Add SE block
        self.se = SEBlock(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Apply SE block
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(x), dim=1)
        context = torch.sum(attention_weights * x, dim=1)
        return context, attention_weights

class CNNGRU(nn.Module):
    def __init__(self, input_channels=1, num_classes=5, hidden_size=128):
        super(CNNGRU, self).__init__()
        
        # Initial convolutional layer
        self.conv_init = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        
        # Residual blocks with SE
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            nn.Dropout(0.3)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            nn.Dropout(0.4)
        )
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size * 2)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input shape handling
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)
        
        # CNN feature extraction
        x = self.conv_init(x)        # Initial convolution
        x = self.layer1(x)           # First residual block
        x = self.layer2(x)           # Second residual block
        x = self.layer3(x)           # Third residual block
        
        # Prepare for GRU (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        
        # Apply GRU
        x, _ = self.gru(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Apply attention
        x, attention_weights = self.attention(x)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_attention_weights(self, x):
        """Get attention weights for visualization."""
        self.eval()
        with torch.no_grad():
            # Input shape handling
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            elif len(x.shape) == 3 and x.shape[1] > x.shape[2]:
                x = x.transpose(1, 2)
            
            # Forward pass until attention
            x = self.conv_init(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = x.transpose(1, 2)
            x, _ = self.gru(x)
            x = self.layer_norm(x)
            _, attention_weights = self.attention(x)
            
            return attention_weights