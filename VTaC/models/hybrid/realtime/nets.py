import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridCAVN(nn.Module):
    def __init__(self, cnn_backbone, num_context_features=4):
        """
        Args:
            cnn_backbone: The pre-trained or initialized CNN model (e.g., CNNClassifier)
            num_context_features: Number of statistical features (kurtosis, skew, etc.)
        """
        super(HybridCAVN, self).__init__()
        self.cnn = cnn_backbone
        
        # Remove the final classification layer of the CNN if necessary
        # Assuming cnn_backbone output is [Batch, 1] (logits), we use it as a feature.
        # Ideally, we'd take the layer *before* the final logit, but taking the logit 
        # as a high-level feature is also a valid valid strategy for late fusion.
        
        # Context Branch: Process statistical features
        self.context_net = nn.Sequential(
            nn.Linear(num_context_features, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        
        # Fusion Layer: CNN output (1 dim) + Context output (4 dims)
        self.fusion = nn.Sequential(
            nn.Linear(1 + 4, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1) # Final Logit
        )

    def forward(self, x, context):
        # 1. Get CNN features (Signal path)
        cnn_out = self.cnn(x) # [Batch, 1]
        
        # 2. Get Context features (Statistics path)
        context_out = self.context_net(context) # [Batch, 4]
        
        # 3. Increase context weighting to force model to listen to context features
        context_out = context_out * 2.0
        
        # 4. Concatenate
        combined = torch.cat((cnn_out, context_out), dim=1)
        
        # 5. Final Prediction
        return self.fusion(combined)