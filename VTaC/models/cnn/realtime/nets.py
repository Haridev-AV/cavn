import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(
        self,
        inputs,
        window_sizes=[25, 50, 100, 150],
        feature_size=64,
        window=90000,
        hidden_signal=128,
        hidden_alarm=64,
        dropout=0.0,
    ):
        super(CNNClassifier, self).__init__()

        self.window = window
        self.hidden_signal = hidden_signal
        self.hidden_alarm = hidden_alarm
        # Use reasonable dropout for conv layers (max 0.2, default 0.05 for stability)
        # Lower dropout in conv layers to prevent gradient instability
        conv_dropout = min(dropout * 0.5, 0.15) if dropout > 0 else 0.05
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(p=conv_dropout),
                    nn.Conv1d(inputs, feature_size, kernel_size=h, stride=5, padding=1),
                    nn.BatchNorm1d(feature_size),
                    nn.ReLU(),
                    nn.Conv1d(
                        feature_size, feature_size, kernel_size=h, stride=5, padding=1
                    ),
                    nn.BatchNorm1d(feature_size),
                    nn.AdaptiveMaxPool1d(1),
                )
                for h in window_sizes
            ]
        )

        self.signal_feature = nn.Sequential(
            nn.Linear(feature_size * len(window_sizes), self.hidden_signal),
            nn.BatchNorm1d(self.hidden_signal),
            nn.ReLU(),
        )

        self.rule_based_label = nn.Sequential(
            nn.Linear(1, self.hidden_alarm),
            nn.BatchNorm1d(self.hidden_alarm),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_signal, 1),
        )
        
        # Initialize weights properly to prevent NaN
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, signal, random_s=None):
        # Apply parallel convolutions with different kernel sizes
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(signal)  # Each conv block outputs [batch, feature_size, 1]
            conv_outputs.append(conv_out)

        # Concatenate along feature dimension: [batch, feature_size * num_convs, 1]
        signal = torch.cat(conv_outputs, dim=1)

        # Flatten to [batch, feature_size * num_convs] for fully connected layers
        signal = signal.view(signal.size(0), -1)
        s_f = self.signal_feature(signal)

        if random_s is not None:
            # Apply same parallel convolutions to random segment
            random_conv_outputs = []
            for conv in self.convs:
                random_conv_out = conv(random_s)
                random_conv_outputs.append(random_conv_out)

            random_s = torch.cat(random_conv_outputs, dim=1)
            random_s = random_s.view(random_s.size(0), -1)
            random_s = self.signal_feature(random_s)

            return self.classifier(s_f), s_f, random_s

        return self.classifier(s_f)
