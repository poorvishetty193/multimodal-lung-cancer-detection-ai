# ml-service/audio_pipeline/crnn_model.py
"""
CRNN model for audio (cough + breath) classification.
Input: log-mel spectrogram (n_mels x time)
Model: CNN feature extractor -> BiLSTM -> FC classifier
Output: probability (0-1) or multi-class logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3,3), pool=(2,2)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=(kernel[0]//2, kernel[1]//2), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel, padding=(kernel[0]//2, kernel[1]//2), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool)
        )
    def forward(self, x):
        return self.conv(x)

class CRNN(nn.Module):
    def __init__(self, n_mels=64, num_classes=1, rnn_hidden=128, rnn_layers=2, dropout=0.3):
        super(CRNN, self).__init__()
        # Expect input shape: (B, 1, n_mels, time)
        self.cnn = nn.Sequential(
            ConvBlock(1, 32, kernel=(3,3), pool=(2,2)),
            ConvBlock(32, 64, kernel=(3,3), pool=(2,2)),
            ConvBlock(64, 128, kernel=(3,3), pool=(2,2)),
            ConvBlock(128, 256, kernel=(3,3), pool=(2,2)),
        )
        # After CNN, collapse frequency dimension and feed into RNN across time
        # We'll adaptively compute RNN input size at forward
        self.rnn_hidden = rnn_hidden
        self.rnn = nn.LSTM(input_size=256, hidden_size=rnn_hidden, num_layers=rnn_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        """
        x: (B, 1, n_mels, T)
        """
        b = x.size(0)
        out = self.cnn(x)  # (B, C, n_mels_reduced, T_reduced)
        # collapse frequency axis
        out = out.mean(dim=2)  # (B, C, T_reduced)
        out = out.permute(0, 2, 1)  # (B, T_reduced, C) -> RNN expects seq-first in batch_first=True
        # ensure input_size matches rnn input size; if not, project:
        if out.size(2) != self.rnn.input_size:
            # linear projection
            proj = nn.Linear(out.size(2), self.rnn.input_size).to(out.device)
            out = proj(out)
        rnn_out, _ = self.rnn(out)  # (B, T, 2*hidden)
        # global pooling over time
        pooled = torch.mean(rnn_out, dim=1)  # (B, 2*hidden)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        if self.classifier.out_features == 1:
            return torch.sigmoid(logits).squeeze(1)
        else:
            return F.softmax(logits, dim=1)

# Quick test
if __name__ == "__main__":
    model = CRNN(n_mels=64, num_classes=1)
    dummy = torch.randn(4, 1, 64, 128)  # batch, channel, n_mels, time
    out = model(dummy)
    print("Output shape:", out.shape)
