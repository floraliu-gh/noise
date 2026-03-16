import torch
import torch.nn as nn

class ServerModel(nn.Module):
    """
    Server 端的模型 (接手後半部的卷積與所有全連接層)
    輸入: [batch, 64, 16, 16] 的 activation
    輸出: 10 個類別的 logits
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 接手原本在 Client 的 Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
            
            # 接手原本在 Client 的吃記憶體怪獸 Flatten and FC
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            
            # 原本 Server 負責的分類層
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  
        )

    def forward(self, A_k):
        return self.net(A_k)