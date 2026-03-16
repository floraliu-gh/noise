import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from thop import profile
from torchinfo import summary
import time
import copy
import numpy as np

from client_model import ClientModel
from server_model import ServerModel
from channel_simulation import CommunicationChannel, PixelNoiseInjector, Denoiser

# === 傳輸環境與雜訊設定 (對齊 SFL) ===
SNR_DB = 25                      # 訊噪比
BIT_ERROR_RATE = 0.001           # 誤碼率
ENABLE_CHANNEL_NOISE = True      # 是否啟用通道雜訊
ENABLE_PIXEL_NOISE = False       # 是否啟用像素雜訊
PIXEL_NOISE_STD = 0.05           # 像素雜訊強度
ENABLE_DENOISING = True          # 是否啟用去雜訊
DENOISING_METHOD = 'dynamic'     # 去雜訊方法: 'ema', 'dynamic', 'none'
DENOISING_BASE_ALPHA = 0.3       # 去雜訊基礎平滑係數

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# === 訓練參數設定 ===
epochs = 100 
batch_size = 128
lr = 0.002       

# --- 建立集中式模型 (包含雜訊通道) ---
class NoisyCentralizedModel(nn.Module):
    def __init__(self, channel=None, denoiser=None, pixel_injector=None):
        super().__init__()
        self.client_part = ClientModel()
        self.server_part = ServerModel()
        self.channel = channel
        self.denoiser = denoiser
        self.pixel_injector = pixel_injector

    def forward(self, x):
        # 1. Pixel Noise
        if self.pixel_injector is not None:
             x = self.pixel_injector.inject_noise(x)
             
        # 2. Client 特徵萃取
        activations = self.client_part(x)
        
        # 3. Channel Noise 模擬傳輸
        if self.channel is not None:
            activations = self.channel.transmit(activations)
            activations = torch.nan_to_num(activations, nan=0.0)
            
            # 特徵標準化 (對齊 SFL 的防禦機制)
            if activations.shape[0] > 1:
                mean = activations.mean(dim=(1, 2, 3), keepdim=True)
                std = activations.std(dim=(1, 2, 3), keepdim=True) + 1e-6
                activations = (activations - mean) / std
            activations = torch.clamp(activations, min=-5.0, max=5.0)

        # 4. Denoiser 去雜訊
        if self.denoiser is not None:
            activations = self.denoiser.denoise(activations)
            
        # 5. Server 決策分類
        outputs = self.server_part(activations)
        return outputs

# 初始化雜訊模組
channel = CommunicationChannel(snr_db=SNR_DB, bit_error_rate=BIT_ERROR_RATE) if ENABLE_CHANNEL_NOISE else None
denoiser = Denoiser(alpha=DENOISING_BASE_ALPHA, method=DENOISING_METHOD) if ENABLE_DENOISING else None
pixel_injector = PixelNoiseInjector(noise_std=PIXEL_NOISE_STD) if ENABLE_PIXEL_NOISE else None

model = NoisyCentralizedModel(channel=channel, denoiser=denoiser, pixel_injector=pixel_injector).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

print("\nPreparing EuroSAT dataset (Centralized)...")
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.344, 0.380, 0.407], std=[0.203, 0.136, 0.114])])

full_dataset = datasets.EuroSAT('./data', download=True, transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\n=== Hardware & Performance Analysis Report (Centralized) ===")
dummy_input = torch.randn(1, 3, 64, 64).to(device)

# 1. 參數數量 & 記憶體估算
print("\n[1] Centralized Model Parameters & Memory:")
summary(model, input_size=(1, 3, 64, 64), device=device)

# 2. 運算量估算 (FLOPs)
dummy_model = copy.deepcopy(model)
macs, params = profile(dummy_model, inputs=(dummy_input, ), verbose=False)
flops = macs * 2
print("\n[2] Computational Complexity (per image):")
print(f" - Centralized Model FLOPs : {flops / 1e6:.2f} Mega-FLOPs (MFLOPs)")
del dummy_model

# 3. 延遲時間測試 (Latency)
print("\n[3] Latency Measurement (Running 100 warmups & 100 tests)...")
def measure_latency(target_model, test_input, dev, num_tests=100):
    target_model.eval()
    for _ in range(100):
        _ = target_model(test_input)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    for _ in range(num_tests):
        _ = target_model(test_input)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    return ((end_time - start_time) / num_tests) * 1000

with torch.no_grad():
    centralized_latency = measure_latency(model, dummy_input, device)
print(f" - Total Inference Latency: {centralized_latency:.3f} ms / image\n")

# --- 訓練迴圈 ---
print(f"\nStart training Centralized Baseline: Epochs={epochs}")
train_losses, test_accuracies = [], []

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return correct / total, all_labels, all_preds

for epoch in range(epochs):
    # Dynamic Alpha 更新邏輯
    current_alpha = DENOISING_BASE_ALPHA
    if ENABLE_DENOISING and denoiser is not None and DENOISING_METHOD == 'dynamic':
        current_alpha = denoiser.update_dynamic_alpha(SNR_DB if ENABLE_CHANNEL_NOISE else 40.0, epoch + 1, epochs)
        
    model.train()
    round_loss = 0.0
    total_steps = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        round_loss += loss.item()
        total_steps += 1
        
    avg_loss = round_loss / total_steps
    train_losses.append(avg_loss)
    
    # 評估
    test_acc, y_true, y_pred = evaluate(model, test_loader)
    test_accuracies.append(test_acc)
    
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    
    alpha_str = f" - Alpha: {current_alpha:.3f}" if ENABLE_DENOISING and DENOISING_METHOD == 'dynamic' else ""
    print(f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f}{alpha_str} - Loss: {avg_loss:.4f} - Test Accuracy: {test_acc*100:.2f}%")

print(f'\n=== Final Result (Centralized) ===')
print(f'Final Test Accuracy: {test_accuracies[-1] * 100:.2f}%')
print(f'Best Test Accuracy: {max(test_accuracies) * 100:.2f}%')

# Training Loss 
plt.figure()
plt.plot(train_losses, linewidth=2, label='Centralized Baseline')
plt.title('Training Loss', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show() 

# Test Accuracy
plt.figure()
plt.plot(test_accuracies, linewidth=2, color='green', label='Centralized Baseline')
plt.title('Test Accuracy', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print("\nFinished!")