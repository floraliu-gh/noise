import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from thop import profile
from torchinfo import summary
import time
import copy  # 引入 copy 模組用來產生替身模型

from client_model import ClientModel
from server_model import ServerModel
from client1 import Client    
from server1 import MainServer  
from robust_aggregation import fedserver
from channel_simulation import CommunicationChannel, PixelNoiseInjector, Denoiser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

K = 5            # Client 數量
rounds = 20      # 總通訊輪數
local_epochs = 5 # 每一輪在本地要跑幾個 Epoch
batch_size = 128
lr = 0.002       

# === 雜訊與去噪參數設定 ===
ENABLE_CHANNEL_NOISE = True      # 是否啟用通道雜訊
ENABLE_PIXEL_NOISE = False       # 是否啟用像素雜訊
ENABLE_DENOISING = True          # 是否啟用去雜訊
DENOISING_METHOD = 'dynamic'     # 去雜訊方法: 'ema', 'dynamic', 'none'
DENOISING_BASE_ALPHA = 0.3       # 去雜訊基礎平滑係數 (若用 dynamic，這是 Alpha 下限)

# 聚合方法
AGGREGATION_METHOD = 'fedavg'  # 'fedavg', 'median', 'trimmed_mean', 'krum'

# 通道參數
SNR_DB = 25                      # 雜訊比 (dB), 越高雜訊越小
CHANNEL_GAIN = 1.0               # 通道增益
BIT_ERROR_RATE = 0.001           # 位元錯誤率
PIXEL_NOISE_STD = 0.05           # 像素雜訊標準差

print(f"\n=== Settings ===")
print(f"Channel noise: {'On' if ENABLE_CHANNEL_NOISE else 'Off'} (SNR={SNR_DB}dB, BER={BIT_ERROR_RATE})")
print(f"Pixel noise: {'On' if ENABLE_PIXEL_NOISE else 'Off'} (std={PIXEL_NOISE_STD})")
print(f"Denoising: {'On' if ENABLE_DENOISING else 'Off'} (method={DENOISING_METHOD}, Base Alpha={DENOISING_BASE_ALPHA})")
print(f"Aggregation method: {AGGREGATION_METHOD}")

print("\nPreparing EuroSAT dataset...")
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
train_dataset_full, test_dataset = random_split(full_dataset, [train_size, test_size])

nk_list = [len(train_dataset_full) // K] * K
nk_list[-1] += len(train_dataset_full) - sum(nk_list)
client_subsets = random_split(train_dataset_full, nk_list)

dataloaders = [DataLoader(sub, batch_size=batch_size, shuffle=True) for sub in client_subsets]
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- 初始化通訊模組 ---
channels = []
pixel_injectors = []

for i in range(K):
    snr_variation = SNR_DB + np.random.randn() * 2 
    
    if ENABLE_CHANNEL_NOISE:
        channel = CommunicationChannel(
            snr_db=snr_variation, 
            channel_gain=CHANNEL_GAIN,
            bit_error_rate=BIT_ERROR_RATE
        )
    else:
        channel = None
    
    pixel_injector = PixelNoiseInjector(noise_std=PIXEL_NOISE_STD) if ENABLE_PIXEL_NOISE else None
    channels.append(channel)
    pixel_injectors.append(pixel_injector)

# Server 端的去雜訊器 (使用動態或固定 Alpha)
if ENABLE_DENOISING:
    denoiser = Denoiser(alpha=DENOISING_BASE_ALPHA, method=DENOISING_METHOD)
else:
    denoiser = None

# --- 初始化物件 ---
global_server_model = ServerModel().to(device)
# 提升 Server 的 LR 解決一輪只更新一次導致的收斂緩慢！
server_lr = 0.1 
main_server = MainServer(global_server_model, device, lr=server_lr, denoiser=denoiser)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(main_server.optimizer, T_max=rounds, eta_min=1e-5)

clients = []
for i in range(K):
    c_model = ClientModel().to(device)
    c_instance = Client(
        model=c_model, 
        device=device, 
        lr=lr,
        channel=channels[i],
        pixel_noise_injector=pixel_injectors[i]
    )
    clients.append(c_instance)

# --- 全域 Client 初始權重同步 (符合演算法 if t=0 then Initialize WC_t ...) ---
print("\nSynchronizing initial client models (t=0) ...")
initial_global_weights = clients[0].model.state_dict()
for c in clients:
    c.model.load_state_dict(initial_global_weights)

# --- 評估函數 ---
def evaluate(c_model, s_model, loader):
    c_model.eval()
    s_model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            activations = c_model(x)
            outputs = s_model(activations)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return correct / total, all_labels, all_preds

print("\n=== Hardware & Performance Analysis Report ===")

# 產生假資料來計算 FLOPs，因為模型切割點已改，我們用 dummy_client_model 先推論一次來取得特徵形狀
dummy_client_input = torch.randn(1, 3, 64, 64).to(device)
with torch.no_grad():
    dummy_server_input = clients[0].model(dummy_client_input)

# 1. 參數數量與記憶體
print("\n[1] Client Model Parameters & Memory:")
summary(clients[0].model, input_size=(1, 3, 64, 64), device=device)

print("\n[2] Server Model Parameters & Memory:")
summary(main_server.model, input_size=dummy_server_input.shape, device=device)

# 2. 運算量估算 (使用替身模型避免污染)
dummy_client_model = copy.deepcopy(clients[0].model)
dummy_server_model = copy.deepcopy(main_server.model)

macs_c, params_c = profile(dummy_client_model, inputs=(dummy_client_input, ), verbose=False)
macs_s, params_s = profile(dummy_server_model, inputs=(dummy_server_input, ), verbose=False)

flops_c = macs_c * 2
flops_s = macs_s * 2

print("\n[3] Computational Complexity (per image):")
print(f" - Client Model FLOPs : {flops_c / 1e6:.2f} Mega-FLOPs (MFLOPs)")
print(f" - Server Model FLOPs : {flops_s / 1e6:.2f} Mega-FLOPs (MFLOPs)")
print(f" - Total System FLOPs : {(flops_c + flops_s) / 1e6:.2f} MFLOPs")

del dummy_client_model
del dummy_server_model

# 3. 延遲時間測試
print("\n[4] Latency Measurement (Running 100 warmups & 100 tests)...")
def measure_latency(model, dummy_input, device, num_tests=100):
    model.eval()
    for _ in range(100):
        _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    for _ in range(num_tests):
        _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    return ((end_time - start_time) / num_tests) * 1000  

with torch.no_grad():
    client_latency = measure_latency(clients[0].model, dummy_client_input, device)
    server_latency = measure_latency(main_server.model, dummy_server_input, device)

print(f" - Client Forward Latency : {client_latency:.3f} ms / image")
print(f" - Server Forward Latency : {server_latency:.3f} ms / image")
print(f" - Total Inference Latency: {(client_latency + server_latency):.3f} ms / image\n")

# --- 訓練迴圈 ---
print(f"\nStart training: Rounds={rounds}, Local Epochs={local_epochs}")
train_losses, test_accuracies = [], []

for r in range(rounds):
    # 如果啟用 Dynamic Alpha，依據當前模擬的 SNR 來進行調整
    current_alpha = DENOISING_BASE_ALPHA
    if ENABLE_DENOISING and denoiser is not None and DENOISING_METHOD == 'dynamic':
        # 取得第一個 Client 模擬的 SNR 作為代表值 (也可以取所有通道的平均)
        avg_snr = np.mean([ch.snr_db for ch in channels if ch is not None]) if ENABLE_CHANNEL_NOISE else 40.0
        current_alpha = denoiser.update_dynamic_alpha(avg_snr, r + 1, rounds)

    print(f'\n--- Round {r+1}/{rounds} (Current Alpha: {current_alpha:.3f}) ---')
    round_loss = 0.0
    total_steps = 0
    
    main_server.optimizer.zero_grad() # 在回合開始前清空 Server 梯度
    
    for i in range(K):
        for epoch in range(local_epochs):
            for batch_idx, (x, y) in enumerate(dataloaders[i]):
                x, y = x.to(device), y.to(device)
                
                # Client Forward
                A_k_received = clients[i].ClientUpdate(x, add_pixel_noise=ENABLE_PIXEL_NOISE)
                                
                if ENABLE_CHANNEL_NOISE:
                    A_k_received = torch.nan_to_num(A_k_received, nan=0.0)
                    
                    # 特徵標準化
                    if A_k_received.shape[0] > 1:
                        if A_k_received.dim() == 4:
                            mean = A_k_received.mean(dim=(1, 2, 3), keepdim=True)
                            std = A_k_received.std(dim=(1, 2, 3), keepdim=True) + 1e-6
                        elif A_k_received.dim() == 2:
                            mean = A_k_received.mean(dim=1, keepdim=True)
                            std = A_k_received.std(dim=1, keepdim=True) + 1e-6
                        else:
                            mean = A_k_received.mean()
                            std = A_k_received.std() + 1e-6
                            
                        A_k_received = (A_k_received - mean) / std

                    # 數值截斷防禦
                    A_k_received = torch.clamp(A_k_received, min=-5.0, max=5.0)
                    A_k_server_input = A_k_received.to(device).detach().clone().requires_grad_(True)
                else:
                    A_k_server_input = A_k_received.to(device).detach().clone().requires_grad_(True)
                
                # Server Update (累積梯度，不立刻 step()，改為回合結束後統一更新)
                dA_k, loss_val = main_server.ServerUpdate(A_k_server_input, y, clear_grad=False)
                
                # Client Backprop (Client 依然接收梯度並在本地更新)
                if ENABLE_CHANNEL_NOISE:
                     dA_k = torch.clamp(dA_k, min=-0.5, max=0.5) 
                     
                clients[i].ClientBackprop(dA_k.clone()) 
                torch.nn.utils.clip_grad_norm_(clients[i].model.parameters(), max_norm=5.0)
                               
                # DEBUG 列印區塊 (觀察梯度健康度)
                if i == 0 and batch_idx == 0:
                    grad_norm = dA_k.abs().mean().item()
                    first_layer_grad = 0
                    for param in clients[i].model.parameters():
                        if param.grad is not None:
                            first_layer_grad = param.grad.abs().mean().item()
                            break
                    print(f"Server回傳梯度={grad_norm:.6f}, Client權重梯度={first_layer_grad:.6f}")

                round_loss += loss_val
                total_steps += 1
            
    avg_loss = round_loss / total_steps if total_steps > 0 else 0
    train_losses.append(avg_loss)
   
    # 【新增】 Server-side model update (對累積的梯度取平均後更新 Server 權重)
    with torch.no_grad():
        for param in main_server.model.parameters():
            if param.grad is not None:
                param.grad /= total_steps # 平均化梯度
    
    torch.nn.utils.clip_grad_norm_(main_server.model.parameters(), max_norm=5.0)
    main_server.step()
   
    # 權重聚合
    client_models_list = [c.model for c in clients]
    aggregated_model = fedserver(
        client_models_list, 
        nk_list, 
        sum(nk_list),
        method=AGGREGATION_METHOD,
        trim_ratio=0.2, 
        f=1 
    )
    global_weights = aggregated_model.state_dict()
    for c in clients:
        c.model.load_state_dict(global_weights)
    
    # 每一輪結束後，重置去雜訊器的 EMA 歷史狀態
    if ENABLE_DENOISING and denoiser is not None:
        denoiser.reset()
        
    # 評估模型
    test_acc, y_true, y_pred = evaluate(clients[0].model, main_server.model, test_loader)
    test_accuracies.append(test_acc)
    
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Current LR: {current_lr:.6f}")
    print(f"Loss: {avg_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")

print(f'\n=== Final Result ===')
print(f'Final Test Accuracy: {test_accuracies[-1] * 100:.2f}%')
print(f'Best Test Accuracy: {max(test_accuracies) * 100:.2f}%')

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Acc: {test_accuracies[-1]*100:.2f}%)')
plt.show()

# Training Loss 
plt.figure()
plt.plot(train_losses, linewidth=2)
plt.title('Training Loss', fontsize=14)
plt.xlabel('Round')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.show() 

# Test Accuracy
plt.figure()
plt.plot(test_accuracies, linewidth=2, color='green')
plt.title('Test Accuracy', fontsize=14)
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)
plt.show()
print("\nFinished!")