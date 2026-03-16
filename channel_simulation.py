import torch
import numpy as np

class CommunicationChannel:
    """
    模擬衛星通訊通道:
    1. Channel Gain (Fading)
    2. AWGN (高斯白雜訊) - 模擬背景雜訊
    3. Bit Error (位元錯誤) - 模擬傳輸錯誤
    """
    def __init__(self, snr_db=20, channel_gain=1.0, bit_error_rate=0.0):
        """
        參數:
        - snr_db: 訊噪比 (Signal-to-Noise Ratio) in dB
        - channel_gain: 通道增益 h (模擬訊號衰減)
        - bit_error_rate: 位元錯誤率 (0.0 to 1.0)
        """
        self.snr_db = snr_db
        self.h = channel_gain
        self.ber = bit_error_rate
        
    def add_awgn_noise(self, x):
        """
        加入加性高斯白雜訊 (Additive White Gaussian Noise)
        公式: P_noise = P_signal / 10^(SNR/10)
        """
        if self.snr_db > 100: # SNR 非常大時直接忽略雜訊
            return x

        # 1. 計算訊號功率 (Mean Square)
        signal_power = torch.mean(x ** 2)
        
        # 2. 根據 SNR 計算雜訊功率
        # 防止除以 0
        if signal_power == 0:
            return x
            
        snr_linear = 10 ** (self.snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        # 3. 生成高斯雜訊 (Mean=0, Std=sqrt(noise_power))
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(x) * noise_std
        
        # 4. 疊加
        return x + noise
    
    def add_bit_errors(self, x, num_bits=8):
        """
        模擬量化後的位元錯誤 (Vectorized 加速版)
        將浮點數量化成整數 -> 隨機翻轉 bits -> 轉回浮點數
        """
        if self.ber <= 0.0:
            return x
        
        # --- 1. 量化 (Quantization) ---
        # 將數值映射到 [0, 2^8 - 1] 的整數區間
        x_min, x_max = x.min(), x.max()
        # 避免除以 0
        if x_max == x_min:
            return x
            
        scale = (2 ** num_bits - 1) / (x_max - x_min)
        x_int = ((x - x_min) * scale).long() # 轉為整數索引
        
        # --- 2. 注入錯誤 (Bit Flipping) ---
        # 這裡使用位元運算加速，取代原本的雙重迴圈
        
        # 產生一個跟 x_int 一樣形狀的 mask，初始為 0
        final_flip_mask = torch.zeros_like(x_int)
        
        # 對每一個 bit position (例如 8-bit 就是 0~7) 獨立判斷
        for b in range(num_bits):
            # 產生隨機機率矩陣
            prob_matrix = torch.rand_like(x, dtype=torch.float32)
            
            # 如果機率 < BER，代表這個位置的第 b 個 bit 要翻轉
            # (prob_matrix < self.ber) 會產生 True/False
            # .long() 轉成 1/0
            flip_decision = (prob_matrix < self.ber).long()
            
            # 將決定要翻轉的 bit 左移到正確位置，並加入 mask
            final_flip_mask = final_flip_mask | (flip_decision << b)
            
        # 使用 XOR 運算進行翻轉 (0^1=1, 1^1=0)
        x_int_corrupted = x_int ^ final_flip_mask
        
        # --- 3. 反量化 (Dequantization) ---
        x_recovered = x_int_corrupted.float() / scale + x_min
        
        return x_recovered
    
    def transmit(self, x, add_awgn=True, add_bit_error=False):
        """
        完整傳輸流程
        參數:
        - x: 輸入訊號 (Tensor)
        - add_awgn: 是否加入高斯雜訊 (背景霧氣)
        - add_bit_error: 是否加入位元錯誤 (脈衝雜訊)
        """
        # 1. 應用通道增益
        y = x * self.h
        
        # 2. (可選) 加入 AWGN
        if add_awgn:
            y = self.add_awgn_noise(y)
            
        # 3. (可選) 加入 Bit Error
        # 注意: 這裡的順序是先加背景雜訊，再發生位元解碼錯誤，符合物理邏輯
        if add_bit_error and self.ber > 0:
            y = self.add_bit_errors(y)
        
        return y


class PixelNoiseInjector:
    """
    在影像像素層面加入高斯雜訊 (Input Noise)
    """
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std
    
    def add_noise(self, images):
        if self.noise_std <= 0:
            return images
            
        noise = torch.randn_like(images) * self.noise_std
        noisy_images = images + noise
        
        # Clamp 到合理範圍 (假設已 Normalize 到標準常態分佈，限制在 -3~3 之間避免極端值)
        noisy_images = torch.clamp(noisy_images, -3.0, 3.0)
        
        return noisy_images


class Denoiser:
    """
    簡單的去雜訊器 (EMA 濾波)
    """
    def __init__(self, alpha=0.3, method='ema'):
        self.base_alpha = alpha
        self.alpha = alpha
        self.method = method
        self.prev_signal = None
        self.prev_shape = None
        
    def update_dynamic_alpha(self, current_snr, current_round, total_rounds):
        """
        根據當前 SNR 和訓練進度動態調整 alpha
        alpha 越大代表越信任新特徵，越小代表越依賴歷史平滑
        """
        if self.method != 'dynamic':
            return self.alpha
            
        # 1. SNR 基礎信任度 (SNR 越高越信任新特徵)
        # 假設極好訊號為 40dB, 極差為 0dB
        snr_factor = max(0.0, min(1.0, current_snr / 40.0))
        
        # 2. 訓練進度加成 (越到後期越需要更新鮮的梯度，越信任新特徵)
        progress_factor = current_round / max(1, total_rounds)
        
        # 3. 組合公式
        # Base Alpha + SNR 帶來的提升 (最高 0.4) + 進度帶來的提升 (最高 0.3)
        dynamic_a = self.base_alpha + (snr_factor * 0.4) + (progress_factor * 0.3)
        
        # 限幅在 [0.1, 1.0] 之間
        self.alpha = max(0.1, min(1.0, dynamic_a))
        return self.alpha
    
    def denoise(self, noisy_signal):
        if self.method == 'none':
            return noisy_signal
        
        current_shape = noisy_signal.shape
        
        # 如果是第一次或 batch size 改變 (例如最後一個 batch)，重置狀態
        if self.prev_signal is None or current_shape != self.prev_shape:
            self.prev_signal = noisy_signal.detach().clone()
            self.prev_shape = current_shape
            return noisy_signal
        
        # EMA 公式: Output = α * New + (1-α) * Old
        # alpha 越小，平滑效果越強 (但也越容易有殘影)
        denoised = self.alpha * noisy_signal + (1 - self.alpha) * self.prev_signal
        
        # 更新狀態 (必須 detach 以免累積梯度圖)
        self.prev_signal = denoised.detach().clone()
        
        return denoised
    
    def reset(self):
        """每回合 (Round) 開始時呼叫"""
        self.prev_signal = None
        self.prev_shape = None