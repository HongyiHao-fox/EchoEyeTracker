import numpy as np
import scipy.signal
from collections import deque


class GazeTrakProcessor:
    def __init__(
        self,
        fs=50000,
        chirp_len=600,
        f_start=18000,
        f_end=21000,
        buffer_size=26,
        band_low=18000,
        band_high=21000,
        roi_len=100,
        center_crop_len=70,
    ):
        """
        初始化 GazeTrak 处理器
        
        参数:
            fs (int): 采样率 (默认 50kHz)
            chirp_len (int): 单帧样本数 (默认 600)
            f_start (int): 起始频率 (Hz)
            f_end (int): 截止频率 (Hz)
            buffer_size (int): 2D 矩阵的时间维度大小 (堆叠的历史帧数)
        """
        self.fs = fs
        self.chirp_len = chirp_len
        self.f_start = f_start
        self.f_end = f_end
        self.band_low = band_low
        self.band_high = band_high
        self.roi_len = roi_len
        self.center_crop_len = center_crop_len
        self.buffer_size = buffer_size
        
        # 1. 生成 FMCW 参考信号 (Chirp)
        t = np.linspace(0, self.chirp_len / self.fs, self.chirp_len, endpoint=False)
        self.ref_chirp = scipy.signal.chirp(t, f0=self.f_start, f1=self.f_end, t1=t[-1], method='linear')
        
        # 2. 设计 4 阶巴特沃斯带通滤波器
        nyquist = 0.5 * self.fs
        low = self.band_low / nyquist
        high = min(self.band_high, nyquist - 1) / nyquist
        self.b, self.a = scipy.signal.butter(4, [low, high], btype='band')
        
        # 初始化回声图缓冲区 (用于生成 2D 矩阵)
        self.echo_buffer = deque(maxlen=self.buffer_size)

    def process_frame(self, raw_audio_frame):
        """
        处理单帧音频数据
        
        参数:
            raw_audio_frame (np.array): 长度为 600 的原始音频数据
            
        返回:
            np.array: 处理后的 2D 矩阵 (buffer_size x 70), 如果缓冲区未满则返回 None
        """
        if len(raw_audio_frame) != self.chirp_len:
            raise ValueError(f"输入数据长度必须为 {self.chirp_len}，当前为 {len(raw_audio_frame)}")

        # A. 带通滤波 (去除环境噪音)
        filtered_signal = scipy.signal.filtfilt(self.b, self.a, raw_audio_frame)

        # B. 互相关 (Cross-correlation) 生成回声轮廓
        # 使用 'valid' 模式通常用于滤波，但在匹配滤波中，我们通常想要完整的卷积
        # 这里使用 'full' 模式，然后取正滞后部分
        correlation = scipy.signal.correlate(filtered_signal, self.ref_chirp, mode='full')
        
        # 提取回声轮廓 (只取正相关的部分，即回声滞后部分)
        # correlation 长度为 2 * M - 1. 中心点在 M-1.
        # 我们对滞后感兴趣，对应索引 [M-1:]
        lags = correlation[self.chirp_len - 1:]
        
        # 取包络 (Envelope) - 使用希尔伯特变换的模
        analytic_signal = scipy.signal.hilbert(lags)
        echo_profile = np.abs(analytic_signal)

        # C. 数据裁剪：先取 ROI，再从 ROI 中取中心 70 点
        roi_profile = echo_profile[:self.roi_len]
        if len(roi_profile) < self.center_crop_len:
            raise ValueError(
                f"ROI length {len(roi_profile)} is smaller than center crop {self.center_crop_len}."
            )
        crop_start = (len(roi_profile) - self.center_crop_len) // 2
        cropped_profile = roi_profile[crop_start : crop_start + self.center_crop_len]

        # 更新缓冲区
        self.echo_buffer.append(cropped_profile)

        # 检查缓冲区是否已满
        if len(self.echo_buffer) < self.buffer_size:
            return None

        # D. 生成 2D 矩阵并标准化
        # 堆叠成 (Time, Range) -> (26, 70)
        matrix_2d = np.array(self.echo_buffer)
        
        # 标准化 (Min-Max Normalization to 0-1 range)
        # 针对整个矩阵进行归一化，保留相对幅度差异
        min_val = np.min(matrix_2d)
        max_val = np.max(matrix_2d)
        
        if max_val - min_val > 1e-6:
            normalized_matrix = (matrix_2d - min_val) / (max_val - min_val)
        else:
            normalized_matrix = np.zeros_like(matrix_2d)
            
        return normalized_matrix

if __name__ == "__main__":
    # 简单测试代码
    processor = GazeTrakProcessor()
    
    # 模拟输入数据 (静音 + 随机噪声)
    dummy_input = np.random.normal(0, 0.1, 600) 
    
    # 填充缓冲区
    for i in range(25):
        result = processor.process_frame(dummy_input)
        if result is not None:
            print(f"Frame {i}: Generated 2D matrix shape: {result.shape}")
