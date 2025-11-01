from transformers import Wav2Vec2FeatureExtractor, WavLMModel
import torchaudio
import torch
import numpy as np
import os
folder_path = '/data2/liuxj/1-mcabsa/test/MCABSA_testset/aud'
wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]

# ✅ 加载模型和特征提取器
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/data2/liuxj/1-mcabsa/microsoftwavlm-base")
model = WavLMModel.from_pretrained("/data2/liuxj/1-mcabsa/microsoftwavlm-base")
model.eval()

def extract(image_path,output_path):
    # ✅ 读取音频
    waveform, sr = torchaudio.load(image_path)  # waveform: [channels, time]
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # ✅ 使用第一通道，转为 [time]
    waveform = waveform[0]  # → shape: [time]

    # ✅ 输入 feature_extractor（注意不要自己 unsqueeze 太多维度）
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")

    # ✅ 输入模型
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden_size]

    np.save(output_path, last_hidden.cpu().numpy())

extract('/data2/liuxj/1-mcabsa/test/MCABSA_testset/aud/aud_0237.wav','/data2/liuxj/1-mcabsa/aud_0237.npy')
# for i,jpg_name in enumerate(wav_files):
#
#     name = jpg_name.split('.')[0]
#     print(name)
#     image_path = os.path.join(folder_path,jpg_name)
#     output_path=os.path.join('/data2/liuxj/1-mcabsa/test/mod_feature/aud',name+'.npy')
#     extract(image_path,output_path)
