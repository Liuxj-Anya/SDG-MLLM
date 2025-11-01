import cv2
import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

folder_path = '/data2/liuxj/1-mcabsa/test/MCABSA_testset/vid'
jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]

# 加载模型和处理器
ckpt = "/data2/liuxj/1-Sentiment-mllm/model_train/google/siglip2-giant-opt-patch16-384"
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)

# 你已有的图像抽取函数
def extract(image_path):
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.vision_model(**inputs)
        patch_embeddings = outputs.last_hidden_state  # [1, num_patches+1, hidden_dim]

    return patch_embeddings  # 去除 CLS token -> [num_patches, hidden_dim]

def extract_video_feature_mean(video_path: str, output_path: str, every_n_frames: int = 5, temp_dir="temp_frames"):
    os.makedirs(temp_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_idx = 0
    all_patches = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n_frames == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_path = os.path.join(temp_dir, f"frame_{saved_idx}.jpg")
            Image.fromarray(frame_rgb).save(img_path)

            patch_embeds = extract(img_path)  # 使用 image path 进行抽取
            all_patches.append(patch_embeds)
            saved_idx += 1
        frame_idx += 1
    cap.release()

    if not all_patches:
        raise ValueError("没有采样到帧，无法提取特征。")

    # 所有帧的 patch 拼接 -> mean pooling
    all_patches_tensor = torch.cat(all_patches, dim=0)
    mean_feature = all_patches_tensor.mean(dim=0)  # shape: [hidden_dim]
    print(mean_feature.shape)

    np.save(output_path, mean_feature.cpu().numpy())
    print(f"视频特征保存到: {output_path}")

    # 删除临时图片
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)

#extract_video_feature_mean("/data2/liuxj/1-mcabsa/data/vid/exp_art_flip_vid_0001.mp4", "/data2/liuxj/1-mcabsa/data/mod_feature/vid/test_video_feat.npy")
#extract_video_feature_mean("/data2/liuxj/1-mcabsa/vid_0001.mp4","/data2/liuxj/1-mcabsa/vid_0001.npy")
for i,jpg_name in enumerate(jpg_files):
    name = jpg_name.split('.')[0]
    print(name)
    vid_path = os.path.join(folder_path,jpg_name)
    output_path=os.path.join('/data2/liuxj/1-mcabsa/test/mod_feature/vid',name+'.npy')
    extract_video_feature_mean(vid_path,output_path)