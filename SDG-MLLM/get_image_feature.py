from transformers import AutoModel, AutoProcessor
import torch
from transformers.image_utils import load_image
import os
import numpy as np
from PIL import Image
import imghdr
def can_read_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证图片完整性，但不加载图片到内存中
        return True
    except IOError:
        print(f"无法读取文件: {file_path}")
        return False

folder_path = '/data2/liuxj/1-mcabsa/test/MCABSA_testset/img'
jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
#print(jpg_files)

# 加载模型和处理器
ckpt = "/data2/liuxj/1-Sentiment-mllm/model_train/google/siglip2-giant-opt-patch16-384"
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)

def extract(image_path,output_path):
    # 加载图像并预处理
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 前向传播，获取所有 patch 特征
    with torch.no_grad():
        outputs = model.vision_model(**inputs)  # 只经过视觉 encoder
        patch_embeddings = outputs.last_hidden_state  # shape: [1, num_patches+1, hidden_dim]

    np.save(output_path, patch_embeddings.cpu().numpy())


for i,jpg_name in enumerate(jpg_files):

    name = jpg_name.split('.')[0]
    print(name)
    image_path = os.path.join(folder_path,jpg_name)
    # if not can_read_image(image_path):
    #     continue
    # file_type = imghdr.what(image_path)
    # if file_type == None:
    #     continue
    output_path=os.path.join('/data2/liuxj/1-mcabsa/test/mod_feature/img',name+'.npy')
    extract(image_path,output_path)


