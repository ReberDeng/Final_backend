import base64
import logging
import os
import time
from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("dogs.tsv", sep='\t')
model = YOLO("best.pt")


def get_dog_info(class_id: int) -> Tuple[str, str]:
    row = df[df['class_id'] == class_id]

    if not row.empty:
        # 返回对应的Brief Introduction
        brief_name = row.iloc[0]['Breed']
        brief_introduction = row.iloc[0]['Brief Introduction']
        return brief_name, brief_introduction

    return "N/A", "N/A"


def image_to_base64(image_path):
    # 打开图片
    with Image.open(image_path) as img:
        # 使用BytesIO将图片内容读入字节流
        buffered = BytesIO()
        img.save(buffered, format=img.format)

        # 获取图片格式
        img_format = img.format.lower()
        if img_format == 'jpeg':
            img_format = 'jpg'

        # 编码为base64
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 构建base64字符串，包含文件头
        img_base64 = f"data:image/{img_format};base64,{img_str}"

        return img_base64


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/api/dogs/detect")
async def dog_detect(image: UploadFile):
    logger.info("Received dog detection request")
    image_bytes = image.file.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img)
    print(results)
    # 将 NumPy 数组转换为 PIL Image 对象
    image = Image.fromarray(img)
    output_dir = "cache"
    output = []
    # 遍历检测结果
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # 获取边界框坐标
        scores = result.boxes.conf.cpu().numpy()  # 获取置信度
        classes = result.boxes.cls.cpu().numpy()  # 获取类别索引

        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box)

            # 裁剪检测到的目标
            cropped_image = image.crop((x1, y1, x2, y2))

            # 获取分类名称
            class_name = model.names[int(cls)]
            # 生成文件名
            output_path = os.path.join(output_dir, f"{class_name}_{i}_{int(round(time.time() * 1000))}.jpg")

            # 保存裁剪后的图像
            cropped_image.save(output_path)
            output.append({
                'name': class_name,
                'id': int(cls),
                "img": image_to_base64(output_path)
            })

            print(f"Saved {class_name} to {output_path}")
    return output


@app.get("/api/dogs/{id}")
async def dog_get(id: int):
    brief_name, brief_introduction = get_dog_info(id)
    if brief_name == "N/A":
        raise HTTPException(status_code=404, detail="Dog not found")
    return {"id": id, "detail": brief_introduction, "breed": brief_name}


@app.get("/api/articles/{id}")
async def get_article(id: int):
    if id in range(1, 3):
        context = open(os.path.join("pages", f"{id}.txt")).readlines()
        article = {
            "id": id,
            "title": "Dog breeds alaskan malamute" if id == 1 else "Dog breeds poodle",
            "body": context
        }
        return article
    raise HTTPException(status_code=404, detail="Article not found")


@app.get("/api/articles")
async def get_articles():
    articles = [
        {"id": 1, "title": "Dog breeds alaskan malamute", "image_url": "https://ice.frostsky.com/2024/08/18/7ba49a30d66838f59b46943b525a9c66.jpeg"},
        {"id": 2, "title": "Dog breeds poodle", "image_url": "https://ice.frostsky.com/2024/08/18/300d7978291c618f4087a258342e83d8.jpeg"},
    ]
    return articles
