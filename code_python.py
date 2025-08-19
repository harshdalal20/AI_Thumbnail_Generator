import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import clip  # Importing OpenAI model
from transformers import BlipProcessor, BlipForConditionalGeneration # Importing Salesforce model

VIDEO_DIR = "Videos"  # Load videos from Videos folder
THUMBNAIL_DIR = "Thumbnails" # Output is stored in Thumbnails
FRAME_INTERVAL_SEC = 2   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loading CLIP & BLIP model
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

# Frame Extraction
def extract_frames(video_path: str, interval_sec: int = 1) -> list:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * interval_sec)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames

# Sharpness Calculation
def calculate_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# BLIP Captioning
def get_blip_caption(image: Image.Image) -> str:
    inputs = blip_processor(images=image, return_tensors="pt").to(DEVICE)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# CLIP Cosine similarity
def get_clip_similarity(image: Image.Image, text: str) -> float:
    image_input = clip_preprocess(image).unsqueeze(0).to(DEVICE)
    text_input = clip.tokenize([text]).to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).item()
    return similarity

# Save Frame as Image
def save_frame_as_image(frame: np.ndarray, path: str):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img.save(path)

# Process and Select Top 4 from Sharpness & Scene
def process_video(video_path: str):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(THUMBNAIL_DIR, video_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔍 Processing: {video_name}")
    frames = extract_frames(video_path, FRAME_INTERVAL_SEC)
    sharpness_scores = []
    scene_scores = []

    for i, frame in enumerate(frames):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        sharpness = calculate_sharpness(frame)
        caption = get_blip_caption(pil_image)
        scene_score = get_clip_similarity(pil_image, caption)

        sharpness_scores.append((sharpness, frame, i))
        scene_scores.append((scene_score, frame, i))

    # Top 4 sharpest frames
    sharpness_scores.sort(reverse=True, key=lambda x: x[0])
    for rank in range(min(4, len(sharpness_scores))):
        score, frame, idx = sharpness_scores[rank]
        path = os.path.join(output_dir, f"sharp_top{rank+1}_frame{idx:03d}_score{score:.2f}.jpg")
        save_frame_as_image(frame, path)

    # Top 4 scene-detected frames
    scene_scores.sort(reverse=True, key=lambda x: x[0])
    for rank in range(min(4, len(scene_scores))):
        score, frame, idx = scene_scores[rank]
        path = os.path.join(output_dir, f"scene_top{rank+1}_frame{idx:03d}_score{score:.4f}.jpg")
        save_frame_as_image(frame, path)

    print(f"Saved thumbnails for {video_name}")

# Main 
def main():
    os.makedirs(THUMBNAIL_DIR, exist_ok=True)
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4"))]
    for file in tqdm(video_files, desc="Processing Videos"):
        video_path = os.path.join(VIDEO_DIR, file)
        process_video(video_path)

main()

# Built an AI based thumbnail generator that automatically picks the best frames from a video based on sharpness and scene relevance. 
# I used two open-source models: BLIP (by Salesforce) to generate captions for each frame, and CLIP (by OpenAI) to measure how well each frame matches its caption.
# It extracts frames every 2 seconds, scores them using sharpness and scene relevance, and saves the top 4 from each category.
# The result is a set of 8 high-quality Thumbnails per video using 2 models.