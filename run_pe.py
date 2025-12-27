import os
import sys
import torch
import argparse
from PIL import Image
import decord # Required for video handling

# --- 1. Import Meta's Core Library ---
# We add the current directory to path to ensure we can import 'core'
# sys.path.append(os.getcwd())

# try:
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
# except ImportError:
#     print("Error: Could not import 'core' modules.")
#     print("Make sure you are running this script from the root of the 'perception_models' repository.")
#     sys.exit(1)

# --- 2. Configuration ---
MODEL_NAME = 'PE-Core-L14-336'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    # This automatically downloads weights to ~/.cache/huggingface if not present
    model = pe.CLIP.from_config(MODEL_NAME, pretrained=True)
    model = model.to(DEVICE)
    model.eval()
    
    # Get transforms specifically designed for this model
    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    
    return model, preprocess, tokenizer

def preprocess_video(video_path, num_frames=8, transform=None):
    """
    Reads video, samples frames uniformly, and applies transforms.
    Based on the official demo code.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Decord is much faster than OpenCV for random access
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    
    # Uniform sampling
    frame_indices = [int(i * (total_frames / num_frames)) for i in range(num_frames)]
    frames = vr.get_batch(frame_indices).asnumpy()
    
    # Apply CLIP/PE transform to every frame
    preprocessed_frames = [transform(Image.fromarray(frame)) for frame in frames]
    
    # Stack into [T, C, H, W]
    return torch.stack(preprocessed_frames, dim=0)

def run_inference(file_path, queries, mode='image'):
    model, preprocess, tokenizer = load_model()

    # Prepare Text Queries
    print(f"Processing queries: {queries}")
    text_tokens = tokenizer(queries).to(DEVICE)

    with torch.no_grad():
        # Encode Text
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Encode Media (Image or Video)
        if mode == 'video':
            print(f"Processing video: {file_path}")
            # PE-Core typically uses 8 frames for video inference
            video_tensor = preprocess_video(file_path, num_frames=8, transform=preprocess)
            video_tensor = video_tensor.unsqueeze(0).to(DEVICE) # Add batch dim [1, T, C, H, W]
            
            # Use specific video encoder method
            media_features = model.encode_video(video_tensor)
        
        else: # Image
            print(f"Processing image: {file_path}")
            image = Image.open(file_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
            
            # Use specific image encoder method
            media_features = model.encode_image(image_tensor)

        # Normalize Media Features
        media_features /= media_features.norm(dim=-1, keepdim=True)

        # Calculate Similarity
        # Shape: [1, num_queries]
        probs = (100.0 * media_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

    # Display Results
    print("\n--- Results ---")
    results = list(zip(queries, probs))
    # Sort by probability descending
    results.sort(key=lambda x: x[1], reverse=True)

    for query, score in results:
        print(f"{query:<25}: {score:.4f} ({score*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PE-Core-L14-336 Inference")
    parser.add_argument("file_path", type=str, help="Path to image or video file")
    parser.add_argument("--type", type=str, choices=['image', 'video'], default='image', help="Type of input file")
    parser.add_argument("--queries", type=str, nargs='+', required=True, help="List of text queries to check against")
    
    args = parser.parse_args()
    
    run_inference(args.file_path, args.queries, args.type)
