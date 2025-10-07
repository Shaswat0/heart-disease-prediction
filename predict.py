# predict.py

"""
Standalone predict script for CLI inference.
"""

import argparse
import torch
from utils.data_preprocessing import preprocess_input
from utils.config import MODEL_PATH, DEVICE, INPUT_DIM
from server.model import get_global_model

def predict(user_data, image_path=None):
    if not user_data or len(user_data) != INPUT_DIM:
        raise ValueError(f"user_data must be a list of length {INPUT_DIM}. Received length {len(user_data)}.")
    model = get_global_model(tab_in=INPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    data = preprocess_input(user_data, image_path=image_path)
    tab = data["tabular"]
    img = data["image"]
    with torch.no_grad():
        out = model(tab.to(DEVICE), img.to(DEVICE))
        pred = torch.argmax(out, dim=1).item()
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    print("Prediction:", "Heart Disease" if pred==1 else "Healthy")
    print(f"Confidence: Healthy={probs[0]:.4f}, HeartDisease={probs[1]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", type=float,
                        help="Provide tabular features as space-separated values in the CSV column order.")
    parser.add_argument("--image", type=str, default=None, help="Optional ECG image path")
    args = parser.parse_args()
    if args.features is None:
        # example default (13 features)
        example = [63,1,3,145,233,1,0,150,0,2.3,0,0,1]
        print("No features provided. Using example:", example)
        predict(example, image_path=args.image)
    else:
        predict(args.features, image_path=args.image)
