from model import DETR
from config import *

import os
import cv2
import glob
import torch
import argparse
import numpy as np

def scale_bbox(iw, ih, bbox):
    y1, x1, h, w = np.hsplit(bbox, 4)
    x2 = x1 + w
    y2 = y1 + h

    x1 = np.squeeze(x1 * iw)
    y1 = np.squeeze(y1 * ih)
    x2 = np.squeeze(x2 * iw)
    y2 = np.squeeze(y2 * ih)

    return np.stack((x1, y1, x2, y2), axis=-1).astype(np.int32)

def transform(image):
    image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image = torch.from_numpy(image)
    image = image.permute(2, 1, 0)
    return image

def run_inference_for_single_image(image, model, device):
    image = image.to(device)
    image = [image]
    model.eval()
    model.to(device)
    cpu_device = torch.device("cpu")
    
    with torch.no_grad():
        outputs = model(image)
        
    outputs = [{k: v.to(cpu_device) for k, v in outputs.items()}] 
    
    y_pred = outputs[0]['pred_logits'][0].softmax(1).detach().cpu().numpy()[:, 0]
    b_pred = outputs[0]['pred_boxes'][0].detach().cpu().numpy()
    
    return y_pred, b_pred

def draw(image, y_pred, b_pred, confidence):
    for i in range(y_pred.shape[0]):
        if y_pred[i] > confidence:
            x1, y1, x2, y2 = b_pred[i]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--model', 
                        help='Enter model name from saved_models',
                        required=True)
    
    parser.add_argument('-f', '--folder', 
                        help='Path to the folder having test images',
                        required=True)

    args = parser.parse_args()
    model_name = args.model
    test_images = args.folder

    model_path = f"saved_models/{model_name}"
    model = DETR(num_classes=num_classes,num_queries=num_queries)
    model.load_state_dict(torch.load(model_path))   

    test_path = f'{test_images}/*'
    out_path = 'samples'

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    confidence = 0.5

    for i, image_path in enumerate(glob.glob(test_path)):
        orig_image = cv2.imread(image_path)
        h, w, _ = orig_image.shape
        
        transformed_image = transform(orig_image)
        y_pred, b_pred = run_inference_for_single_image(transformed_image, model=model, device=torch.device('cuda'))        
        
        b_pred = scale_bbox(w, h, b_pred)
        out_image = draw(orig_image, y_pred, b_pred, confidence)
        cv2.imwrite(f'samples/{i}.jpg', out_image)        
    