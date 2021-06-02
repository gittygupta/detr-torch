import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

DIR_TRAIN = 'dataset/train'

def x1y1x2y2_to_x1y1wh(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h

def x1y1wh_to_x1y1x2y2(x1, y1, w, h):
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2

def normalize_bbox(bbox, iw, ih):
    x, y, w, h = bbox
    x = x / iw
    y = y / ih
    w = w / iw
    h = h / ih
    return (x, y, w, h)

def normalize_bboxes(bboxes, iw, ih):
    return [normalize_bbox(bbox, iw, ih) for bbox in bboxes]

def get_train_transforms():
    return A.Compose([A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.9),
                            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)],p=0.9),
                            A.ToGray(p=0.01),
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.Resize(height=512, width=512, p=1),
                            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
                            ToTensorV2(p=1.0)],
                        p=1.0,
                        bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])
                        )

def get_valid_transforms():
    return A.Compose([A.Resize(height=512, width=512, p=1.0),
                      ToTensorV2(p=1.0)], 
                      p=1.0, 
                      bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])
                      )

class DroneDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, dataframe, transforms=None):
        self.filenames = filenames
        self.df = dataframe
        self.transforms = transforms
        
        
    def __len__(self) -> int:
        return self.filenames.shape[0]
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        records = self.df[self.df['filename'] == filename]

        image = cv2.imread(f'{DIR_TRAIN}/{filename}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        # DETR takes in data in coco format 
        boxes = records[['x', 'y', 'w', 'h']].values
        
        #Area of bb
        area = boxes[:,2]*boxes[:,3]
        area = torch.as_tensor(area, dtype=torch.float32)
        
        # AS pointed out by PRVI It works better if the main class is labelled as zero
        labels =  np.zeros(len(boxes), dtype=np.int32)

        
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']
            
            
        #Normalizing BBOXES
            
        _, h, w = image.shape
        boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'], rows=h, cols=w)
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.long)
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        
        return image, target, filename


if __name__ == '__main__':
    sett = ['train', 'test']
    for s in sett:
        df = pd.read_csv(f'dataset/{s}_labels.csv')
        df['x'], df['y'], df['w'], df['h'] = x1y1x2y2_to_x1y1wh(df['xmin'], df['ymin'], df['xmax'], df['ymax'])
        df = df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
        df = df.sample(frac=1)
        df.to_csv(f'dataset/{s}_coco.csv', index=False)