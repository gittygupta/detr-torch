from sklearn.model_selection import train_test_split
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion
from tqdm.autonotebook import tqdm
from utils import AverageMeter
from data_utils import *
from model import DETR
from config import *

import torch

matcher = HungarianMatcher()
weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}
losses = ['labels', 'boxes', 'cardinality']

def train_step(data_loader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    criterion.train()
    
    summary_loss = AverageMeter()
    
    tk0 = tqdm(data_loader, total=len(data_loader))
    for step, (images, targets, image_ids) in enumerate(tk0):
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        

        output = model(images)
        
        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        optimizer.zero_grad()

        losses.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        summary_loss.update(losses.item(), BATCH_SIZE)
        tk0.set_postfix(loss=summary_loss.avg)
        
    return summary_loss

def eval_step(data_loader, model, criterion, device):
    model.eval()
    criterion.eval()
    summary_loss = AverageMeter()
    
    with torch.no_grad():
        
        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (images, targets, image_ids) in enumerate(tk0):
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(images)
        
            loss_dict = criterion(output, targets)
            weight_dict = criterion.weight_dict
        
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            summary_loss.update(losses.item(),BATCH_SIZE)
            tk0.set_postfix(loss=summary_loss.avg)
    
    return summary_loss

def collate_fn(batch):
    return tuple(zip(*batch))

def run(train_filenames, valid_filenames, save_path=''):
    train_dataset = DroneDataset(filenames=train_filenames,
                                dataframe=df,
                                transforms=get_train_transforms()
                                )

    valid_dataset = DroneDataset(filenames=valid_filenames,
                                dataframe=df,
                                transforms=get_valid_transforms()
                                )
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False,
                                                    num_workers=4,
                                                    collate_fn=collate_fn
                                                    )

    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False,
                                                    num_workers=4,
                                                    collate_fn=collate_fn
                                                    )

    device = torch.device('cuda')
    model = DETR(num_classes=num_classes, num_queries=num_queries)
    model = model.to(device)
    criterion = SetCriterion(num_classes-1, matcher, weight_dict, eos_coef = null_class_coef, losses=losses)
    criterion = criterion.to(device)
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_loss = 10**5
    for epoch in range(EPOCHS):
        train_loss = train_step(train_data_loader, model,criterion, optimizer, device, scheduler=None, epoch=epoch)
        valid_loss = eval_step(valid_data_loader, model, criterion, device)
        
        print('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch+1, train_loss.avg, valid_loss.avg))
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            print('Epoch {}........Saving Model'.format(epoch+1))
            torch.save(model.state_dict(), f'{save_path}detr_{epoch}.pth')

if __name__ == '__main__':
    df = pd.read_csv('dataset/train_coco.csv')
    save_path = '/content/drive/MyDrive/Colab Notebooks/object_detection_3/saved_models/'
    train_df, valid_df = train_test_split(df, test_size=0.2)
    train_filenames = train_df.groupby('filename').min().index.values
    valid_filenames = valid_df.groupby('filename').min().index.values
    run(train_filenames, valid_filenames, save_path)