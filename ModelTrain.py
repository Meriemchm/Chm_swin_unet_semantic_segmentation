import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
from SatelliteData import SatelliteDataset
from LossFunctions import DiceLoss
from torch.utils.data import DataLoader, random_split
from ConfigFile import config
from Metrics import SegmentationMetrics

def trainer(config, model, snapshot_path):
    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(config))

    base_lr = config.TRAIN.BASE_LR
    num_classes = config.MODEL.NUM_CLASSES
    batch_size = config.DATA.BATCH_SIZE * config.TRAIN.NUM_GPUS
    momentum = config.TRAIN.OPTIMIZER.MOMENTUM
    weight_decay = config.TRAIN.WEIGHT_DECAY

    images_path_train = "D:/Documents/telechargement/dataset_split/train/images"
    masks_path_train = "D:/Documents/telechargement/dataset_split/train/binary masks"
    images_path_val = "D:/Documents/telechargement/dataset_split/val/images"
    masks_path_val = "D:/Documents/telechargement/dataset_split/val/binary masks"

    db_train = SatelliteDataset(images_path_train, masks_path_train, transform=None)
    db_val = SatelliteDataset(images_path_val, masks_path_val, transform=None)

    """images_path = "D:/Documents/telechargement/landcoverv2/images_t"
    masks_path = "D:/Documents/telechargement/landcoverv2/masks_t
    # Charger le dataset complet
    full_dataset = SatelliteDataset(images_path, masks_path)

    # Séparer en train (80%) et validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    db_train, db_val = random_split(full_dataset, [train_size, val_size])"""

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

    if config.TRAIN.NUM_GPUS > 1:
        model = nn.DataParallel(model)

    model = model.cuda()
    model.train()

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    iter_num = 0
    max_epoch = config.TRAIN.EPOCHS
    max_iterations = max_epoch * len(trainloader)
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations.")

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    metrics = SegmentationMetrics(num_classes=config.MODEL.NUM_CLASSES)

    for epoch in range(max_epoch):
        model.train()
        train_loss = 0
        train_dice, train_iou = 0, 0
        
        with tqdm(trainloader, unit="batch", desc=f"Epoch {epoch+1}/{max_epoch}") as tepoch:
            for i_batch, (images, masks) in enumerate(tepoch):
                images, masks = images.cuda(), masks.cuda()
                outputs = model(images)

                loss_ce = ce_loss(outputs, masks.long())
                loss_dice = dice_loss(outputs, masks, softmax=True)
                loss = 0.4 * loss_ce + 0.6 * loss_dice

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_num += 1
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                train_loss += loss.item()
                train_dice += metrics.calculate_dice(outputs, masks).item()
                train_iou += metrics.calculate_iou(outputs, masks).item()
                tepoch.set_postfix(train_loss=train_loss/(i_batch+1), 
                                   train_dice=train_dice/(i_batch+1), 
                                   train_iou=train_iou/(i_batch+1)
                                   )

        avg_train_loss = train_loss / len(trainloader)

        # Validation
        model.eval()
        val_loss = 0
        val_dice, val_iou = 0, 0

        with torch.no_grad():
            with tqdm(valloader, unit="batch", desc=f"Validation Epoch {epoch+1}/{max_epoch}") as tepoch_val:
                for i_batch, (images, masks) in enumerate(tepoch_val):
                    images, masks = images.cuda(), masks.cuda()
                    outputs = model(images)

                    loss_ce = ce_loss(outputs, masks.long())
                    loss_dice = dice_loss(outputs, masks, softmax=True)
                    loss = 0.4 * loss_ce + 0.6 * loss_dice
                    
                    val_loss += loss.item()
                    val_dice += metrics.calculate_dice(outputs, masks).item()
                    val_iou += metrics.calculate_iou(outputs, masks).item()
                        #val_pixel_acc += metrics.calculate_pixel_accuracy(outputs, masks).item()

                    tepoch.set_postfix(val_loss=val_loss/(i_batch+1), 
                                        val_dice=val_dice/(i_batch+1), 
                                        val_iou=val_iou/(i_batch+1)
                                        ) #val_pixel_acc=val_pixel_acc/(batch_idx+1)

        avg_val_loss = val_loss / len(valloader)

        log_msg = (f"Epoch {epoch+1}/{max_epoch}, "
           f"Train Loss: {avg_train_loss:.4f}, "
           f"Train Dice: {train_dice/len(trainloader):.4f}, "
           f"Train IoU: {train_iou/len(trainloader):.4f}, "
           f"Val Loss: {avg_val_loss:.4f}, "
           f"Val Dice: {val_dice/len(valloader):.4f}, "
           f"Val IoU: {val_iou/len(valloader):.4f}")

        logging.info(log_msg)
        with open(os.path.join(snapshot_path, 'training_logs_cds.txt'), 'a') as log_file:
            log_file.write(log_msg + "\n")

        scheduler.step(avg_val_loss)
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(snapshot_path, "best_model.pth"))
            logging.info("! Nouveau meilleur modèle sauvegardé.")
        else:
            patience_counter += 1
            logging.info(f"!! Pas d'amélioration. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logging.info("!!! Early stopping activé.")
            break

    return "Training Finished!"
