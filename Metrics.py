import torch

class SegmentationMetrics:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes

    def calculate_dice(self, preds, targets):
        smooth = 1.0
        preds = torch.argmax(preds, dim=1)
        dice = 0.0
        for i in range(self.num_classes):
            pred_i = (preds == i).float()
            target_i = (targets == i).float()
            intersection = torch.sum(pred_i * target_i)
            union = torch.sum(pred_i) + torch.sum(target_i)
            dice += (2. * intersection + smooth) / (union + smooth)
        return dice / self.num_classes

    def calculate_iou(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        iou = 0.0
        for i in range(self.num_classes):
            pred_i = (preds == i).float()
            target_i = (targets == i).float()
            intersection = torch.sum(pred_i * target_i)
            union = torch.sum(pred_i) + torch.sum(target_i) - intersection
            iou += intersection / (union + 1e-6)
        return iou / self.num_classes

    def calculate_pixel_accuracy(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        correct = torch.sum(preds == targets).float()
        total = targets.numel()
        return correct / total
