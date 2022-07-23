import pytorch_lightning as pl

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
from torchvision.ops.boxes import box_iou
from torchvision.models.detection._utils import Matcher
from torchvision.ops import nms, box_convert
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch

class LitWheat(pl.LightningModule):
    def __init__(self, n_classes, learning_rate=1e-4 ):
        super().__init__()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        self.lr = learning_rate
        
    def forward(self, imgs, targets=None):
        # TorchVision FasterRCNN return the loss during the training
        self.detector.eval()
        return self.detector(imgs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    
    
    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        
        loss_dict = self.detector(pixel_values, labels)
        loss = sum(loss for loss in loss_dict.values())
        self.log_dict(loss_dict)
        return {'loss': loss, 'log':loss_dict}
    
    def validation_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        boxes = batch['labels']
        pred_boxes = self.forward(pixel_values)
        
        accuracy = torch.mean(torch.stack([self.accuracy(b['boxes'], pb['boxes'], iou_threshold=0.5 ) for b, pb in zip(boxes, pred_boxes)]))
        metrics = {'val_accuracy': accuracy}
        self.log_dict(metrics)
        
        return metrics
        
    def test_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        boxes = batch['labels']
        pred_boxes = self.forward(pixel_values)
        
        test_accuracy = torch.mean(torch.stack([self.accuracy(b['boxes'], pb['boxes'], iou_threshold=0.5 ) for b, pb in zip(boxes, pred_boxes)]))
        metrics = {'test_accuracy': test_accuracy}
        return self.test_accuracy
        
    
    def configure_optimizers(self):
        params = [p for p in self.detector.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        return [optimizer], [lr_scheduler]
    
    def accuracy(self, src_boxes, pred_boxes, iou_threshold=0.5):
        """ The accuracy method is not the one used in the evaluator but very similar"""
        total_gt = len(src_boxes)
        total_pred = len(pred_boxes)
        if total_gt > 0 and total_pred > 0:
            matcher = Matcher(iou_threshold,iou_threshold,allow_low_quality_matches=False) 
            match_quality_matrix = box_iou(src_boxes,pred_boxes)

            results = matcher(match_quality_matrix)

            true_positive = torch.count_nonzero(results.unique() != -1)
            matched_elements = results[results > -1]

            #in Matcher, a pred element can be matched only twice 
            false_positive = torch.count_nonzero(results == -1) + ( len(matched_elements) - len(matched_elements.unique()))
            false_negative = total_gt - true_positive


            return  true_positive / ( true_positive + false_positive + false_negative )

        elif total_gt == 0:
            if total_pred > 0:
                return torch.tensor(0.).cuda()
            else:
                return torch.tensor(1.).cuda()
        elif total_gt > 0 and total_pred == 0:
              return torch.tensor(0.).cuda()            
        
