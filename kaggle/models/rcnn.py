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
from typing import Iterable 

class LitWheat(pl.LightningModule):
    def __init__(self, n_classes, learning_rate=1e-4 ):
        super().__init__()
        self.save_hyperparameters()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.hparams.n_classes)
        
    def forward(self, imgs, targets=None):
        # TorchVision FasterRCNN return the loss during the training
        
        opt = self.optimizers()
        self.detector.train()
        loss_dict = self.detector(imgs, targets)
        loss = sum(loss for loss in loss_dict.values())
        opt.zero_grad()
        
        self.detector.eval()
        outputs = self.detector(imgs)
        return {'loss': loss, 'output': outputs}
    
    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        
        loss_dict = self.detector(pixel_values, labels)
        loss = sum(loss for loss in loss_dict.values())
        self.log_dict(loss_dict)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        boxes = batch['labels']
        outputs = self.forward(pixel_values, boxes)
        
        accuracy = torch.mean(torch.stack([self.accuracy(b['boxes'], pb['boxes'], iou_threshold=0.75 ) for b, pb in zip(boxes, outputs['output'])]))
        metrics = {'val_accuracy': accuracy, 'val_loss': outputs['loss']}
        self.log_dict(metrics)
        
        return outputs['loss']
        
    def test_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        boxes = batch['labels']
        outputs = self.forward(pixel_values, boxes)
        
        test_accuracy = torch.mean(torch.stack([self.accuracy(b['boxes'], pb['boxes'], iou_threshold=0.75 ) for b, pb in zip(boxes, outputs['output'])]))
        metrics = {'test_accuracy': test_accuracy, 'test_loss': outputs['loss']}
        
        return metrics
        
        
    
    def configure_optimizers(self):
        params = [p for p in self.detector.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.hparams.learning_rate, momentum=0.9, weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        return [optimizer], [lr_scheduler]
    
    def predict_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']

        with torch.no_grad():
            output = self.detector(pixel_values)
        return output
    
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
        