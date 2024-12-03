import re
import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer

def extract_subtype_from_filename(filename):
    """
    Eextract subtypes from the filename.
    """
    match = re.search(r'quiz_(\d+)_', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Unable to extract subtype from filename: {filename}")

class nnUnnUNetTrainerClassification(nnUNetTrainer):
    def initialize(self):
        super().initialize()
        
        # Add a classification head to the network
        num_classes = 3  # Adjust according to your dataset
        classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(self.network.encoder.output_channels[-1], num_classes)
        )
        self.network.classification_head = classification_head.to(self.device)

        # Modify the loss to include classification
        self.classification_loss = nn.CrossEntropyLoss()
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target_seg = batch['target']
        filenames = batch['filenames']
        
        # Extract subtype from filenames for classification
        target_class = torch.tensor([extract_subtype_from_filename(f) for f in filenames], device=self.device)
        
        data = data.to(self.device, non_blocking=True)
  
