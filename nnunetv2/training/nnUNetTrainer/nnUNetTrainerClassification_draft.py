import torch
import torch.nn as nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerClassification(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
             device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        print("Plans loaded:", self.plans)
        
        # Ensure classification_num_classes is loaded
        self.classification_num_classes = self.plans['configurations'][configuration]['classification_num_classes']
        print(f"Loaded classification_num_classes: {self.classification_num_classes}")
        assert self.classification_num_classes is not None, (
            "classification_num_classes must be defined in the plans file."
        )
        assert isinstance(self.classification_num_classes, int), (
            "classification_num_classes must be an integer. "
            f"Got {type(self.classification_num_classes).__name__}."
        )

    def build_network_architecture(self):
        """
        Build segmentation network, and add classification head
        """
        # Invoke the parent method and build the segmentation network
        super().build_network_architecture() 

        # Check the parent class initialize the size of encoder
        assert hasattr(self.network, "encoder"), "The network does not have an encoder attribute."
        assert hasattr(self.network, "encoder_output_size"), (
            "The network does not define 'encoder_output_size'. Please ensure it is set correctly in the network."
        )
        

        # Add classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),  
            nn.Flatten(),                     
            nn.Linear(self.network.encoder_output_size, 128), 
            nn.ReLU(),
            nn.Linear(128, self.classification_num_classes)  
        )

    def forward(self, x):
        encoder_output = self.network.encoder(x)  
        segmentation_output = self.network.decoder(encoder_output)  
        classification_output = self.classification_head(encoder_output)  
        return segmentation_output, classification_output
    
    def compute_loss(self, outputs, targets):
        seg_output, class_output = outputs
        seg_target, class_target = targets

        seg_loss = self.loss(seg_output, seg_target)

        classification_loss_fn = nn.CrossEntropyLoss()
        class_loss = classification_loss_fn(class_output, class_target)

        total_loss = seg_loss + 0.5 * class_loss 
        return total_loss 

    def train_step(self, batch: dict) -> dict:
        """
        Re-define the train step
        """
        data = batch["data"].to(self.device, non_blocking=True)
        seg_target = batch["seg_target"].to(self.device, non_blocking=True)
        class_target = batch["class_target"].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else torch.no_grad():
            seg_output, class_output = self.forward(data)
            loss = self.loss((seg_output, class_output), (seg_target, class_target))

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return {'loss': loss.item()}
    
    def validation_step(self, batch: dict) -> dict:
        """
        Redefine the validation step to include classification.
        """
        assert "data" in batch and "seg_target" in batch and "class_target" in batch, "Batch is missing required keys."

        data = batch["data"].to(self.device, non_blocking=True)
        seg_target = batch["seg_target"].to(self.device, non_blocking=True)
        class_target = batch["class_target"].to(self.device, non_blocking=True)

        with torch.no_grad():
            seg_output, class_output = self.forward(data)
            loss = self.compute_loss((seg_output, class_output), (seg_target, class_target))
            correct = (class_output.argmax(dim=1) == class_target).sum().item()
            accuracy = correct / class_target.size(0)
        
        return {"loss": loss.item(), "accuracy": accuracy}

    def on_validation_epoch_end(self, val_outputs: list):
        """
        Handle the end of a validation epoch.
        """
        avg_loss = torch.tensor([x["loss"] for x in val_outputs]).mean().item()
        avg_accuracy = torch.tensor([x["accuracy"] for x in val_outputs]).mean().item()

        self.logger.log("val_loss", avg_loss, self.current_epoch)
        self.logger.log("val_accuracy", avg_accuracy, self.current_epoch)
        self.print_to_log_file(f"Validation loss: {avg_loss:.4f}, accuracy: {avg_accuracy:.4f}")

    def initialize(self):
        """
        Override the initialization method to ensure compatibility with classification head.
        """
        super().initialize()
        assert hasattr(self, "classification_head"), "Classification head not initialized. Call build_network_architecture first."

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint including the classification head.
        """
        if self.local_rank == 0:
            checkpoint = {
                "network_state_dict": self.network.state_dict(),
                "classification_head_state_dict": self.classification_head.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.current_epoch,
                "loss_scaler_state_dict": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            }
            torch.save(checkpoint, filename)

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint including the classification head.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.classification_head.load_state_dict(checkpoint["classification_head_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.grad_scaler is not None and "loss_scaler_state_dict" in checkpoint:
            self.grad_scaler.load_state_dict(checkpoint["loss_scaler_state_dict"])
        self.current_epoch = checkpoint["epoch"]

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()

        dataset_tr = nnUNetDatasetSubtype(
            folder=self.preprocessed_dataset_folder,
            case_identifiers=tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
        dataset_val = nnUNetDatasetSubtype(
            folder=self.preprocessed_dataset_folder,
            case_identifiers=val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )

        return dataset_tr, dataset_val
