import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import wandb

# -----------------------------
# Config
# -----------------------------
CONFIG = {
    "image_size": 512,
    "batch_size": 8,
    "val_batch_size": 1,
    "test_batch_size": 1,
    "epochs": 100,
    "lr": 5e-5, # Adjusted learning rate for fine-tuning
    "num_classes": 3,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "save_model_every": 10,  # Save model every N epochs
    "test_every": 5,        # Run test evaluation every N epochs
    "model_checkpoint": "nvidia/segformer-b0-finetuned-ade-512-512", # Hugging Face model
}

wandb.init(
    project="segmentasi-jamur",
    name="segformer-fungi-v1",
    config=CONFIG
)

# -----------------------------
# Dataset
# -----------------------------
class FungiDataset(Dataset):
    """
    Fungi dataset compatible with SegFormer.
    It uses SegformerImageProcessor to prepare images for the model.
    """
    def __init__(self, image_dir, mask_dir, image_processor, split_name="unknown"):
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
        self.image_paths = []
        self.image_processor = image_processor
        self.split_name = split_name

        valid_pairs = []
        for mask_path in self.mask_paths:
            base_name = os.path.splitext(os.path.basename(mask_path))[0]
            image_path = os.path.join(image_dir, f"{base_name}.jpg")
            if os.path.exists(image_path):
                valid_pairs.append((image_path, mask_path))
            else:
                print(f"[WARNING] Image not found for mask: {mask_path}")

        if len(valid_pairs) == 0:
            raise ValueError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")

        self.image_paths, self.mask_paths = zip(*valid_pairs)
        print(f"✅ Loaded {len(self.image_paths)} image-mask pairs for {split_name} split")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        # Ensure image is resized to a consistent size before processing
        image = image.resize((CONFIG["image_size"], CONFIG["image_size"]))
        
        # The image processor handles normalization and conversion to tensor
        inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0) # Remove batch dim

        # Process mask
        mask = mask.resize((CONFIG["image_size"], CONFIG["image_size"]), Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))

        return pixel_values, mask_tensor

# -----------------------------
# Metrics
# -----------------------------
def calculate_metrics(pred, target):
    """Calculates macro-averaged metrics."""
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()

    precision = precision_score(target, pred, average='macro', zero_division=0)
    recall = recall_score(target, pred, average='macro', zero_division=0)
    f1 = f1_score(target, pred, average='macro', zero_division=0)
    acc = accuracy_score(target, pred)
    return precision, recall, f1, acc

def calculate_per_class_metrics(pred, target, num_classes):
    """Calculate per-class metrics for detailed analysis."""
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    
    per_class_metrics = {}
    for class_id in range(num_classes):
        # Create binary masks for current class
        pred_binary = (pred == class_id).astype(int)
        target_binary = (target == class_id).astype(int)
        
        if target_binary.sum() > 0:  # Only calculate if class exists in target
            precision = precision_score(target_binary, pred_binary, zero_division=0)
            recall = recall_score(target_binary, pred_binary, zero_division=0)
            f1 = f1_score(target_binary, pred_binary, zero_division=0)
            
            per_class_metrics[f'class_{class_id}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    return per_class_metrics

# -----------------------------
# Train, Validate & Test Functions
# -----------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for i, (images, masks) in enumerate(tqdm(loader, desc="Training", leave=False)):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(pixel_values=images, labels=masks)
        
        # SegFormer returns loss and logits. We can use its loss directly.
        loss = outputs.loss
        logits = outputs.logits
        
        # Upsample logits to match mask size for metrics calculation
        upsampled_logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        if torch.isnan(upsampled_logits).any():
            print(f"[NaN DETECTED] in outputs at batch {i}")
            continue

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(upsampled_logits, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    prec, rec, f1, acc = calculate_metrics(all_preds, all_targets)

    return total_loss / len(loader), prec, rec, f1, acc

def validate(model, loader, loss_fn, device, split_name="validation"):
    """Validates the model performance."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(loader, desc=f"{split_name.capitalize()}", leave=False)):
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(pixel_values=images, labels=masks)
            loss = outputs.loss
            logits = outputs.logits

            # Upsample logits to match mask size for metrics
            upsampled_logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            if torch.isnan(upsampled_logits).any():
                print(f"[NaN DETECTED] in {split_name} batch {i}")
                continue

            total_loss += loss.item()

            preds = torch.argmax(upsampled_logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

            if i % 10 == 0 and split_name == "validation":
                print(f"[{split_name.capitalize()} Batch {i}] loss={loss.item():.4f}, shape={upsampled_logits.shape}")

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    prec, rec, f1, acc = calculate_metrics(all_preds, all_targets)
    
    # Calculate per-class metrics for detailed analysis
    per_class_metrics = calculate_per_class_metrics(all_preds, all_targets, CONFIG["num_classes"])

    return total_loss / len(loader), prec, rec, f1, acc, per_class_metrics

def test_model(model, test_loader, loss_fn, device):
    """Dedicated test function with detailed logging."""
    print("\n🧪 Running Test Evaluation...")
    test_loss, test_prec, test_rec, test_f1, test_acc, per_class_metrics = validate(
        model, test_loader, loss_fn, device, "test"
    )
    
    print(f"📊 Test Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   Precision: {test_prec:.4f}")
    print(f"   Recall: {test_rec:.4f}")
    print(f"   F1-Score: {test_f1:.4f}")
    
    # Log per-class metrics
    print(f"📋 Per-Class Metrics:")
    for class_name, metrics in per_class_metrics.items():
        print(f"   {class_name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    
    return test_loss, test_prec, test_rec, test_f1, test_acc, per_class_metrics

# -----------------------------
# Model Saving & Loading
# -----------------------------
def save_model_checkpoint(model, optimizer, epoch, best_val_f1, filepath):
    """Save model checkpoint with training state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_f1': best_val_f1,
        'config': CONFIG
    }, filepath)
    print(f"💾 Checkpoint saved to {filepath}")

def save_best_model(model, filepath):
    """Save only the model weights (for inference)."""
    model.save_pretrained(filepath)
    print(f"🏆 Best model saved to {filepath}")

# -----------------------------
# Main Training Loop
# -----------------------------
def main():
    # Initialize image processor from pre-trained model
    image_processor = SegformerImageProcessor.from_pretrained(CONFIG["model_checkpoint"])
    image_processor.do_resize = False # We handle resizing in the dataset
    image_processor.do_normalize = True

    # Load all datasets
    print("📂 Loading datasets...")
    train_dataset = FungiDataset("dataset_masks/train/images", "dataset_masks/train/label_masks", image_processor, "train")
    val_dataset = FungiDataset("dataset_masks/valid/images", "dataset_masks/valid/label_masks", image_processor, "validation")
    test_dataset = FungiDataset("dataset_masks/test/images", "dataset_masks/test/label_masks", image_processor, "test")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["val_batch_size"], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["test_batch_size"], shuffle=False, num_workers=2)

    # Initialize SegFormer model for fine-tuning
    # ignore_mismatched_sizes=True allows loading pre-trained weights
    # into a new head with a different number of classes.
    model = SegformerForSemanticSegmentation.from_pretrained(
        CONFIG["model_checkpoint"],
        num_labels=CONFIG["num_classes"],
        ignore_mismatched_sizes=True
    ).to(CONFIG["device"])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"]) # AdamW is often better for transformers
    criterion = nn.CrossEntropyLoss() # Still useful for the validate function signature, though model computes its own loss
    
    # Training tracking variables
    best_val_f1 = 0.0
    best_test_f1 = 0.0
    
    print(f"🚀 Starting training for {CONFIG['epochs']} epochs...")
    print(f"📊 Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    for epoch in range(CONFIG["epochs"]):
        print(f"\n📅 Epoch {epoch+1}/{CONFIG['epochs']}")

        # Training
        train_loss, train_prec, train_rec, train_f1, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, CONFIG["device"]
        )
        
        # Validation
        val_loss, val_prec, val_rec, val_f1, val_acc, val_per_class = validate(
            model, val_loader, criterion, CONFIG["device"], "validation"
        )

        # Initialize wandb log with training and validation metrics
        wandb_log = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/precision": train_prec,
            "train/recall": train_rec,
            "train/f1": train_f1,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/precision": val_prec,
            "val/recall": val_rec,
            "val/f1": val_f1,
            "val/accuracy": val_acc,
        }
        
        # Add per-class validation metrics to wandb
        for class_name, metrics in val_per_class.items():
            wandb_log[f"val_per_class/{class_name}_precision"] = metrics['precision']
            wandb_log[f"val_per_class/{class_name}_recall"] = metrics['recall']
            wandb_log[f"val_per_class/{class_name}_f1"] = metrics['f1']

        # Test evaluation (every N epochs or last epoch)
        if (epoch + 1) % CONFIG["test_every"] == 0 or epoch == CONFIG["epochs"] - 1:
            test_loss, test_prec, test_rec, test_f1, test_acc, test_per_class = test_model(
                model, test_loader, criterion, CONFIG["device"]
            )
            
            # Add test metrics to main wandb log
            wandb_log.update({
                "test/loss": test_loss,
                "test/precision": test_prec,
                "test/recall": test_rec,
                "test/f1": test_f1,
                "test/accuracy": test_acc,
            })
            
            # Add per-class test metrics
            for class_name, metrics in test_per_class.items():
                wandb_log[f"test_per_class/{class_name}_precision"] = metrics['precision']
                wandb_log[f"test_per_class/{class_name}_recall"] = metrics['recall']
                wandb_log[f"test_per_class/{class_name}_f1"] = metrics['f1']
            
            # Update best test score
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                wandb_log["best_test_f1"] = best_test_f1
        else:
            # If no test evaluation this epoch, still log the current best test F1
            wandb_log["best_test_f1"] = best_test_f1

        # Log to wandb
        wandb.log(wandb_log)

        # Save best model based on validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_best_model(model, "segformer_fungi_best_model")
            wandb.log({"best_val_f1": best_val_f1})

        # Save checkpoint periodically
        if (epoch + 1) % CONFIG["save_model_every"] == 0:
            checkpoint_path = f"segformer_fungi_checkpoint_epoch_{epoch+1}.pth"
            save_model_checkpoint(model, optimizer, epoch, best_val_f1, checkpoint_path)

        print(f"📈 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Best Val F1: {best_val_f1:.4f}")

    # Final test evaluation
    print("\n🎯 Final Test Evaluation...")
    final_test_loss, final_test_prec, final_test_rec, final_test_f1, final_test_acc, final_test_per_class = test_model(
        model, test_loader, criterion, CONFIG["device"]
    )
    
    # Log final test results
    final_log = {
        "final_test/loss": final_test_loss,
        "final_test/precision": final_test_prec,
        "final_test/recall": final_test_rec,
        "final_test/f1": final_test_f1,
        "final_test/accuracy": final_test_acc,
    }
    
    for class_name, metrics in final_test_per_class.items():
        final_log[f"final_test_per_class/{class_name}_precision"] = metrics['precision']
        final_log[f"final_test_per_class/{class_name}_recall"] = metrics['recall']
        final_log[f"final_test_per_class/{class_name}_f1"] = metrics['f1']
    
    wandb.log(final_log)

    # Save final model
    model.save_pretrained("segformer_fungi_final_model")
    print("✅ Final model saved to segformer_fungi_final_model/")
    
    # Create summary table for wandb
    summary_table = wandb.Table(columns=["Metric", "Train", "Validation", "Test"])
    summary_table.add_data("Loss", f"{train_loss:.4f}", f"{val_loss:.4f}", f"{final_test_loss:.4f}")
    summary_table.add_data("Accuracy", f"{train_acc:.4f}", f"{val_acc:.4f}", f"{final_test_acc:.4f}")
    summary_table.add_data("Precision", f"{train_prec:.4f}", f"{val_prec:.4f}", f"{final_test_prec:.4f}")
    summary_table.add_data("Recall", f"{train_rec:.4f}", f"{val_rec:.4f}", f"{final_test_rec:.4f}")
    summary_table.add_data("F1-Score", f"{train_f1:.4f}", f"{val_f1:.4f}", f"{final_test_f1:.4f}")
    
    wandb.log({"training_summary": summary_table})
    
    print(f"\n🏁 Training completed!")
    print(f"📊 Final Results:")
    print(f"   Best Validation F1: {best_val_f1:.4f}")
    print(f"   Best Test F1: {best_test_f1:.4f}")
    print(f"   Final Test F1: {final_test_f1:.4f}")

if __name__ == "__main__":
    main()
