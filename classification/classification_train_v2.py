import time
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os

os.chdir("/home/administrator/Desktop/Muhammad Rizki Fadhil (00000079478)/Spyder/TJS_Project_V2/classification")

from Prepfortrain import data_split
from Prepfortrain import preprocess
from Visualization import plot_performance_with_auroc
from ViT_v2 import BinaryClassificationViTModel

###############################################################################
# Global device and random seed setup
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###############################################################################
# Vision Transformer (ViT) Training
###############################################################################
from cosmicannealing import CosineAnnealingWarmUpRestarts # Added import

def train_vit(X, y):
    print("========== Starting ViT Training ==========")

    import torch.nn.functional as F

    def pad_to_224(x):
        return F.pad(x, (0, 224 - x.shape[2], 0, 224 - x.shape[1]), "constant", -5)

    batch_size = 16
    num_epochs = 200
    learning_rate = 0.000001
    patience = 15
    pos_weight = torch.tensor(10)
    l2_reg = 0.001
    l1_reg = 0
    threshold = 0.35

    for i in range(100, 1001, 100):
        seed = i
        set_seed(seed)

        # Split
        X_train_tensor, X_valid_tensor, X_test_tensor, y_train_tensor, y_valid_tensor, y_test_tensor = data_split(X, y, seed)

        # Exact count
        print(f"--- Class Distribution for Seed {seed} ---")
        print(f"Total: {len(y_train_tensor) + len(y_valid_tensor) + len(y_test_tensor)}")
        print(f"Train: Healthy={torch.sum(y_train_tensor == 0)}, Injured={torch.sum(y_train_tensor == 1)}")
        print(f"Valid: Healthy={torch.sum(y_valid_tensor == 0)}, Injured={torch.sum(y_valid_tensor == 1)}")
        print(f"Test : Healthy={torch.sum(y_test_tensor == 0)}, Injured={torch.sum(y_test_tensor == 1)}")

        # Pad to 224x224
        X_train_tensor = pad_to_224(X_train_tensor)
        X_valid_tensor = pad_to_224(X_valid_tensor)
        X_test_tensor  = pad_to_224(X_test_tensor)

        # DataLoader
        train_dataset = TensorDataset(X_train_tensor.to(device), y_train_tensor.to(device))
        valid_dataset = TensorDataset(X_valid_tensor.to(device), y_valid_tensor.to(device))
        test_dataset = TensorDataset(X_test_tensor.to(device), y_test_tensor.to(device))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        print(f"{i+1}th iteration | seed={seed}")
        model = BinaryClassificationViTModel(threshold, pos_weight).to(device)

        # --- ADDED SCHEDULER INITIALIZATION ---
        # Note: We create the optimizer here so we can pass it to the scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0=10,
            T_mult=1,
            eta_max=0.000001,
            T_up=2,
            gamma=0.5
        )

        if device.type == 'cuda':
            torch.cuda.reset_max_memory_allocated(device)

        start_time = time.time()

        # --- UPDATED CALL ---
        # We now pass the scheduler into the train_model method
        loss_log, best_model_wts, best_loss, valid_log = model.train_model(
            train_loader,
            valid_loader,
            num_epochs,
            learning_rate,
            patience,
            l2_reg,
            l1_reg,
            scheduler=scheduler  # Make sure your ViT.py can accept this!
        )
        total_train_time = time.time() - start_time
        print(f"Total Training Time: {total_train_time:.2f} seconds")

        if device.type == 'cuda':
            max_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
            print(f"Maximum GPU memory allocated: {max_memory:.2f} MB")

        # Only try to load and plot if a best model was actually found
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
            plot_performance_with_auroc(
                model,
                X_test_tensor.cpu(),
                y_test_tensor.cpu().numpy(),
                device,
                threshold,
                loss_log,
                valid_log
            )
        else:
            print(f"Skipping plot for seed {seed}: No valid model weights found (Loss was NaN)")

###############################################################################
# Main routine to sequentially train all models
###############################################################################
def main():
    print("=== Starting Full Training Pipeline ===")
    X, y = preprocess()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    train_vit(X, y)

    print("=== All Trainings Completed ===")

if __name__ == "__main__":
    main()
