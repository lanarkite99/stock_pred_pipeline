import copy
import logging

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config

logger = logging.getLogger(__name__)


def model_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    optimizer: torch.optim.Optimizer = None,
    lr: float = 0.001,
    criterion: nn.Module = None,
) -> tuple[nn.Module, float, dict]:
    model.to(Config().device)
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = criterion or nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    best_loss = float('inf')
    best_state = None
    patience, counter = 5, 0
    best_epoch = 0
    history = {
        "train_losses": [],
        "val_losses": [],
        "learning_rates": [],
        "epochs_requested": epochs,
        "epochs_ran": 0,
    }

    for epoch_idx in range(epochs):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(Config().device), y.to(Config().device)
            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch_idx + 1}/{epochs}, Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(Config().device), y.to(Config().device)
                predictions = model(X)
                val_loss += criterion(predictions, y).item()
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")

        try:
            current_lr = optimizer.param_groups[0]['lr']
            history["train_losses"].append(float(avg_loss))
            history["val_losses"].append(float(avg_val_loss))
            history["learning_rates"].append(float(current_lr))
            history["epochs_ran"] = epoch_idx + 1
            mlflow.log_metrics(
                {
                    'train_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'lr': current_lr,
                },
                step=epoch_idx,
            )
        except Exception as exc:
            logger.warning(f"failed to log metrics to mlflow: {exc}")

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch_idx + 1
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch_idx + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best checkpoint from epoch {best_epoch}")

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = float(best_loss)
    history["final_train_loss"] = history["train_losses"][-1] if history["train_losses"] else None
    history["final_val_loss"] = history["val_losses"][-1] if history["val_losses"] else None
    return model, best_loss, history
