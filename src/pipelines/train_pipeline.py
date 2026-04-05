import json
import logging
import os
from typing import Dict, List, Optional

import joblib
import mlflow
import torch
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.config import Config
from src.data.ingestion import fetch_ohlcv
from src.data.preparation import StockDataset
from src.exception import PipelineError
from src.model.definition import StockLSTM
from src.model.evaluation import evaluate_model
from src.model.training import model_train
from src.utils import init_dir, save_json, setup_dagshub_mlflow

logger = logging.getLogger(__name__)


def promote_model_to_production(model_name: str, version: int) -> None:
    try:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
        )
        logger.info(f"Model {model_name} version {version} promoted to Production")
    except Exception as e:
        logger.error(f"Failed to promote model {model_name} version {version} to Production: {e}")


def get_output_paths(base_dir: str, ticker: str, model_type: str) -> Dict[str, str]:
    os.makedirs(base_dir, exist_ok=True)
    prefix = f"{ticker}_{model_type}"
    return {
        "model_path": os.path.join(base_dir, f"{prefix}_model.pt"),
        "scaler_path": os.path.join(base_dir, f"{prefix}_scaler.pkl"),
        "summary_path": os.path.join(base_dir, f"{prefix}_training_summary.json"),
    }


def _config_snapshot(config: Config) -> Dict[str, object]:
    return {
        "context_len": config.context_len,
        "pred_len": config.pred_len,
        "input_size": config.input_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "fine_tune_lr": config.fine_tune_lr,
        "features": list(config.features),
        "device": config.device,
        "start_date": config.start_date,
        "transfer_strategy": config.transfer_strategy,
        "parent_ticker": config.parent_ticker,
    }


def _validate_transfer_compatibility(model: StockLSTM, config: Config, parent_summary_path: str) -> None:
    if getattr(model, "input_size", None) != config.input_size:
        raise PipelineError(
            f"Parent model input_size={getattr(model, 'input_size', None)} does not match child config input_size={config.input_size}"
        )
    if getattr(model, "pred_len", None) != config.pred_len:
        raise PipelineError(
            f"Parent model pred_len={getattr(model, 'pred_len', None)} does not match child config pred_len={config.pred_len}"
        )

    if not os.path.exists(parent_summary_path):
        return

    try:
        with open(parent_summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read parent summary for compatibility checks: {e}")
        return

    parent_config = summary.get("config")
    if not parent_config:
        return

    parent_features = parent_config.get("features")
    if parent_features and list(parent_features) != list(config.features):
        raise PipelineError(
            f"Parent model features {parent_features} do not match child config features {list(config.features)}"
        )
    if int(parent_config.get("input_size", config.input_size)) != config.input_size:
        raise PipelineError("Parent summary input_size does not match child config input_size")
    if int(parent_config.get("pred_len", config.pred_len)) != config.pred_len:
        raise PipelineError("Parent summary pred_len does not match child config pred_len")


def _build_dataset_and_loaders(df, config: Config):
    split_row = int(0.8 * len(df))
    min_rows = config.context_len + config.pred_len + 1
    if split_row < min_rows:
        raise PipelineError("training window is too small after chronological split")

    train_df = df.iloc[:split_row].copy()
    val_df = df.iloc[max(0, split_row - config.context_len) :].copy()

    scaler = StandardScaler()
    scaler.fit(train_df[config.features].values)

    train_dataset = StockDataset(
        data=train_df,
        scaler=scaler,
        context_len=config.context_len,
        pred_len=config.pred_len,
    )
    val_dataset = StockDataset(
        data=val_df,
        scaler=scaler,
        context_len=config.context_len,
        pred_len=config.pred_len,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise PipelineError("dataset too small to create chronological train/validation sequences")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    dataset_stats = {
        "raw_rows": int(len(df)),
        "sequence_count": int(len(train_dataset) + len(val_dataset)),
        "train_size": int(len(train_dataset)),
        "val_size": int(len(val_dataset)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
    }
    return scaler, train_loader, val_loader, dataset_stats


def _log_training_run(config: Config, ticker: str, metrics: Dict[str, float], model_type: str, dataset_stats: Dict[str, int]) -> None:
    mlflow.log_params(
        {
            "ticker": ticker,
            "model_type": model_type,
            "context_len": config.context_len,
            "pred_len": config.pred_len,
            "input_size": config.input_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "fine_tune_lr": config.fine_tune_lr,
            "features": config.features,
            "device": config.device,
            "start_date": config.start_date,
            "transfer_strategy": config.transfer_strategy,
            **dataset_stats,
        }
    )
    mlflow.log_metrics(metrics)


def _save_and_log_summary(summary: Dict, summary_path: str) -> None:
    save_json(summary, summary_path)
    mlflow.log_artifact(summary_path)
    mlflow.log_text(json.dumps(summary, indent=2), "training_summary.json")


def _print_training_summary(summary: Dict) -> None:
    print("Training summary")
    print(f"ticker: {summary['ticker']}")
    print(f"raw_rows: {summary['dataset']['raw_rows']}")
    print(f"sequence_count: {summary['dataset']['sequence_count']}")
    print(f"train_size: {summary['dataset']['train_size']}")
    print(f"val_size: {summary['dataset']['val_size']}")
    print(f"epochs_requested: {summary['training']['epochs_requested']}")
    print(f"epochs_ran: {summary['training']['epochs_ran']}")
    print(f"best_epoch: {summary['training']['best_epoch']}")
    print(f"best_val_loss: {summary['training']['best_val_loss']:.6f}")
    print(f"mse: {summary['metrics']['mse']:.6f}")
    print(f"rmse: {summary['metrics']['rmse']:.6f}")
    print(f"mae: {summary['metrics']['mae']:.6f}")
    print(f"r2: {summary['metrics']['r2']:.6f}")


def train_parent_model() -> Dict:
    config = Config()
    init_dir()
    setup_dagshub_mlflow()

    ticker = config.parent_ticker
    output_paths = get_output_paths(config.parent_dir, ticker, "parent")

    try:
        df = fetch_ohlcv(ticker, start=config.start_date)
        model = StockLSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            pred_len=config.pred_len,
        )
        scaler, train_loader, val_loader, dataset_stats = _build_dataset_and_loaders(df, config)

        with mlflow.start_run(run_name=f"train_parent_{ticker}"):
            model, best_loss, history = model_train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config.parent_epochs,
                lr=config.learning_rate,
            )
            metrics = evaluate_model(model, val_loader)
            metrics["best_val_loss"] = float(best_loss)
            _log_training_run(config, ticker, metrics, "parent", dataset_stats)

            torch.save(model.state_dict(), output_paths["model_path"])
            joblib.dump(scaler, output_paths["scaler_path"])
            mlflow.log_artifact(output_paths["model_path"])
            mlflow.log_artifact(output_paths["scaler_path"])

            summary = {
                "ticker": ticker,
                "model_type": "parent",
                "config": _config_snapshot(config),
                "dataset": dataset_stats,
                "training": history,
                "metrics": metrics,
                "artifacts": output_paths,
                "tracking_uri": mlflow.get_tracking_uri(),
            }
            _save_and_log_summary(summary, output_paths["summary_path"])

        logger.info(f"Parent model trained successfully for {ticker}")
        _print_training_summary(summary)
        return {
            "ticker": ticker,
            "model_path": output_paths["model_path"],
            "scaler_path": output_paths["scaler_path"],
            "summary_path": output_paths["summary_path"],
            "metrics": metrics,
            "dataset": dataset_stats,
            "training": history,
        }
    except Exception as e:
        logger.error(f"Failed to train parent model for {ticker}: {e}")
        raise PipelineError(f"Failed to train parent model for {ticker}: {e}") from e


def train_child_model(ticker: str, strategy: Optional[str] = None) -> Dict:
    config = Config()
    setup_dagshub_mlflow()
    strategy = strategy or config.transfer_strategy

    parent_paths = get_output_paths(config.parent_dir, config.parent_ticker, "parent")
    if not os.path.exists(parent_paths["model_path"]):
        raise PipelineError(f"Parent model not found at {parent_paths['model_path']}")

    child_dir = os.path.join(config.workdir, ticker.lower())
    output_paths = get_output_paths(child_dir, ticker, "child")

    try:
        df = fetch_ohlcv(ticker, start=config.start_date)
        scaler, train_loader, val_loader, dataset_stats = _build_dataset_and_loaders(df, config)

        model = StockLSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            pred_len=config.pred_len,
        ).to(config.device)
        model.load_state_dict(torch.load(parent_paths["model_path"], map_location=config.device))
        _validate_transfer_compatibility(model, config, parent_paths["summary_path"])

        if strategy == "freeze":
            for name, param in model.named_parameters():
                if "lstm" in name:
                    param.requires_grad = False
            optimizer = torch.optim.Adam(
                [param for param in model.parameters() if param.requires_grad],
                lr=config.learning_rate,
                weight_decay=1e-5,
            )
        elif strategy == "fine_tune":
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=config.fine_tune_lr, weight_decay=1e-5)
        else:
            raise PipelineError(f"Invalid transfer strategy: {strategy}")

        with mlflow.start_run(run_name=f"train_child_{ticker}"):
            model, best_loss, history = model_train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config.child_epochs,
                optimizer=optimizer,
                lr=config.fine_tune_lr if strategy == "fine_tune" else config.learning_rate,
            )
            metrics = evaluate_model(model, val_loader)
            metrics["best_val_loss"] = float(best_loss)
            metrics["transfer_strategy"] = 0.0 if strategy == "freeze" else 1.0
            _log_training_run(config, ticker, metrics, "child", dataset_stats)

            torch.save(model.state_dict(), output_paths["model_path"])
            joblib.dump(scaler, output_paths["scaler_path"])
            mlflow.log_artifact(output_paths["model_path"])
            mlflow.log_artifact(output_paths["scaler_path"])

            summary = {
                "ticker": ticker,
                "model_type": "child",
                "config": _config_snapshot(config),
                "dataset": dataset_stats,
                "training": history,
                "metrics": metrics,
                "artifacts": output_paths,
                "tracking_uri": mlflow.get_tracking_uri(),
            }
            _save_and_log_summary(summary, output_paths["summary_path"])

        logger.info(f"Child model trained successfully for {ticker}")
        _print_training_summary(summary)
        return {
            "ticker": ticker,
            "model_path": output_paths["model_path"],
            "scaler_path": output_paths["scaler_path"],
            "summary_path": output_paths["summary_path"],
            "metrics": metrics,
            "dataset": dataset_stats,
            "training": history,
        }
    except Exception as e:
        logger.error(f"Failed to train child model for {ticker}: {e}")
        raise PipelineError(f"Failed to train child model for {ticker}: {e}") from e


def train_child_models(tickers: Optional[List[str]] = None) -> List[Dict]:
    config = Config()
    tickers = tickers or config.child_tickers
    results = []

    for ticker in tickers:
        results.append(train_child_model(ticker))

    return results
