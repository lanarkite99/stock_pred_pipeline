import logging
import os
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.config import Config
from src.data.ingestion import fetch_ohlcv
from src.data.preparation import StockDataset
from src.model.definition import StockLSTM
from src.model.training import model_train


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("FEAST_ONLINE_STORE", "sqlite")

    config = Config()
    ticker = "RELIANCE.NS"
    start_date = "2024-01-01"

    print(f"[1/4] Fetching data for {ticker} from {start_date}...")
    df = fetch_ohlcv(ticker, start=start_date)
    print(f"Fetched shape: {df.shape}")

    feature_store_dir = Path("feature_store")
    parquet_path = feature_store_dir / "data" / "features.parquet"
    registry_path = feature_store_dir / "data" / "registry.db"
    print(f"[2/4] Feast parquet exists: {parquet_path.exists()}")
    print(f"[2/4] Feast registry exists: {registry_path.exists()}")

    print("[3/4] Building dataset...")
    scaler = StandardScaler()
    scaler.fit(df[config.features].values)
    dataset = StockDataset(
        data=df,
        scaler=scaler,
        context_len=config.context_len,
        pred_len=config.pred_len,
    )
    if len(dataset) == 0:
        raise RuntimeError("dataset is empty after sequence construction")
    print(f"Dataset length: {len(dataset)}")

    batch_size = min(config.batch_size, len(dataset))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("[4/4] Running one training epoch...")
    model = StockLSTM(input_size=config.input_size, pred_len=config.pred_len)
    _, best_loss = model_train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
    )
    print(f"Training smoke test passed. Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
