"""Market data acquisition, Parquet incremental caching, and causal feature engineering."""

from universal_trader.data.alpaca import AlpacaDataProvider
from universal_trader.data.binance import BinanceDataProvider
from universal_trader.data.cache import ParquetCacheStore
from universal_trader.data.features import FeaturePipeline
from universal_trader.data.local import CsvParquetProvider
from universal_trader.data.utils import DataProviderError, normalize_ohlcv_dataframe
from universal_trader.data.yahoo import YahooDataProvider

__all__ = [
    "AlpacaDataProvider",
    "BinanceDataProvider",
    "CsvParquetProvider",
    "DataProviderError",
    "FeaturePipeline",
    "ParquetCacheStore",
    "YahooDataProvider",
    "normalize_ohlcv_dataframe",
]
