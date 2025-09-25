"""
Advanced caching system for maritime data preprocessing pipeline.

This module provides a hierarchical caching system that caches data at multiple
stages of the preprocessing pipeline to minimize redundant computation.
"""

import hashlib
import json
import logging
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels in the preprocessing pipeline."""

    RAW = "raw"  # Parsed AIS messages
    CLEANED = "cleaned"  # After validation and filtering
    FEATURES = "features"  # Engineered features per task
    SEQUENCES = "sequences"  # Final sequences per configuration


class CacheFormat(Enum):
    """Supported cache storage formats."""

    PICKLE = "pkl"
    PARQUET = "parquet"
    NUMPY = "npy"
    JSON = "json"


@dataclass
class CacheMetadata:
    """Metadata for cached data."""

    level: CacheLevel
    created_at: datetime
    cache_key: str
    source_files: list[str]
    source_hashes: dict[str, str]
    parameters: dict[str, Any]
    format: CacheFormat
    size_bytes: int
    version: str = "1.0"


class CacheManager:
    """
    Hierarchical cache manager for maritime data preprocessing.

    Provides multi-level caching with automatic invalidation and dependency tracking.
    """

    def __init__(self, cache_dir: str | Path = "data/cache", version: str = "1.0"):
        """
        Initialize cache manager.

        Args:
            cache_dir: Base directory for cache storage
            version: Cache system version for migration support
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Create level-specific subdirectories
        for level in CacheLevel:
            (self.cache_dir / level.value).mkdir(exist_ok=True)

    def _compute_file_hash(self, file_path: str | Path) -> str:
        """Compute hash of file for change detection."""
        file_path = Path(file_path)
        if not file_path.exists():
            return ""

        stat = file_path.stat()
        # Use modification time and size for quick hash
        return hashlib.md5(f"{stat.st_mtime}_{stat.st_size}".encode()).hexdigest()

    def _compute_params_hash(self, params: dict[str, Any]) -> str:
        """Compute hash of parameters dictionary."""
        # Sort keys for consistent hashing
        params_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(params_str.encode()).hexdigest()[:12]

    def _generate_cache_key(
        self, level: CacheLevel, source_files: list[str | Path], params: dict[str, Any]
    ) -> str:
        """Generate hierarchical cache key."""
        # File hashes
        file_hashes = []
        for file_path in source_files:
            file_hash = self._compute_file_hash(file_path)
            file_hashes.append(file_hash)

        # Combine file and parameter hashes
        files_hash = hashlib.md5("".join(file_hashes).encode()).hexdigest()[:8]
        params_hash = self._compute_params_hash(params)

        return f"{level.value}_{files_hash}_{params_hash}"

    def _get_cache_path(
        self, cache_key: str, level: CacheLevel, format: CacheFormat
    ) -> Path:
        """Get path for cache file."""
        return self.cache_dir / level.value / f"{cache_key}.{format.value}"

    def _get_metadata_path(self, cache_key: str, level: CacheLevel) -> Path:
        """Get path for metadata file."""
        return self.cache_dir / level.value / f"{cache_key}_meta.json"

    def _save_metadata(self, metadata: CacheMetadata, level: CacheLevel) -> None:
        """Save cache metadata."""
        metadata_path = self._get_metadata_path(metadata.cache_key, level)
        metadata_dict = asdict(metadata)
        metadata_dict["created_at"] = metadata.created_at.isoformat()
        metadata_dict["level"] = metadata.level.value
        metadata_dict["format"] = metadata.format.value

        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2, default=str)

    def _load_metadata(self, cache_key: str, level: CacheLevel) -> CacheMetadata | None:
        """Load cache metadata."""
        metadata_path = self._get_metadata_path(cache_key, level)
        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            metadata_dict = json.load(f)

        metadata_dict["created_at"] = datetime.fromisoformat(
            metadata_dict["created_at"]
        )
        metadata_dict["level"] = CacheLevel(metadata_dict["level"])
        metadata_dict["format"] = CacheFormat(metadata_dict["format"])

        return CacheMetadata(**metadata_dict)

    def _is_cache_valid(
        self, metadata: CacheMetadata, source_files: list[str | Path]
    ) -> bool:
        """Check if cached data is still valid."""
        # Check if source files have changed
        for file_path in source_files:
            file_path_str = str(file_path)
            if file_path_str not in metadata.source_hashes:
                self.logger.debug(f"Cache invalid: new source file {file_path_str}")
                return False

            current_hash = self._compute_file_hash(file_path)
            if current_hash != metadata.source_hashes[file_path_str]:
                self.logger.debug(f"Cache invalid: {file_path_str} changed")
                return False

        return True

    def _save_data(self, data: Any, cache_path: Path, format: CacheFormat) -> int:
        """Save data in specified format."""
        if format == CacheFormat.PICKLE:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif format == CacheFormat.PARQUET:
            if isinstance(data, pd.DataFrame):
                data.to_parquet(cache_path)
            else:
                raise ValueError(f"Cannot save {type(data)} as Parquet")
        elif format == CacheFormat.NUMPY:
            if isinstance(data, np.ndarray):
                np.save(cache_path, data)
            else:
                raise ValueError(f"Cannot save {type(data)} as NumPy")
        elif format == CacheFormat.JSON:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported cache format: {format}")

        return cache_path.stat().st_size

    def _load_data(self, cache_path: Path, format: CacheFormat) -> Any:
        """Load data from cache file."""
        if format == CacheFormat.PICKLE:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        elif format == CacheFormat.PARQUET:
            return pd.read_parquet(cache_path)
        elif format == CacheFormat.NUMPY:
            return np.load(cache_path, allow_pickle=True)
        elif format == CacheFormat.JSON:
            with open(cache_path) as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported cache format: {format}")

    def get(
        self,
        level: CacheLevel,
        source_files: list[str | Path],
        params: dict[str, Any],
        format: CacheFormat = CacheFormat.PICKLE,
    ) -> Any | None:
        """
        Retrieve data from cache if available and valid.

        Args:
            level: Cache level
            source_files: Source files that data depends on
            params: Parameters used to generate the data
            format: Expected cache format

        Returns:
            Cached data or None if not available/invalid
        """
        cache_key = self._generate_cache_key(level, source_files, params)
        cache_path = self._get_cache_path(cache_key, level, format)

        if not cache_path.exists():
            self.logger.debug(f"Cache miss: {cache_key}")
            return None

        # Load and validate metadata
        metadata = self._load_metadata(cache_key, level)
        if metadata is None:
            self.logger.warning(f"Cache metadata missing for {cache_key}")
            return None

        if not self._is_cache_valid(metadata, source_files):
            self.logger.info(f"Cache invalid: {cache_key}")
            self.invalidate(cache_key, level)
            return None

        # Load data
        try:
            data = self._load_data(cache_path, format)
            self.logger.info(f"Cache hit: {cache_key}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load cache {cache_key}: {e}")
            return None

    def put(
        self,
        data: Any,
        level: CacheLevel,
        source_files: list[str | Path],
        params: dict[str, Any],
        format: CacheFormat = CacheFormat.PICKLE,
    ) -> str:
        """
        Store data in cache.

        Args:
            data: Data to cache
            level: Cache level
            source_files: Source files that data depends on
            params: Parameters used to generate the data
            format: Cache storage format

        Returns:
            Cache key for the stored data
        """
        cache_key = self._generate_cache_key(level, source_files, params)
        cache_path = self._get_cache_path(cache_key, level, format)

        # Save data
        try:
            size_bytes = self._save_data(data, cache_path, format)
        except Exception as e:
            self.logger.error(f"Failed to save cache {cache_key}: {e}")
            raise

        # Create metadata
        source_hashes = {str(f): self._compute_file_hash(f) for f in source_files}
        metadata = CacheMetadata(
            level=level,
            created_at=datetime.now(),
            cache_key=cache_key,
            source_files=[str(f) for f in source_files],
            source_hashes=source_hashes,
            parameters=params,
            format=format,
            size_bytes=size_bytes,
            version=self.version,
        )

        self._save_metadata(metadata, level)
        self.logger.info(
            f"Cached data: {cache_key} ({size_bytes / 1024 / 1024:.1f} MB)"
        )
        return cache_key

    def invalidate(self, cache_key: str, level: CacheLevel) -> None:
        """Remove specific cache entry."""
        # Remove data files for all formats
        for format in CacheFormat:
            cache_path = self._get_cache_path(cache_key, level, format)
            if cache_path.exists():
                cache_path.unlink()

        # Remove metadata
        metadata_path = self._get_metadata_path(cache_key, level)
        if metadata_path.exists():
            metadata_path.unlink()

        self.logger.info(f"Invalidated cache: {cache_key}")

    def clear_level(self, level: CacheLevel) -> int:
        """Clear all cache entries at specified level."""
        level_dir = self.cache_dir / level.value
        if not level_dir.exists():
            return 0

        count = 0
        for file_path in level_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
                count += 1

        self.logger.info(f"Cleared {count} files from {level.value} cache")
        return count

    def clear_all(self) -> int:
        """Clear entire cache."""
        total_count = 0
        for level in CacheLevel:
            total_count += self.clear_level(level)
        return total_count

    def get_info(self) -> dict[str, Any]:
        """Get cache statistics and information."""
        info = {"cache_dir": str(self.cache_dir), "version": self.version, "levels": {}}

        total_size = 0
        total_files = 0

        for level in CacheLevel:
            level_dir = self.cache_dir / level.value
            level_files = 0
            level_size = 0

            if level_dir.exists():
                for file_path in level_dir.iterdir():
                    if file_path.is_file() and not file_path.name.endswith(
                        "_meta.json"
                    ):
                        level_files += 1
                        level_size += file_path.stat().st_size

            info["levels"][level.value] = {
                "files": level_files,
                "size_mb": level_size / 1024 / 1024,
            }

            total_files += level_files
            total_size += level_size

        info["total"] = {"files": total_files, "size_mb": total_size / 1024 / 1024}

        return info

    def list_entries(self, level: CacheLevel | None = None) -> list[dict[str, Any]]:
        """List cache entries with metadata."""
        entries = []
        levels = [level] if level else list(CacheLevel)

        for lvl in levels:
            level_dir = self.cache_dir / lvl.value
            if not level_dir.exists():
                continue

            for metadata_file in level_dir.glob("*_meta.json"):
                cache_key = metadata_file.stem.replace("_meta", "")
                metadata = self._load_metadata(cache_key, lvl)

                if metadata:
                    entry = {
                        "cache_key": cache_key,
                        "level": lvl.value,
                        "created_at": metadata.created_at.isoformat(),
                        "size_mb": metadata.size_bytes / 1024 / 1024,
                        "format": metadata.format.value,
                        "source_files": metadata.source_files,
                    }
                    entries.append(entry)

        return sorted(entries, key=lambda x: x["created_at"], reverse=True)
