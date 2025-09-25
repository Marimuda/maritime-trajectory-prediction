#!/usr/bin/env python3
"""
Cache management utility for maritime trajectory prediction system.

Provides command-line tools for inspecting, managing, and optimizing the
hierarchical cache system.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cache_manager import CacheLevel, CacheManager


def format_size(size_bytes):
    """Format size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def cmd_info(args):
    """Show cache information."""
    cache_manager = CacheManager(args.cache_dir)
    info = cache_manager.get_info()

    print(f"Cache Directory: {info['cache_dir']}")
    print(f"Cache Version: {info['version']}")
    print(f"Total Files: {info['total']['files']}")
    print(f"Total Size: {format_size(info['total']['size_mb'] * 1024 * 1024)}")
    print()

    print("Cache Levels:")
    for level, stats in info["levels"].items():
        print(
            f"  {level.upper():10} | {stats['files']:4} files | {format_size(stats['size_mb'] * 1024 * 1024):>8}"
        )


def cmd_list(args):
    """List cache entries."""
    cache_manager = CacheManager(args.cache_dir)
    level = CacheLevel(args.level) if args.level else None
    entries = cache_manager.list_entries(level)

    if not entries:
        print("No cache entries found.")
        return

    # Print header
    print(
        f"{'KEY':<20} {'LEVEL':<10} {'SIZE':<8} {'FORMAT':<8} {'CREATED':<20} {'SOURCES'}"
    )
    print("-" * 100)

    # Print entries
    for entry in entries:
        key = entry["cache_key"][:20]
        level = entry["level"]
        size = format_size(entry["size_mb"] * 1024 * 1024)
        format_type = entry["format"]
        created = entry["created_at"][:19]  # Remove microseconds
        sources = ", ".join([Path(f).name for f in entry["source_files"]])[:30]

        print(
            f"{key:<20} {level:<10} {size:<8} {format_type:<8} {created:<20} {sources}"
        )


def cmd_clear(args):
    """Clear cache entries."""
    cache_manager = CacheManager(args.cache_dir)

    if args.level:
        level = CacheLevel(args.level)
        count = cache_manager.clear_level(level)
        print(f"Cleared {count} files from {level.value} cache.")
    elif args.all:
        if not args.force:
            response = input("This will delete ALL cache data. Are you sure? (y/N): ")
            if response.lower() != "y":
                print("Cancelled.")
                return

        count = cache_manager.clear_all()
        print(f"Cleared {count} files from all cache levels.")
    else:
        print("Error: Must specify either --level or --all")
        return


def cmd_validate(args):
    """Validate cache integrity."""
    cache_manager = CacheManager(args.cache_dir)
    entries = cache_manager.list_entries()

    valid_count = 0
    invalid_count = 0

    print("Validating cache entries...")

    for entry in entries:
        cache_key = entry["cache_key"]
        level = CacheLevel(entry["level"])
        source_files = entry["source_files"]

        try:
            # Try to load metadata
            metadata = cache_manager._load_metadata(cache_key, level)
            if metadata is None:
                print(f"  ❌ {cache_key}: Missing metadata")
                invalid_count += 1
                continue

            # Check if source files exist
            missing_sources = []
            for source_file in source_files:
                if not Path(source_file).exists():
                    missing_sources.append(source_file)

            if missing_sources:
                print(
                    f"  ❌ {cache_key}: Missing source files: {', '.join(missing_sources)}"
                )
                invalid_count += 1
                continue

            # Check cache file exists
            cache_path = cache_manager._get_cache_path(
                cache_key, level, metadata.format
            )
            if not cache_path.exists():
                print(f"  ❌ {cache_key}: Missing cache file: {cache_path}")
                invalid_count += 1
                continue

            if args.verbose:
                print(f"  ✅ {cache_key}: Valid")
            valid_count += 1

        except Exception as e:
            print(f"  ❌ {cache_key}: Error - {e}")
            invalid_count += 1

    print("\nValidation Results:")
    print(f"  Valid entries: {valid_count}")
    print(f"  Invalid entries: {invalid_count}")

    if invalid_count > 0:
        if args.cleanup:
            print("Cleaning up invalid entries...")
            # Implementation would go here
        else:
            print("Use --cleanup to remove invalid entries.")


def cmd_stats(args):
    """Show detailed cache statistics."""
    cache_manager = CacheManager(args.cache_dir)
    entries = cache_manager.list_entries()

    # Statistics by level
    level_stats = {}
    for entry in entries:
        level = entry["level"]
        if level not in level_stats:
            level_stats[level] = {"count": 0, "size": 0, "formats": {}}

        level_stats[level]["count"] += 1
        level_stats[level]["size"] += entry["size_mb"]

        format_type = entry["format"]
        if format_type not in level_stats[level]["formats"]:
            level_stats[level]["formats"][format_type] = 0
        level_stats[level]["formats"][format_type] += 1

    print("Cache Statistics by Level:")
    print("-" * 40)

    for level, stats in level_stats.items():
        print(f"\n{level.upper()}:")
        print(f"  Entries: {stats['count']}")
        print(f"  Total Size: {format_size(stats['size'] * 1024 * 1024)}")
        print(
            f"  Formats: {', '.join(f'{fmt}({count})' for fmt, count in stats['formats'].items())}"
        )


def main():
    parser = argparse.ArgumentParser(description="Cache management utility")
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="Cache directory path (default: data/cache)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show cache information")

    # List command
    list_parser = subparsers.add_parser("list", help="List cache entries")
    list_parser.add_argument(
        "--level", choices=[l.value for l in CacheLevel], help="Filter by cache level"
    )

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cache entries")
    clear_group = clear_parser.add_mutually_exclusive_group(required=True)
    clear_group.add_argument(
        "--level", choices=[l.value for l in CacheLevel], help="Clear specific level"
    )
    clear_group.add_argument("--all", action="store_true", help="Clear all cache data")
    clear_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate cache integrity")
    validate_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    validate_parser.add_argument(
        "--cleanup", action="store_true", help="Remove invalid entries"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show detailed statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    command_map = {
        "info": cmd_info,
        "list": cmd_list,
        "clear": cmd_clear,
        "validate": cmd_validate,
        "stats": cmd_stats,
    }

    command_map[args.command](args)


if __name__ == "__main__":
    main()
