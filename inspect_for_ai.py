"""
AI Inspection Checkpoint Utilities
为AI代码分析添加变量检查点
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from typing import Any, Dict, Union


def inspect_for_ai(var: Any, var_name: str,
                   max_preview: int = 5,
                   save_to_file: bool = False,
                   output_dir: str = ".workflow/.scratchpad/") -> None:
    """
    Print comprehensive information about a variable for AI code analysis.
    为AI代码分析打印变量的完整信息

    Parameters:
    -----------
    var : Any
        The variable to inspect
    var_name : str
        Name of the variable (for labeling)
    max_preview : int
        Maximum number of items to preview for collections
    save_to_file : bool
        Whether to save the inspection report to a file
    output_dir : str
        Directory to save inspection reports
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    separator = "=" * 80

    report_lines = []
    report_lines.append(f"\n{separator}")
    report_lines.append(f"🔍 AI CHECKPOINT: {var_name}")
    report_lines.append(f"Timestamp: {timestamp}")
    report_lines.append(separator)

    # Basic type info
    var_type = type(var).__name__
    report_lines.append(f"\n📌 Type: {var_type}")

    # Handle different types
    if isinstance(var, np.ndarray):
        _inspect_numpy_array(var, report_lines, max_preview)
    elif isinstance(var, pd.DataFrame):
        _inspect_dataframe(var, report_lines, max_preview)
    elif isinstance(var, dict):
        _inspect_dict(var, report_lines, max_preview)
    elif isinstance(var, (list, tuple)):
        _inspect_sequence(var, report_lines, max_preview)
    elif isinstance(var, (int, float, str, bool)):
        report_lines.append(f"Value: {var}")
    else:
        # Generic inspection
        report_lines.append(f"Object: {str(var)[:200]}")
        if hasattr(var, '__dict__'):
            report_lines.append(f"Attributes: {list(var.__dict__.keys())[:max_preview]}")

    report_lines.append(f"\n{separator}\n")

    # Print report
    report = "\n".join(report_lines)
    print(report)

    # Optionally save to file
    if save_to_file:
        import os
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}inspect_{var_name}_{timestamp}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"💾 Inspection report saved to: {filename}")


def _inspect_numpy_array(arr: np.ndarray, report_lines: list, max_preview: int) -> None:
    """Inspect numpy array"""
    report_lines.append(f"Shape: {arr.shape}")
    report_lines.append(f"Dtype: {arr.dtype}")
    report_lines.append(f"Size: {arr.size} elements")
    report_lines.append(f"Memory: {arr.nbytes / 1024 / 1024:.2f} MB")

    if np.issubdtype(arr.dtype, np.number):
        report_lines.append(f"Min: {np.nanmin(arr):.6f}")
        report_lines.append(f"Max: {np.nanmax(arr):.6f}")
        report_lines.append(f"Mean: {np.nanmean(arr):.6f}")
        report_lines.append(f"Std: {np.nanstd(arr):.6f}")
        report_lines.append(f"NaN count: {np.isnan(arr).sum()}")

    # Preview data
    if arr.size > 0:
        report_lines.append(f"\n📊 Preview (first {max_preview} elements):")
        flat = arr.flatten()
        preview = flat[:min(max_preview, len(flat))]
        report_lines.append(str(preview))


def _inspect_dataframe(df: pd.DataFrame, report_lines: list, max_preview: int) -> None:
    """Inspect pandas DataFrame"""
    report_lines.append(f"Shape: {df.shape} (rows × cols)")
    report_lines.append(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    report_lines.append(f"\n📋 Columns ({len(df.columns)}):")
    report_lines.append(str(list(df.columns)))

    report_lines.append(f"\n📊 Data Types:")
    for col, dtype in df.dtypes.items():
        report_lines.append(f"  {col}: {dtype}")

    report_lines.append(f"\n📈 Numeric Summary:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols[:max_preview]:
            report_lines.append(f"  {col}:")
            report_lines.append(f"    Min: {df[col].min():.6f}")
            report_lines.append(f"    Max: {df[col].max():.6f}")
            report_lines.append(f"    Mean: {df[col].mean():.6f}")
            report_lines.append(f"    NaN: {df[col].isna().sum()}")

    report_lines.append(f"\n🔎 First {max_preview} rows:")
    report_lines.append(str(df.head(max_preview)))


def _inspect_dict(d: dict, report_lines: list, max_preview: int) -> None:
    """Inspect dictionary"""
    report_lines.append(f"Keys count: {len(d)}")
    report_lines.append(f"\n🔑 Keys (first {max_preview}):")
    keys_preview = list(d.keys())[:max_preview]
    report_lines.append(str(keys_preview))

    # Inspect nested structure
    if len(d) > 0:
        first_key = list(d.keys())[0]
        first_value = d[first_key]
        report_lines.append(f"\n📦 First value type: {type(first_value).__name__}")

        if isinstance(first_value, dict):
            report_lines.append(f"   Nested dict with {len(first_value)} keys")
            nested_keys = list(first_value.keys())[:max_preview]
            report_lines.append(f"   Nested keys: {nested_keys}")
        elif isinstance(first_value, (list, tuple)):
            report_lines.append(f"   Sequence with {len(first_value)} items")
        elif isinstance(first_value, np.ndarray):
            report_lines.append(f"   Array shape: {first_value.shape}")
        elif isinstance(first_value, pd.DataFrame):
            report_lines.append(f"   DataFrame shape: {first_value.shape}")


def _inspect_sequence(seq: Union[list, tuple], report_lines: list, max_preview: int) -> None:
    """Inspect list or tuple"""
    report_lines.append(f"Length: {len(seq)}")

    if len(seq) > 0:
        first_item = seq[0]
        report_lines.append(f"Item type: {type(first_item).__name__}")

        report_lines.append(f"\n📊 Preview (first {max_preview} items):")
        for i, item in enumerate(seq[:max_preview]):
            if isinstance(item, (np.ndarray, pd.DataFrame)):
                report_lines.append(f"  [{i}]: {type(item).__name__} shape={getattr(item, 'shape', 'N/A')}")
            else:
                item_str = str(item)
                if len(item_str) > 100:
                    item_str = item_str[:100] + "..."
                report_lines.append(f"  [{i}]: {item_str}")


def save_checkpoint_pickle(var: Any, var_name: str,
                           output_dir: str = ".workflow/.scratchpad/") -> str:
    """
    Save variable to pickle file for later inspection
    保存变量到pickle文件以便后续检查

    Returns:
        str: Path to saved file
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}checkpoint_{var_name}_{timestamp}.pkl"

    with open(filename, 'wb') as f:
        pickle.dump(var, f)

    print(f"💾 Checkpoint saved: {filename}")
    return filename


# Convenience function for nested dict structures
def inspect_nested_dict_structure(d: dict, var_name: str, max_depth: int = 3) -> None:
    """
    Recursively inspect nested dictionary structure
    递归检查嵌套字典结构
    """
    print(f"\n{'='*80}")
    print(f"🔍 NESTED STRUCTURE: {var_name}")
    print(f"{'='*80}\n")

    def _recurse(obj, depth=0, prefix=""):
        indent = "  " * depth

        if depth >= max_depth:
            print(f"{indent}{prefix}<max depth reached>")
            return

        if isinstance(obj, dict):
            print(f"{indent}{prefix}dict ({len(obj)} keys)")
            for i, (key, value) in enumerate(list(obj.items())[:3]):
                print(f"{indent}  [{key}]:")
                _recurse(value, depth + 1, "")
            if len(obj) > 3:
                print(f"{indent}  ... ({len(obj) - 3} more keys)")

        elif isinstance(obj, (list, tuple)):
            type_name = "list" if isinstance(obj, list) else "tuple"
            print(f"{indent}{prefix}{type_name} ({len(obj)} items)")
            if len(obj) > 0:
                print(f"{indent}  [0]:")
                _recurse(obj[0], depth + 1, "")

        elif isinstance(obj, np.ndarray):
            print(f"{indent}{prefix}ndarray: shape={obj.shape}, dtype={obj.dtype}")

        elif isinstance(obj, pd.DataFrame):
            print(f"{indent}{prefix}DataFrame: shape={obj.shape}")
            print(f"{indent}  Columns: {list(obj.columns)[:5]}")

        else:
            obj_str = str(obj)
            if len(obj_str) > 80:
                obj_str = obj_str[:80] + "..."
            print(f"{indent}{prefix}{type(obj).__name__}: {obj_str}")

    _recurse(d)
    print(f"\n{'='*80}\n")
