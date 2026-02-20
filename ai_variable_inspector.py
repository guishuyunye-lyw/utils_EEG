"""
AI Variable Inspector - 将Python运行时状态转化为AI可读的结构化信息
专门设计用于Jupyter Notebook环境，帮助AI理解EEG分析中的数据流动

特性：
    - 智能显示策略：根据数据大小自动调整显示详细程度
    - 完全确定性：相同输入产生相同输出，适合科研环境
    - 零成本：无需API调用，即时响应
    - 可配置详细程度：支持 minimal/auto/normal/full 四种模式

使用方法：
    from utils_EEG.ai_variable_inspector import inspect_for_ai

    # 基础使用 - 自动调整显示策略
    epochs = mne.read_epochs(...)
    inspect_for_ai(epochs, name="epochs")

    # 指定详细程度
    inspect_for_ai(large_df, name="results", verbosity="minimal")  # 仅统计
    inspect_for_ai(small_df, name="metadata", verbosity="full")    # 完整数据
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from pathlib import Path

# 延迟导入mne，避免在没有mne环境时导入失败
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


def inspect_for_ai(var: Any, name: str = "variable", max_depth: int = 2, verbosity: str = "auto") -> None:
    """
    打印AI agent需要的变量结构化信息

    Parameters
    ----------
    var : Any
        要检查的变量
    name : str
        变量名称（用于输出标识）
    max_depth : int
        嵌套字典/对象的最大检查深度
    verbosity : str
        详细程度控制 ("auto", "minimal", "normal", "full")
        - "auto": 根据数据大小自动调整（默认）
        - "minimal": 最少信息，仅显示统计摘要
        - "normal": 标准信息
        - "full": 完整信息，显示所有数据
    """
    print(f"\n{'='*80}")
    print(f"🤖 AI VARIABLE INSPECTION: {name}")
    print(f"{'='*80}\n")

    # 1. 基础类型信息
    print(f"📌 Type: {type(var).__module__}.{type(var).__name__}")
    print(f"📌 Memory: {_get_size_mb(var):.2f} MB")

    # 2. 根据类型分发检查逻辑
    if isinstance(var, (np.ndarray, list, tuple)):
        _inspect_array_like(var, name, verbosity)
    elif isinstance(var, pd.DataFrame):
        _inspect_dataframe(var, name, verbosity)
    elif isinstance(var, dict):
        _inspect_dict(var, name, max_depth, verbosity=verbosity)
    elif _is_mne_object(var):
        _inspect_mne_object(var, name)
    elif hasattr(var, '__dict__'):
        _inspect_custom_object(var, name, max_depth)
    else:
        _inspect_primitive(var, name)

    # 3. 数据流提示
    print(f"\n💡 AI Usage Hints:")
    _print_usage_hints(var, name)

    print(f"\n{'='*80}\n")


def _get_size_mb(obj: Any) -> float:
    """估算对象内存占用"""
    try:
        if isinstance(obj, np.ndarray):
            return obj.nbytes / (1024**2)
        elif isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum() / (1024**2)
        else:
            return 0.0  # 其他类型暂不精确计算
    except:
        return 0.0


def _inspect_array_like(var: Any, name: str, verbosity: str = "auto") -> None:
    """检查数组类对象 - 智能采样策略"""
    if isinstance(var, np.ndarray):
        arr = var
    else:
        arr = np.array(var) if len(var) > 0 else np.array([])

    print(f"\n📊 Array Structure:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Dimensions: {arr.ndim}D")
    print(f"  Total elements: {arr.size}")

    if arr.size == 0:
        print(f"  (Empty array)")
        return

    # 检查是否为数值类型
    is_numeric = np.issubdtype(arr.dtype, np.number)

    if is_numeric:
        # 基础统计
        print(f"\n  统计信息:")
        print(f"    Range: [{np.min(arr):.4f}, {np.max(arr):.4f}]")
        print(f"    Mean: {np.mean(arr):.4f}, Std: {np.std(arr):.4f}")

        # 根据大小显示不同详细程度
        if arr.size > 10000 or verbosity == "full":
            # 大数组：添加分位数
            print(f"    Quantiles:")
            print(f"      25%: {np.percentile(arr, 25):.4f}")
            print(f"      50%: {np.percentile(arr, 50):.4f}")
            print(f"      75%: {np.percentile(arr, 75):.4f}")
    else:
        print(f"  (Non-numeric array, skipping statistics)")

    # 采样显示
    print(f"\n  采样值:")
    if arr.size <= 20 or verbosity == "full":
        # 小数组：显示全部
        print(f"    All values: {arr.flat[:20]}")
    elif arr.size <= 100:
        # 中等数组：显示头尾
        print(f"    First 5: {arr.flat[:5]}")
        print(f"    Last 5: {arr.flat[-5:]}")
    else:
        # 大数组：稀疏采样
        print(f"    First 3: {arr.flat[:3]}")
        print(f"    Middle 3: {arr.flat[arr.size//2-1:arr.size//2+2]}")
        print(f"    Last 3: {arr.flat[-3:]}")


def _inspect_dataframe(df: pd.DataFrame, name: str, verbosity: str = "auto") -> None:
    """检查DataFrame - 智能显示策略"""
    print(f"\n📊 DataFrame Structure:")
    print(f"  Shape: {df.shape} (rows × columns)")
    print(f"  Columns: {list(df.columns)}")

    # 数据类型信息
    print(f"\n  数据类型:")
    print(f"    {df.dtypes.to_string(max_rows=10)}")

    # 缺失值统计
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n  缺失值: {missing[missing > 0].to_dict()}")
    else:
        print(f"\n  缺失值: 无")

    # 根据大小智能显示数据
    n_rows = df.shape[0]

    if verbosity == "minimal":
        # 最小模式：仅显示统计
        print(f"\n  数据摘要:")
        print(df.describe().to_string())

    elif n_rows <= 10 or verbosity == "full":
        # 小表格或完整模式：显示全部数据
        print(f"\n  完整数据:")
        print(df.to_string())

    elif n_rows <= 100:
        # 中等表格：显示头尾
        print(f"\n  前5行:")
        print(df.head(5).to_string())
        print(f"\n  后5行:")
        print(df.tail(5).to_string())

        # 数值列的统计摘要
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n  数值列统计:")
            print(df[numeric_cols].describe().to_string())

    else:
        # 大表格：统计 + 采样
        print(f"\n  数据摘要:")
        print(df.describe().to_string())

        print(f"\n  前3行:")
        print(df.head(3).to_string())

        # 随机采样（固定种子保证可重复性）
        print(f"\n  随机采样3行 (种子=42):")
        sample_size = min(3, len(df))
        print(df.sample(n=sample_size, random_state=42).to_string())


def _inspect_dict(d: dict, name: str, max_depth: int, _current_depth: int = 0, verbosity: str = "auto") -> None:
    """检查字典 - 智能展开策略"""
    print(f"\n📦 Dictionary Structure (depth {_current_depth}):")

    n_keys = len(d)
    print(f"  总键数: {n_keys}")

    # 根据字典大小决定显示策略
    if n_keys <= 20 or verbosity == "full":
        print(f"  所有键: {list(d.keys())}")
    else:
        print(f"  前10个键: {list(d.keys())[:10]}")
        print(f"  后10个键: {list(d.keys())[-10:]}")

    # 统计值类型分布
    type_counts = {}
    for value in d.values():
        vtype = type(value).__name__
        type_counts[vtype] = type_counts.get(vtype, 0) + 1
    print(f"\n  值类型分布: {type_counts}")

    # 智能展开重要的键值对
    print(f"\n  键值详情:")
    items_to_show = list(d.items())[:20] if n_keys > 20 else list(d.items())

    for key, value in items_to_show:
        value_type = type(value).__name__

        if isinstance(value, np.ndarray):
            shape = value.shape
            dtype = value.dtype
            print(f"    '{key}': {value_type} {shape} {dtype}")

            # 对于重要的数组，显示简要统计
            if value.size > 0 and np.issubdtype(value.dtype, np.number):
                print(f"      → Range: [{np.min(value):.4f}, {np.max(value):.4f}]")

        elif isinstance(value, list):
            print(f"    '{key}': {value_type} len={len(value)}")
            if len(value) > 0:
                first_elem = value[0]
                first_type = type(first_elem).__name__
                print(f"      → First element type: {first_type}")

                # 如果是 DataFrame list，显示更多细节
                if isinstance(first_elem, pd.DataFrame):
                    print(f"      → Sample DataFrame [0]:")
                    print(f"         Shape: {first_elem.shape}")

                    # 显示所有列名（如果列数较多，分行显示）
                    cols = list(first_elem.columns)
                    if len(cols) <= 10:
                        print(f"         Columns: {cols}")
                    else:
                        print(f"         Columns ({len(cols)} total):")
                        # 每行显示5个列名
                        for i in range(0, len(cols), 5):
                            chunk = cols[i:i+5]
                            print(f"           {chunk}")

                    # 检查是否所有 DataFrame 结构相同
                    if len(value) > 1 and all(isinstance(v, pd.DataFrame) for v in value[:min(10, len(value))]):
                        shapes = [v.shape for v in value[:min(10, len(value))]]
                        all_same = all(s == shapes[0] for s in shapes)
                        print(f"         All DataFrames same structure: {all_same}")
                        if not all_same:
                            print(f"         Shape variations: {set(shapes)}")

                # 如果是 ndarray list，显示更多细节
                elif isinstance(first_elem, np.ndarray):
                    print(f"      → Sample array [0]: shape {first_elem.shape}, dtype {first_elem.dtype}")
                    if len(value) > 1:
                        shapes = [v.shape for v in value[:min(10, len(value))] if isinstance(v, np.ndarray)]
                        all_same = all(s == shapes[0] for s in shapes)
                        print(f"         All arrays same shape: {all_same}")
                        if not all_same:
                            print(f"         Shape variations: {set(shapes)}")

        elif isinstance(value, pd.DataFrame):
            print(f"    '{key}': DataFrame {value.shape}")
            print(f"      → Columns: {list(value.columns)[:5]}{'...' if len(value.columns) > 5 else ''}")

            # 显示数据类型分布
            dtype_counts = value.dtypes.value_counts().to_dict()
            print(f"      → Dtypes: {dtype_counts}")

            # 显示缺失值情况
            missing_count = value.isnull().sum().sum()
            if missing_count > 0:
                print(f"      → Missing values: {missing_count} total")

            # 显示数据预览（前2行）
            if verbosity != "minimal" and value.shape[0] > 0:
                print(f"      → Preview (first 2 rows):")
                preview = value.head(2).to_string(max_cols=5, max_colwidth=20)
                for line in preview.split('\n'):
                    print(f"         {line}")

        elif isinstance(value, dict) and _current_depth < max_depth:
            print(f"    '{key}': dict with {len(value)} items")
            # 递归展开嵌套字典
            _inspect_dict(value, f"{name}['{key}']", max_depth, _current_depth + 1, verbosity)

        else:
            # 其他类型
            if isinstance(value, (int, float, str, bool)) and verbosity != "minimal":
                print(f"    '{key}': {value_type} = {value}")
            else:
                print(f"    '{key}': {value_type}")

    if n_keys > 20 and verbosity != "full":
        print(f"  ... (省略 {n_keys - 20} 个键)")


def _is_mne_object(obj: Any) -> bool:
    """判断是否为MNE对象"""
    if not HAS_MNE:
        return False
    return any(base.__module__.startswith('mne') for base in type(obj).__mro__)


def _inspect_mne_object(obj: Any, name: str) -> None:
    """检查MNE对象（Epochs, Evoked, Raw等）"""
    print(f"\n🧠 MNE Object Structure:")

    # 通用MNE属性
    if hasattr(obj, 'info'):
        info = obj.info
        print(f"  Channels: {len(info['ch_names'])} ({info['ch_names'][:5]}...)")
        print(f"  Sampling rate: {info['sfreq']} Hz")

    # Epochs特有
    if hasattr(obj, 'events'):
        print(f"  Events: {len(obj.events)} trials")
        print(f"  Event IDs: {obj.event_id}")
        print(f"  Time range: [{obj.tmin}, {obj.tmax}] sec")
        if hasattr(obj, '_data'):
            print(f"  Data shape: {obj._data.shape} (epochs × channels × timepoints)")

    # Evoked特有
    elif hasattr(obj, 'nave'):
        print(f"  Averaged trials: {obj.nave}")
        print(f"  Time range: [{obj.times[0]:.3f}, {obj.times[-1]:.3f}] sec")
        print(f"  Data shape: {obj.data.shape} (channels × timepoints)")

    # Connectivity结果
    elif hasattr(obj, 'get_data'):
        try:
            data = obj.get_data()
            print(f"  Data shape: {data.shape}")
            print(f"  Method: {obj.method if hasattr(obj, 'method') else 'unknown'}")
        except:
            pass

    # Metadata
    if hasattr(obj, 'metadata') and obj.metadata is not None:
        print(f"  Metadata: {obj.metadata.shape[1]} columns")
        print(f"    Columns: {list(obj.metadata.columns)}")


def _inspect_custom_object(obj: Any, name: str, max_depth: int) -> None:
    """检查自定义对象"""
    print(f"\n🔧 Custom Object Attributes:")

    attrs = {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}

    for attr_name, attr_value in list(attrs.items())[:10]:  # 限制输出数量
        attr_type = type(attr_value).__name__

        if isinstance(attr_value, np.ndarray):
            print(f"  {attr_name}: {attr_type} {attr_value.shape}")
        elif isinstance(attr_value, (list, tuple)):
            print(f"  {attr_name}: {attr_type} len={len(attr_value)}")
        else:
            print(f"  {attr_name}: {attr_type}")


def _inspect_primitive(var: Any, name: str) -> None:
    """检查基础类型"""
    print(f"\n📝 Value: {var}")


def _print_usage_hints(var: Any, name: str) -> None:
    """根据变量类型提供AI使用建议"""
    var_type = type(var).__name__

    if isinstance(var, np.ndarray):
        print(f"  - Access data: {name}[index] or {name}.flatten()")
        print(f"  - Shape manipulation: {name}.reshape(...)")

    elif isinstance(var, pd.DataFrame):
        print(f"  - Access columns: {name}['column_name']")
        print(f"  - Filter rows: {name}[{name}['col'] > value]")
        print(f"  - Groupby: {name}.groupby('col').mean()")

    elif isinstance(var, dict):
        print(f"  - Access values: {name}['key']")
        print(f"  - Iterate: for k, v in {name}.items()")

    elif _is_mne_object(var):
        if hasattr(var, 'get_data'):
            print(f"  - Extract data: {name}.get_data()")
        if hasattr(var, 'crop'):
            print(f"  - Crop time: {name}.crop(tmin, tmax)")
        if hasattr(var, 'apply_baseline'):
            print(f"  - Baseline: {name}.apply_baseline((tmin, tmax))")
        if hasattr(var, 'metadata'):
            print(f"  - Filter by metadata: {name}[{name}.metadata['condition'] == 'M']")


# ============================================================================
# 批量检查工具
# ============================================================================

def batch_inspect(variables: Dict[str, Any], max_depth: int = 2) -> None:
    """
    批量检查多个变量

    Parameters
    ----------
    variables : dict
        {'变量名': 变量值} 的字典
    max_depth : int
        检查深度

    Example
    -------
    >>> batch_inspect({
    ...     'epochs': epochs,
    ...     'connectivity': conn_results,
    ...     'behavior_df': df
    ... })
    """
    for name, var in variables.items():
        inspect_for_ai(var, name=name, max_depth=max_depth)


# ============================================================================
# 数据流追踪装饰器
# ============================================================================

def track_data_flow(func):
    """
    装饰器：自动追踪函数输入输出

    Example
    -------
    >>> @track_data_flow
    ... def process_epochs(epochs):
    ...     return epochs.crop(0, 1)
    """
    def wrapper(*args, **kwargs):
        print(f"\n🔄 FUNCTION CALL: {func.__name__}")
        print(f"{'='*80}")

        # 输入检查
        print("📥 INPUTS:")
        for i, arg in enumerate(args):
            inspect_for_ai(arg, name=f"arg{i}")
        for key, val in kwargs.items():
            inspect_for_ai(val, name=key)

        # 执行函数
        result = func(*args, **kwargs)

        # 输出检查
        print("\n📤 OUTPUT:")
        inspect_for_ai(result, name="return_value")

        return result

    return wrapper


# ============================================================================
# Notebook环境快捷函数
# ============================================================================

def quick_check(*vars_with_names):
    """
    快速检查多个变量（简化版）

    Example
    -------
    >>> quick_check(
    ...     ('epochs', epochs),
    ...     ('df', behavior_df)
    ... )
    """
    for name, var in vars_with_names:
        print(f"\n🔍 {name}: {type(var).__name__}", end="")

        if isinstance(var, np.ndarray):
            print(f" {var.shape} {var.dtype}")
        elif isinstance(var, pd.DataFrame):
            print(f" {var.shape}")
        elif hasattr(var, '_data'):
            print(f" {var._data.shape}")
        else:
            print()


if __name__ == "__main__":
    # 测试示例 - 展示智能显示策略
    print("=" * 80)
    print("AI Variable Inspector - Enhanced Test Mode")
    print("=" * 80)

    # 测试1: 小数组 - 显示完整信息
    print("\n\n【测试1】小数组 (自动显示完整)")
    small_array = np.array([1, 2, 3, 4, 5])
    inspect_for_ai(small_array, name="small_array")

    # 测试2: 大数组 - 智能采样
    print("\n\n【测试2】大数组 (智能采样)")
    large_array = np.random.randn(10, 64, 500)
    inspect_for_ai(large_array, name="eeg_data")

    # 测试3: 小DataFrame - 显示完整
    print("\n\n【测试3】小DataFrame (显示完整)")
    small_df = pd.DataFrame({
        'subject': ['pre001', 'pre002'],
        'condition': ['M', 'S'],
        'accuracy': [0.85, 0.92]
    })
    inspect_for_ai(small_df, name="small_behavior_data")

    # 测试4: 大DataFrame - 统计+采样
    print("\n\n【测试4】大DataFrame (统计+采样)")
    large_df = pd.DataFrame({
        'subject': [f'pre{i:03d}' for i in range(200)],
        'condition': ['M', 'S'] * 100,
        'accuracy': np.random.rand(200),
        'rt': np.random.randn(200) * 100 + 500
    })
    inspect_for_ai(large_df, name="large_behavior_data")

    # 测试5: 字典 - 智能展开
    print("\n\n【测试5】字典 (智能展开)")
    test_dict = {
        'connectivity': np.random.randn(64, 64),
        'freqs': np.linspace(4, 30, 27),
        'times': np.linspace(-0.5, 1.5, 500),
        'metadata': {'n_subjects': 10, 'condition': 'M'}
    }
    inspect_for_ai(test_dict, name="analysis_results")

    # 测试6: verbosity参数
    print("\n\n【测试6】verbosity='minimal' (仅统计)")
    inspect_for_ai(large_df, name="large_df_minimal", verbosity="minimal")

    print("\n\n【测试6】verbosity='full' (完整数据)")
    inspect_for_ai(small_df, name="small_df_full", verbosity="full")

    print("\n\n" + "=" * 80)
    print("测试完成！所有功能正常工作。")
    print("=" * 80)
