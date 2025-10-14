import time
import tracemalloc
import pandas as pd
from typing import Tuple

from preprocess_optimized import optimized_preprocess


def original_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    A slower, apply-heavy baseline that mirrors the original notebook-style code
    for fair comparison.
    """
    dataset = df.copy()

    # Parse Ram and Weight
    dataset['Ram'] = dataset['Ram'].str.replace('GB', '').astype('int32')
    dataset['Weight'] = dataset['Weight'].str.replace('kg', '').astype('float64')

    # Company consolidation via Python function + apply
    def add_company(inpt):
        if inpt == 'Samsung' or inpt == 'Razer' or inpt == 'Mediacom' or inpt == 'Microsoft' or inpt == 'Xiaomi' or inpt == 'Vero' or inpt == 'Chuwi' or inpt == 'Google' or inpt == 'Fujitsu' or inpt == 'LG' or inpt == 'Huawei':
            return 'Other'
        else:
            return inpt
    dataset['Company'] = dataset['Company'].apply(add_company)

    # Touchscreen and IPS via apply + lambda
    dataset['Touchscreen'] = dataset['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    dataset['IPS'] = dataset['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

    # CPU name processing with apply
    dataset['Cpu_name'] = dataset['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))

    def set_processor(name):
        if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
            return name
        else:
            if name.split()[0] == 'AMD':
                return 'AMD'
            else:
                return 'Other'
    dataset['Cpu_name'] = dataset['Cpu_name'].apply(set_processor)

    # GPU name and filter
    dataset['Gpu_name'] = dataset['Gpu'].apply(lambda x: " ".join(x.split()[0:1]))
    dataset = dataset[dataset['Gpu_name'] != 'ARM']

    # OS mapping via function
    def set_os(inpt):
        if inpt == 'Windows 10' or inpt == 'Windows 7' or inpt == 'Windows 10 S':
            return 'Windows'
        elif inpt == 'macOS' or inpt == 'Mac OS X':
            return 'Mac'
        elif inpt == 'Linux':
            return inpt
        else:
            return 'Other'
    dataset['OpSys'] = dataset['OpSys'].apply(set_os)

    # Drop and dummies
    dataset = dataset.drop(columns=['laptop_ID','Inches','Product','ScreenResolution','Cpu','Gpu'])
    dataset = pd.get_dummies(dataset)

    X = dataset.drop('Price_euros', axis=1)
    y = dataset['Price_euros']
    return X, y


def measure(func, df: pd.DataFrame, repeats: int = 3):
    # Warmup
    func(df.copy(deep=True))

    times = []
    peaks = []
    for _ in range(repeats):
        snapshot_df = df.copy(deep=True)
        tracemalloc.start()
        t0 = time.perf_counter()
        _ = func(snapshot_df)
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(elapsed)
        peaks.append(peak)
    return sum(times) / len(times), max(peaks)


def main():
    df = pd.read_csv('laptop_price.csv', encoding='latin-1')

    t_orig, m_orig = measure(original_preprocess, df)
    t_opt, m_opt = measure(optimized_preprocess, df)

    def fmt_bytes(n):
        for unit in ['B','KB','MB','GB']:
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} TB"

    print('Preprocessing micro-benchmark (avg over 3 runs):')
    print(f"- Original:  time={t_orig:.4f}s, peak_mem={fmt_bytes(m_orig)}")
    print(f"- Optimized: time={t_opt:.4f}s, peak_mem={fmt_bytes(m_opt)}")
    if t_opt > 0:
        print(f"Speedup: {t_orig / t_opt:.2f}x")
    if m_opt > 0:
        print(f"Peak memory reduction: {m_orig / m_opt:.2f}x")


if __name__ == '__main__':
    main()
