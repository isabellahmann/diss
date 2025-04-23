from typing import Optional, List, Tuple

import torch
import pynvml


def get_stats() -> list[int]:
    """
    Returns a list of CUDA device IDs sorted by their current GPU utilization in ascending order.

    Requires `pynvml` (install via `pip install nvidia-ml-py`).
    """
    if not torch.cuda.is_available():
        return 'mps' if torch.backends.mps.is_available() else 'cpu'

    pynvml.nvmlInit()
    devices = []

    for i in range(torch.cuda.device_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        devices.append((i, utilization))

    pynvml.nvmlShutdown()

    return [(id, util) for id, util in sorted(devices, key=lambda x: x[1])]


def get_best_device(stats: Optional[List[Tuple[int, int]]] = None):
    if stats is None:
        stats = get_stats()
        if stats in ['mps', 'cpu']: 
            return stats

    return f"cuda:{stats[0][0]}"


if __name__ == "__main__":
    statistics = get_stats()
    print("ID\tUtilisation")
    for device, utilisation in statistics:
        print(f"cuda:{device}\t\t{utilisation}%")

    print(f"\nSelected: {get_best_device(statistics)}")