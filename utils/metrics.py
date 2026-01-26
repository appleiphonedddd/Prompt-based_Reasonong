import numpy as np
from typing import Dict, List, Optional

class Efficiency:
    
    """
    Tracks reasoning efficiency
        T = (1/N) * sum_{i=1..M} sum_{j=1..N} T_{ij}
    where:
        M = number of samples
        N = number of tasks
        T_{ij} = time for task j on sample i
    """


    def __init__(self, num_tasks:int):

        self.num_tasks = int(num_tasks)
        self.samples: List[np.ndarray] = []
    
    def record_sample(self, task_times: List[float] | np.ndarray):
        
        """
        Record timings for ONE sample across ALL tasks.

        task_times: length must be exactly N (num_tasks),
                    where task_times[j] = T_{i,j+1}
        """

        arr = np.asarray(task_times, dtype=float)
        
        if arr.ndim != 1 or arr.shape[0] != self.num_tasks:
            raise ValueError(f"task_times must be a 1D array of length {self.num_tasks}.")
        if np.any(arr < 0):
            raise ValueError("task_times must be non-negative.")
        
        self.samples.append(arr)
    
    def get_T(self) -> float:
        
        """
        Compute T exactly as Eq.(9):
            T = (1/N) * sum_{i=1..M} sum_{j=1..N} T_{ij}
        """

        total_time = float(np.sum(self.samples))
        return total_time / self.num_tasks

    def get_M(self) -> int:
        
        """Return number of recorded samples (M)."""
        return len(self.samples)

    def reset(self):

        """Reset all recorded samples."""
        self.samples = []
