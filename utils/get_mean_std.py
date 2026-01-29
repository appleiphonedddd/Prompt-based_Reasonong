import numpy as np
from typing import List, Dict

class AccuracyStatistics:

    def __init__(self):
        self.accuracy_list: List[float] = []

    def add_result(self, accuracy_value: float):

        self.accuracy_list.append(accuracy_value)

    def get_mean(self) -> float:

        if not self.accuracy_list:
            return 0.0
        return float(np.mean(self.accuracy_list))

    def get_std(self) -> float:

        if len(self.accuracy_list) < 2:
            return 0.0
        return float(np.std(self.accuracy_list, ddof=1))

    def get_num_runs(self) -> int:

        return len(self.accuracy_list)

    def reset(self):

        self.accuracy_list = []

    def summary(self) -> Dict[str, float]:

        return {
            "mean": self.get_mean(),
            "std": self.get_std(),
            "num_runs": self.get_num_runs()
        }

    def print_summary(self, baseline_name: str = ""):

        print(f"\n{'=' * 50}")
        print(f"Statistics Summary")
        if baseline_name:
            print(f"Baseline: {baseline_name}")
        print(f"{'=' * 50}")
        print(f"Number of Runs: {self.get_num_runs()}")
        print(f"Mean Accuracy: {self.get_mean():.2f}%")
        print(f"Std Deviation: {self.get_std():.2f}%")
        print(f"{'=' * 50}")