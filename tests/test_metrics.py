import unittest
import numpy as np
from utils.metrics import Efficiency, Accuracy


class TestEfficiency(unittest.TestCase):
    """Test cases for the Efficiency class."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_tasks = 3
        self.efficiency = Efficiency(num_tasks=self.num_tasks)

    def test_initialization(self):
        """Test that Efficiency initializes correctly."""
        self.assertEqual(self.efficiency.num_tasks, 3)
        self.assertEqual(self.efficiency.samples, [])
        self.assertEqual(self.efficiency.get_M(), 0)

    def test_initialization_with_float_num_tasks(self):
        """Test that num_tasks is converted to int."""
        eff = Efficiency(num_tasks=3.7)
        self.assertEqual(eff.num_tasks, 3)
        self.assertIsInstance(eff.num_tasks, int)

    def test_record_sample_with_list(self):
        """Test recording a sample with a Python list."""
        task_times = [1.0, 2.0, 3.0]
        self.efficiency.record_sample(task_times)
        
        self.assertEqual(self.efficiency.get_M(), 1)
        np.testing.assert_array_equal(self.efficiency.samples[0], np.array([1.0, 2.0, 3.0]))

    def test_record_sample_with_numpy_array(self):
        """Test recording a sample with a numpy array."""
        task_times = np.array([1.5, 2.5, 3.5])
        self.efficiency.record_sample(task_times)
        
        self.assertEqual(self.efficiency.get_M(), 1)
        np.testing.assert_array_equal(self.efficiency.samples[0], task_times)

    def test_record_multiple_samples(self):
        """Test recording multiple samples."""
        self.efficiency.record_sample([1.0, 2.0, 3.0])
        self.efficiency.record_sample([4.0, 5.0, 6.0])
        self.efficiency.record_sample([7.0, 8.0, 9.0])
        
        self.assertEqual(self.efficiency.get_M(), 3)

    def test_record_sample_wrong_length_raises_error(self):
        """Test that recording a sample with wrong length raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.efficiency.record_sample([1.0, 2.0])  # Too few
        
        self.assertIn("1D array of length 3", str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.efficiency.record_sample([1.0, 2.0, 3.0, 4.0])  # Too many
        
        self.assertIn("1D array of length 3", str(context.exception))

    def test_record_sample_negative_time_raises_error(self):
        """Test that negative task times raise ValueError."""
        with self.assertRaises(ValueError) as context:
            self.efficiency.record_sample([1.0, -2.0, 3.0])
        
        self.assertIn("non-negative", str(context.exception))

    def test_record_sample_2d_array_raises_error(self):
        """Test that 2D arrays raise ValueError."""
        with self.assertRaises(ValueError):
            self.efficiency.record_sample([[1.0, 2.0, 3.0]])

    def test_get_T_single_sample(self):
        """Test T calculation with a single sample."""
        # T = (1/N) * sum of all times
        # N = 3, times = [1.0, 2.0, 3.0], sum = 6.0
        # T = 6.0 / 3 = 2.0
        self.efficiency.record_sample([1.0, 2.0, 3.0])
        
        self.assertAlmostEqual(self.efficiency.get_T(), 2.0)

    def test_get_T_multiple_samples(self):
        """Test T calculation with multiple samples.

        get_T() returns the mean time per task per run:
            T = total_time / (N * M)
        N=3 tasks, M=2 runs, total_time = 6.0+15.0 = 21.0
        T = 21.0 / (3 * 2) = 3.5
        """
        self.efficiency.record_sample([1.0, 2.0, 3.0])
        self.efficiency.record_sample([4.0, 5.0, 6.0])

        self.assertAlmostEqual(self.efficiency.get_T(), 3.5)

    def test_get_T_zero_times(self):
        """Test T calculation when all times are zero."""
        self.efficiency.record_sample([0.0, 0.0, 0.0])
        
        self.assertAlmostEqual(self.efficiency.get_T(), 0.0)

    def test_get_M(self):
        """Test get_M returns correct sample count."""
        self.assertEqual(self.efficiency.get_M(), 0)
        
        self.efficiency.record_sample([1.0, 2.0, 3.0])
        self.assertEqual(self.efficiency.get_M(), 1)
        
        self.efficiency.record_sample([4.0, 5.0, 6.0])
        self.assertEqual(self.efficiency.get_M(), 2)

    def test_reset(self):
        """Test that reset clears all samples."""
        self.efficiency.record_sample([1.0, 2.0, 3.0])
        self.efficiency.record_sample([4.0, 5.0, 6.0])
        
        self.assertEqual(self.efficiency.get_M(), 2)
        
        self.efficiency.reset()
        
        self.assertEqual(self.efficiency.get_M(), 0)
        self.assertEqual(self.efficiency.samples, [])


class TestAccuracy(unittest.TestCase):
    """Test cases for the Accuracy class."""

    def setUp(self):
        """Set up test fixtures."""
        self.accuracy = Accuracy()

    def test_initialization(self):
        """Test that Accuracy initializes correctly."""
        self.assertEqual(self.accuracy.correct_count, 0)
        self.assertEqual(self.accuracy.total_count, 0)

    def test_record_correct(self):
        """Test recording a correct answer."""
        self.accuracy.record(is_correct=True)
        
        self.assertEqual(self.accuracy.correct_count, 1)
        self.assertEqual(self.accuracy.total_count, 1)

    def test_record_incorrect(self):
        """Test recording an incorrect answer."""
        self.accuracy.record(is_correct=False)
        
        self.assertEqual(self.accuracy.correct_count, 0)
        self.assertEqual(self.accuracy.total_count, 1)

    def test_record_multiple(self):
        """Test recording multiple answers."""
        self.accuracy.record(True)
        self.accuracy.record(True)
        self.accuracy.record(False)
        self.accuracy.record(True)
        self.accuracy.record(False)
        
        self.assertEqual(self.accuracy.correct_count, 3)
        self.assertEqual(self.accuracy.total_count, 5)

    def test_record_batch(self):
        """Test batch recording."""
        self.accuracy.record_batch(correct=7, total=10)
        
        self.assertEqual(self.accuracy.correct_count, 7)
        self.assertEqual(self.accuracy.total_count, 10)

    def test_record_batch_multiple_times(self):
        """Test multiple batch recordings."""
        self.accuracy.record_batch(correct=3, total=5)
        self.accuracy.record_batch(correct=4, total=5)
        
        self.assertEqual(self.accuracy.correct_count, 7)
        self.assertEqual(self.accuracy.total_count, 10)

    def test_record_and_record_batch_combined(self):
        """Test combining single and batch recordings."""
        self.accuracy.record(True)
        self.accuracy.record(False)
        self.accuracy.record_batch(correct=8, total=10)
        
        self.assertEqual(self.accuracy.correct_count, 9)
        self.assertEqual(self.accuracy.total_count, 12)

    def test_get_accuracy_zero_total(self):
        """Test that accuracy is 0.0 when no records exist."""
        self.assertEqual(self.accuracy.get_accuracy(), 0.0)

    def test_get_accuracy_all_correct(self):
        """Test 100% accuracy."""
        self.accuracy.record_batch(correct=10, total=10)
        
        self.assertAlmostEqual(self.accuracy.get_accuracy(), 100.0)

    def test_get_accuracy_none_correct(self):
        """Test 0% accuracy."""
        self.accuracy.record_batch(correct=0, total=10)
        
        self.assertAlmostEqual(self.accuracy.get_accuracy(), 0.0)

    def test_get_accuracy_partial(self):
        """Test partial accuracy."""
        # 3 correct out of 4 = 75%
        self.accuracy.record(True)
        self.accuracy.record(True)
        self.accuracy.record(True)
        self.accuracy.record(False)
        
        self.assertAlmostEqual(self.accuracy.get_accuracy(), 75.0)

    def test_get_accuracy_returns_percentage(self):
        """Test that accuracy is returned as a percentage (0-100)."""
        self.accuracy.record_batch(correct=1, total=2)
        
        # Should be 50.0, not 0.5
        self.assertAlmostEqual(self.accuracy.get_accuracy(), 50.0)

    def test_get_correct_count(self):
        """Test get_correct_count method."""
        self.accuracy.record_batch(correct=5, total=10)
        
        self.assertEqual(self.accuracy.get_correct_count(), 5)

    def test_get_total_count(self):
        """Test get_total_count method."""
        self.accuracy.record_batch(correct=5, total=10)
        
        self.assertEqual(self.accuracy.get_total_count(), 10)

    def test_reset(self):
        """Test that reset clears all counts."""
        self.accuracy.record_batch(correct=5, total=10)
        
        self.assertEqual(self.accuracy.get_correct_count(), 5)
        self.assertEqual(self.accuracy.get_total_count(), 10)
        
        self.accuracy.reset()
        
        self.assertEqual(self.accuracy.get_correct_count(), 0)
        self.assertEqual(self.accuracy.get_total_count(), 0)
        self.assertEqual(self.accuracy.get_accuracy(), 0.0)


class TestEfficiencyEdgeCases(unittest.TestCase):
    """Edge case tests for Efficiency class."""

    def test_single_task(self):
        """Test with only one task."""
        eff = Efficiency(num_tasks=1)
        eff.record_sample([5.0])
        
        self.assertAlmostEqual(eff.get_T(), 5.0)

    def test_many_tasks(self):
        """Test with many tasks."""
        num_tasks = 100
        eff = Efficiency(num_tasks=num_tasks)
        task_times = [1.0] * num_tasks  # All 1.0
        eff.record_sample(task_times)
        
        # T = 100 / 100 = 1.0
        self.assertAlmostEqual(eff.get_T(), 1.0)

    def test_floating_point_precision(self):
        """Test floating point precision."""
        eff = Efficiency(num_tasks=3)
        eff.record_sample([0.1, 0.2, 0.3])
        
        # Sum = 0.6, T = 0.6 / 3 = 0.2
        self.assertAlmostEqual(eff.get_T(), 0.2, places=10)


class TestAccuracyEdgeCases(unittest.TestCase):
    """Edge case tests for Accuracy class."""

    def test_large_numbers(self):
        """Test with large numbers."""
        acc = Accuracy()
        acc.record_batch(correct=1000000, total=2000000)
        
        self.assertAlmostEqual(acc.get_accuracy(), 50.0)

    def test_floating_point_accuracy(self):
        """Test accuracy calculation with values that might cause floating point issues."""
        acc = Accuracy()
        acc.record_batch(correct=1, total=3)
        
        # 1/3 * 100 = 33.333...
        self.assertAlmostEqual(acc.get_accuracy(), 33.333333333333336, places=5)


if __name__ == "__main__":
    unittest.main()