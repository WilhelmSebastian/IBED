# ======================================================================================================================
# This class represents the implementation of the IBED according to Moshtaghi et al. in 2015.
#
# Original Source:
# Moshtaghi, M., Zukerman, I. & Russell, R.A. Statistical models for unobtrusively detecting abnormal periods of
# inactivity in older adults. User Model User-Adap Inter 25, 231–265 (2015). https://doi.org/10.1007/s11257-015-9162-6
# ======================================================================================================================

# Imports
import IBED
from datetime import timedelta, datetime
from typing import Tuple, List
from Objects import Event
from enum import Enum
from collections import deque, Counter
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import linregress, iqr


# ======================================================================================================================
# ===== ENUMs =====
# ======================================================================================================================

class ThresholdEstimatorType(Enum):
    ThresholdEstimator_Pareto = "pareto",
    ThresholdEstimator_CV = "hyper_exp_cv",
    ThresholdEstimator_DE = "hyper_exp_de",


# ======================================================================================================================
# ===== Helper Classes =====
# ======================================================================================================================


class ThresholdEstimator(ABC):
    def __init__(self, window_size):
        """
        Initialize the CV threshold estimator with a window size.
        """
        self.data_window = deque(maxlen=window_size)  # Use deque with a max length

    def add_data_point(self, data_point):
        """
        Add a new data point to the window. Adds to all hours within the inactivity.
        """
        self.data_window.append(data_point)  # deque automatically handles the max length


class ThresholdEstimator_Pareto(ThresholdEstimator):
    def __init__(self, window_size, alpha=0.01):
        super().__init__(window_size=window_size)
        self.alpha = alpha  # Desired confidence level (e.g., 0.1 for 90%)

    def estimate_lambda(self, data):
        """
        Estimate the shape parameter (lambda) of the ThresholdEstimator_ParetoThresholdEstimator_Pareto distribution
        using Maximum Likelihood Estimation (MLE).
        """
        n = len(data)
        self.xm = min(data)
        return n / np.sum(np.log(data / self.xm))

    def calculate_threshold(self, lambda_):
        """
        Calculate the ThresholdEstimator_ParetoThresholdEstimator_Pareto threshold given the estimated lambda.
        """
        xm = self.xm
        alpha = self.alpha
        return xm * (1 / alpha) ** (1 / lambda_)

    def estimate_threshold(self):
        """
        Fit the ThresholdEstimator_ParetoThresholdEstimator_Pareto model to the data and estimate the threshold.
        """
        if len(self.data_window) < 2:
            return 0
        data = np.array(self.data_window)
        lambda_ = self.estimate_lambda(data)
        threshold = self.calculate_threshold(lambda_)
        return threshold


class ThresholdEstimator_CV(ThresholdEstimator):
    """
    This class implements the threshold estimator based on the coefficient of variation (CV) as proposed by Moshtaghi et al. (2015).
    """

    def __init__(self, window_size, alpha=0.1):
        """
        Initialize the CV threshold estimator with a window size and an alpha value.
        :param alpha: The significance level for the coefficient of variation (CV).
        """
        super().__init__(window_size)
        self.alpha = alpha

    def _calculate_cv(self, data):
        """
        Calculate the coefficient of variation (CV) of the given data.
        """
        mean = np.mean(data)
        std_dev = np.std(data)
        return std_dev / mean

    def _calc_bin_widht(self, data):
        """
        Calculate the bin width for the histogram based on the Freedman-Diaconis rule.
        """
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        n_I = len(self.data_window)
        bin_widht = 2 * iqr * n_I ** (-1 / 3)
        return max(bin_widht, 1)

    def estimate_threshold(self):
        """
        Estimate the threshold based on the coefficient of variation (CV).
        """
        # Check for unique values
        if len(self.data_window) < 2:
            return 0

        # Convert deque to numpy array for processing
        data = np.array(self.data_window)

        # Calculate histogram
        bins = int((np.max(data) - np.min(data)) / self._calc_bin_widht(data))
        bins = max(bins, 1)
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Accumulate bins from largest to smallest
        accumulated_bins = []
        for i in reversed(range(len(hist))):
            accumulated_bins.append(data[data >= bin_centers[i]])
            accumulated_data = np.concatenate(accumulated_bins)
            if self._calculate_cv(accumulated_data) >= 1:
                break

        # sample_mean = 1 / np.log(2) * np.median(accumulated_data)
        # outlier_region = np.percentile(data, (1 - self.alpha) * 100) # 5
        # g_ng_alpha = outlier_region / sample_mean
        # estimated_threshold = sample_mean * g_ng_alpha

        # Current estimation of g cancels out [see above]
        estimated_threshold = np.percentile(data, (1 - self.alpha) * 100)

        return estimated_threshold


class_mapping = {
    ThresholdEstimatorType.ThresholdEstimator_Pareto: ThresholdEstimator_Pareto,
    ThresholdEstimatorType.ThresholdEstimator_CV: ThresholdEstimator_CV,
}


# ======================================================================================================================
# ===== IBED =====
# ======================================================================================================================

class IBED_Moshtaghi_2015(IBED.InactivityBasedEmergencyDetection):
    def __init__(self,
                 time: datetime,
                 tick_interval: int = 1,
                 buffer_history: bool = False,
                 regions: list = None,
                 threshold_estimator_type: ThresholdEstimatorType = ThresholdEstimatorType.ThresholdEstimator_CV,
                 last_event_time_init: datetime = None,
                 last_even_region_init: str = None,
                 estimator_buffer_size: int = 50,
                 estimators_init:dict = None
                 ):
        """
        Implementation of the abstract method '__init__' defined in BaseClass.

        This method overrides the abstract method from BaseClass and provides the specific implementation details unique
        to DerivedClass. Please refer to the BaseClass's documentation for the intended use and requirements of
        '__init__'.

        :param time: Please refer to the BaseClass's documentation
        :param tick_interval: Please refer to the BaseClass's documentation
        :param buffer_history: Please refer to the BaseClass's documentation
        :param threshold_estimator_type: The _threshold_estimator_type of the IBED according to Moshtaghi et al. (2015)
        """
        super().__init__(time=time, tick_interval=tick_interval, buffer_history=buffer_history)

        # Initialize the IBED
        self._threshold_estimator_type = threshold_estimator_type
        if self._threshold_estimator_type == ThresholdEstimatorType.ThresholdEstimator_DE:
            raise NotImplementedError("Direct Estimation (DE) is not functionally implemented yet.")
        self._regions = regions

        # current IBED state (last event)
        self._last_event_time = last_event_time_init if last_event_time_init is not None else self.IBED_time
        self._last_even_region = last_even_region_init
        self._last_threshold = -1

        # Initialize the threshold estimator for each hour of the day and each room
        if estimators_init is not None:
            self._threshold_estimators = estimators_init
        else:
            self._threshold_estimators = {}
            for region in regions:
                self._threshold_estimators[region] = []
                for _ in range(24):
                    self._threshold_estimators[region].append(class_mapping[self._threshold_estimator_type](window_size=estimator_buffer_size))

        # check on init if there is an alarm state
        if self._last_even_region is not None:
            emergency_threshold = self._threshold_estimators[self._last_even_region][self.IBED_time.hour].estimate_threshold()
        else:
            emergency_threshold = None
        self.alarm = (self.IBED_time - self._last_event_time).total_seconds() > emergency_threshold if emergency_threshold is not None else False

        # Initialize the history buffer if required
        if buffer_history:
            self._buffer_last_event_time = {}
            self._buffer_last_even_region = {}
            self._buffer_last_threshold = {}
            self._buffer_threshold_estimators = {}

    def __repr__(self):
        """
        Returns the official string representation of the IBED instance.

        This method provides a concise summary of the IBED object, including its current time, score, and alarm status,
        which can be particularly useful for debugging purposes.

        :return: string representation of the IBED instance.
        :rtype: str
        """
        class_name = self.__class__.__name__
        return (f"{class_name}: \t t={self.IBED_time}, \t "
                f"last_event_time={self._last_event_time}, \t last_even_region={self._last_threshold}, \t "
                f"last_threshold={self._last_threshold}, \t")

    def _get_inactivity_time(self) -> float:
        """
        Calculate the current inactivity time.

        This method calculates the time difference between the current timestamp and the last inactivity start time.

        An inactivity period x_i > 0 in region r_k is the time elapsed since the end of a
        sensor event in r_k and the start of the next sensor event in any region.

        :return: The current inactivity time in seconds.
        :rtype: float
        """
        return int((self.IBED_time - self._last_event_time).total_seconds())

    def tick(self, events: List[Event]) -> Tuple[float, float, bool]:
        """
        Implementation of the abstract method 'tick' defined in BaseClass.

        This method overrides the abstract method from BaseClass and provides the specific implementation details unique
        to DerivedClass. Please refer to the BaseClass's documentation for the intended use and requirements of 'tick'.
        """
        # increase current timestamp
        self.IBED_time += timedelta(seconds=self._tick_interval)

        # process events
        for event in events:
            # Add the current inactivity time to the threshold estimator of every hour in the previous room
            for _hour in range(self._last_event_time.hour, self.IBED_time.hour+1):
                if self._last_even_region is not None:
                    self._threshold_estimators[self._last_even_region][_hour].add_data_point(self._get_inactivity_time())

            # Update the current room and inactivity start time
            self._last_even_region = event.room_id
            self._last_event_time = self.IBED_time

        # Estimate the threshold
        threshold = self._threshold_estimators[self._last_even_region][self.IBED_time.hour].estimate_threshold()

        # Limit the threshold to reachable values
        if self._last_threshold == -1:
            self._last_threshold = threshold
        if threshold > self._last_threshold + self._tick_interval:
            threshold = self._last_threshold + self._tick_interval
            self._last_threshold += self._tick_interval
        else:
            self._last_threshold = threshold

        # Store the updates to the history buffer for IBED cloning
        if self._buffer_history:
            self._buffer_last_event_time[self.IBED_time] = self._last_event_time
            self._buffer_last_even_region[self.IBED_time] = self._last_even_region
            self._buffer_last_threshold[self.IBED_time] = self._last_threshold
            self._buffer_threshold_estimators[self.IBED_time] = self._threshold_estimators.copy()

        # check against alarm
        if threshold < self._get_inactivity_time():
            self.alarm = True
        else:
            self.alarm = False

        return self._get_inactivity_time(), threshold, self.alarm

    def clone(self, time: datetime, buffer_history: bool = False) -> 'IBED_Moshtaghi_2015':
        """
        Implementation of the abstract method 'clone' defined in BaseClass.

        This method overrides the abstract method from BaseClass and provides the specific implementation details unique
        to DerivedClass. Please refer to the Base Class's documentation for the intended use.
        """
        if not self._buffer_history:
            raise IBED.IBEDCloneError(f"Cannot clone IBED history because buffer history attribute is not set")

        reset_time = max((key for key in self._buffer_last_event_time.keys() if key <= time), default=None)

        return IBED_Moshtaghi_2015(time=reset_time,
                                   tick_interval=self._tick_interval,
                                   buffer_history=buffer_history,
                                   regions= self._regions,
                                   threshold_estimator_type=self._threshold_estimator_type,
                                   last_event_time_init=self._buffer_last_event_time[reset_time],
                                   last_even_region_init=self._buffer_last_even_region[reset_time],
                                   estimators_init=self._buffer_threshold_estimators[reset_time]
                                   )