# ======================================================================================================================
# This class represents the implementation of the IBED according to Cuddihy et al. in 2007.
#
# Original Source:
# Cuddihy, P., Weisenberg, J., Graichen, C., & Ganesh, M. (2007). Algorithm to automatically detect abnormally long
# periods of inactivity in a home. In Proceedings of the 1st ACM SIGMOBILE international workshop on Systems and
# networking support for healthcare and assisted living environments. Mobisys07: The Fifth International Conference on
# Mobile Systems, Applications, and Services. ACM. https://doi.org/10.1145/1248054.1248081
# ======================================================================================================================

# Imports
from datetime import datetime, timedelta
from typing import List, Tuple
from IBED import InactivityBasedEmergencyDetection, IBEDCloneError
from Objects import Event
from collections import deque
import numpy as np


# ======================================================================================================================
# ===== Helper Methods =====
# ======================================================================================================================

def round_to_next_30_minutes(dt: datetime) -> datetime:
    """
    Round the datetime to the next 30-minute interval.

    Parameters:
    - dt (datetime): The datetime to round.

    Returns:
    - datetime: Rounded datetime.

    Explanation: This method is used to align events to the 30-minute intervals.
    """
    dt = dt.replace(second=0, microsecond=0)
    if dt.minute < 30:
        return dt.replace(minute=30)
    else:
        return (dt + timedelta(hours=1)).replace(minute=0)


def get_interval_index(dt: datetime) -> int:
    """
    Calculate the interval index for a given datetime.

    Parameters:
    - dt (datetime): The datetime to calculate the interval for.

    Returns:
    - int: Interval index.

    Explanation: Each day is divided into 48 intervals of 30 minutes each.
    """
    minutes_since_midnight = dt.hour * 60 + dt.minute
    return minutes_since_midnight // 30


def shift_index(i: int, shift: int) -> int:
    """
    Calculate the shifted index considering the 48 intervals in a day.

    Parameters:
    - i (int): Original interval index.
    - shift (int): Number of intervals to shift.

    Returns:
    - int: Shifted interval index.
    """
    return (i + shift) % 48


# ======================================================================================================================
# IBED Implementation
# ======================================================================================================================

class IBED_CuddihyEtAl_2007(InactivityBasedEmergencyDetection):

    def __init__(self,
                 time: datetime,
                 tick_interval: int = 1,
                 buffer_history: bool = False,
                 data_buffer_days: int = 60,  # 60 days - according to (Cuddihy et al., 2007)
                 maximum_percentile: float = 0.97,  # MP=0.97 - according to (Cuddihy et al., 2007)
                 uniform_buffer_percentage: float = 0.30,  # UBP=0.30 - according to (Cuddihy et al., 2007)
                 variable_buffer_percentage: float = 0.40,  # VBP=0.40 - according to (Cuddihy et al., 2007)
                 variable_buffer_weights: np.array = np.array([1, 2, 3, 2, 1]),  # W={1,2,3,2,1} - according to
                 # (Cuddihy et al., 2007)
                 last_activity_time_init: datetime = None,
                 inactivity_data_init: deque = None
                 ):
        """
        Initialize the IBED_CuddihyEtAl_2007 instance.

        Parameters:
        - time (datetime): Please refer to the BaseClass's documentation
        - tick_interval (int): Please refer to the BaseClass's documentation
        - buffer_history (bool): Please refer to the BaseClass's documentation
        - data_buffer_days (int): Number of days to keep in the local buffer.
        - maximum_percentile (float): Maximum percentile for inactivity detection.
        - uniform_buffer_percentage (float): Percentage for uniform buffer.
        - variable_buffer_percentage (float): Percentage for variable buffer.
        - variable_buffer_weights (np.array): Weights for variable buffer calculation.
        - last_activity_time_init (datetime): Initial last activity time.
        - inactivity_data_init (deque): Initial inactivity data.
        """
        super().__init__(time, tick_interval, buffer_history)

        self.maximum_percentile = maximum_percentile
        self.uniform_buffer_percentage = uniform_buffer_percentage
        self.variabel_buffer_percentage = variable_buffer_percentage
        self.variabel_buffer_weights = variable_buffer_weights

        self._last_activity_time = last_activity_time_init

        self.inactivity_data = inactivity_data_init if inactivity_data_init is not None else [
            deque(maxlen=data_buffer_days) for _ in range(48)]  # 48 intervals for 30 minutes each

        self.alarm = False

        # Initialize the history buffer if required
        if buffer_history:
            self._buffer_last_activity_time = {}
            self._buffer_inactivity_data = {}

    def tick(self, events: List[Event]) -> Tuple[float, float, bool]:
        """
        Implementation of the abstract method 'tick' defined in BaseClass.

        This method overrides the abstract method from BaseClass and provides the specific implementation details unique
        to DerivedClass. Please refer to the BaseClass's documentation for the intended use and requirements of 'tick'.
        """
        # increase current timestamp
        self.IBED_time += timedelta(seconds=self._tick_interval)
        inactivity_duration = 0
        alert_line = 0

        if len(events) == 0:
            if self._last_activity_time is not None:
                current_time = round_to_next_30_minutes(self._last_activity_time)

                # Compute alert line components
                i = shift_index(get_interval_index(current_time), -1)
                alert_line = self._calc_M(i) + self._calc_UB(i) + self._calc_VB(i)

                inactivity_duration = (self.IBED_time - self._last_activity_time).total_seconds()

        else:
            for event in events:
                if self._last_activity_time is None:
                    self._last_activity_time = event.timestamp
                    continue

                current_time = round_to_next_30_minutes(self._last_activity_time)

                while current_time < event.timestamp:
                    inactivity_duration = (current_time - self._last_activity_time).total_seconds()

                    # Compute alert line components
                    i = get_interval_index(current_time)
                    alert_line = self._calc_M(i) + self._calc_UB(i) + self._calc_VB(i)

                    # Store inactivity duration in the appropriate interval
                    self.inactivity_data[i].append(inactivity_duration)

                    current_time += timedelta(minutes=30)

                self._last_activity_time = event.timestamp

        # Store the updates to the history buffer for IBED cloning
        if self._buffer_history:
            self._buffer_last_activity_time[self.IBED_time] = self._last_activity_time
            self._buffer_inactivity_data[self.IBED_time] = self.inactivity_data.copy()

        # check against alarm
        self.alarm = inactivity_duration > alert_line

        return inactivity_duration, alert_line, self.alarm

    def _calc_M(self, i: int) -> float:
        """
        Calculate the maximum of the high percentile values considering the current and neighboring intervals.

        Parameters:
        - i (int): Interval index.

        Returns:
        - float: Maximum high percentile value.

        Formula: M(i) = max(m(i-2), m(i-1), m(i), m(i+1), m(i+2))
        """
        indices = [shift_index(i, shift) for shift in range(-2, 3)]
        return max(self._calc_m(idx) for idx in indices)

    def _calc_m(self, i: int) -> float:
        """
        Calculate the high percentile value of the historical elapsed inactive data for interval i.

        Parameters:
        - i (int): Interval index.

        Returns:
        - float: High percentile value.

        Formula: m(i) = PERCENTILE(data_i, maximum_percentile)
        """
        size = len(self.inactivity_data[i])
        if size:
            return sorted(self.inactivity_data[i])[int(size * self.maximum_percentile)]
        return 0.0

    def _calc_UB(self, i: int) -> float:
        """
        Calculate the uniform buffer for interval i.

        Parameters:
        - i (int): Interval index.

        Returns:
        - float: Uniform buffer value.

        Formula: UB = uniform_buffer_percentage * m(i)
        """
        return self.uniform_buffer_percentage * self._calc_m(i)

    def _calc_VB(self, i: int) -> float:
        """
        Calculate the variable buffer for interval i using weighted average.

        Parameters:
        - i (int): Interval index.

        Returns:
        - float: Variable buffer value.

        Formula: VB = (1 / sum_W) * sum(variable_buffer_percentage * m(shift_index(i, r)) *
        variable_buffer_weights[r + 2])
        where r ranges from -2 to 2
        """
        sum_W = np.sum(self.variabel_buffer_weights)
        weighted_sum = sum(
            self.variabel_buffer_percentage * self._calc_m(shift_index(i, r)) * self.variabel_buffer_weights[r + 2] for
            r
            in range(-2, 3))
        return (1 / sum_W) * weighted_sum

    def clone(self, time: datetime, buffer_history: bool = False) -> 'IBED_CuddihyEtAl_2007':
        """
        Implementation of the abstract method 'clone' defined in BaseClass.

        This method overrides the abstract method from BaseClass and provides the specific implementation details unique
        to DerivedClass. Please refer to the Base Class's documentation for the intended use.
        """
        if not self._buffer_history:
            raise IBEDCloneError(f"Cannot clone IBED history because buffer history attribute is not set")

        reset_time = max((key for key in self._buffer_last_activity_time.keys() if key <= time), default=None)

        return IBED_CuddihyEtAl_2007(time=reset_time,
                                     tick_interval=self._tick_interval,
                                     buffer_history=buffer_history,
                                     maximum_percentile=self.maximum_percentile,
                                     uniform_buffer_percentage=self.uniform_buffer_percentage,
                                     variable_buffer_percentage=self.variabel_buffer_percentage,
                                     variable_buffer_weights=self.variabel_buffer_weights,
                                     last_activity_time_init=self._buffer_last_activity_time[reset_time],
                                     inactivity_data_init=self._buffer_inactivity_data[reset_time]
                                     )