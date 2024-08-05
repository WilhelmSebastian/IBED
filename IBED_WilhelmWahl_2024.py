# %%
# ======================================================================================================================
# This class represents the implementation of the new IBED according to Wilhelm & Wahl (2024).
# no source available
# ======================================================================================================================

# Imports
from enum import Enum
from datetime import datetime, timedelta
from typing import List, Tuple, NoReturn, Optional
from collections import deque
from Objects import Event
import IBED
import pandas as pd
import numpy as np
from IBED import IBEDCloneError


# ======================================================================================================================
# ===== ENUMs =====
# ======================================================================================================================

class WeightingFunction(Enum):
    """
    Class representing the weighting function options.

    Attributes:
        - LINEAR: Represents the linear weighting function.
    """
    LINEAR = "linear"


class ThresholdComparisonType(Enum):
    """
    Enumeration class representing different day-comparisons that can be used for threshold calculation.

    Attributes:
        SAME_WEEKDAY: Only include days that have the same weekday as the reference day.
        EVERY_DAY: Include every day in the range for comparison.
        WEEKDAYS_AND_WEEKEND: Separate the filtering of weekdays and weekends.
        WEEKDAYS_AND_SATURDAY_AND_SUNDAY:  Separate weekdays, Saturdays, and Sundays into different comparison groups.

    """
    SAME_WEEKDAY = "same_weekday"
    EVERY_DAY = "every_day"
    WEEKDAYS_AND_WEEKEND = "weekdays_and_weekend"
    WEEKDAYS_AND_SATURDAY_AND_SUNDAY = "weekdays_and_saturdays_and_sunday"


class ThresholdCalculationMethod(Enum):
    """
    Enumeration class representing different threshold calculation methods.

    Attributes:
       MAX: Threshold will be calculated as maximum value out of the relevant time periods
    """
    MAXIMUM = "maximum"
    MAXIMUM_FILTERED = "maximum_filtered"
    INTER_QUARTILE_RANGE = "InterQuartileRange"
    STANDARD_DEVIATION = "StandardDeviation"


class Weekdays(Enum):
    """
    Enumeration class representing  the days of the week.

    Attributes:
        MONDAY (int): Integer value representing Monday (0).
        TUESDAY (int): Integer value representing Tuesday (1).
        WEDNESDAY (int): Integer value representing Wednesday (2).
        THURSDAY (int): Integer value representing Thursday (3).
        FRIDAY (int): Integer value representing Friday (4).
        SATURDAY (int): Integer value representing Saturday (5).
        SUNDAY (int): Integer value representing Sunday (6).
    """
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


# ======================================================================================================================
# ===== Helper Classes =====
# ======================================================================================================================


class FiFo_ScoreBuffer:
    def __init__(self, max_length: int):
        self.storage = deque(maxlen=max_length)  # Begrenzung der maximalen Länge
        self.lookup = {}
        self.max_timestamp = None

    def add(self, t_datetime: datetime, value: float):
        # Hinzufügen des Werts zur Doppelliste
        self.storage.append((t_datetime, value))

        # Hinzufügen des Werts zum Dictionary mit dem Zeitstempel als Schlüssel
        self.lookup[t_datetime] = value

        self.max_timestamp = t_datetime

    def get_current_score(self) -> float:
        return self.get_by_timestamp(max(self.lookup.keys(), default=0))

    def get_min_timestamp(self) -> datetime | None:
        # Abrufen des ältesten Elements aus der Doppelliste
        if self.storage:
            return self.storage[0][0]
        else:
            return None

    def get_max_timestamp(self) -> datetime:
        return max(self.lookup.keys())

    def get_by_timestamp(self, timestamp: datetime) -> float | None:
        # Abrufen des Werts anhand des Zeitstempels aus dem Dictionary
        return self.lookup.get(timestamp, None)

    def get_all_timestamps(self) -> list[datetime]:
        # Abrufen aller vorhandenen Zeitstempel aus dem Dictionary
        return list(self.lookup.keys())

    def get_max_value_in_range(self, start_timestamp: datetime, end_timestamp: datetime) -> float:
        # Ermitteln des maximalen Werts innerhalb eines bestimmten Zeitbereichs
        values_in_range = [value for timestamp, value in self.lookup.items() if
                           start_timestamp <= timestamp <= end_timestamp]
        max_value = max(values_in_range, default=None)
        return max_value if max_value else -1

    def get_max_value_in_ranges(self, time_ranges: Tuple[datetime, datetime]) -> float:
        max_values = []

        for start_time, end_time in time_ranges:
            max_value = self.get_max_value_in_range(start_time, end_time)
            max_values.append(max_value)

        return max(max_values) if max_values else -1

    def __len__(self):
        return len(self.storage)


# ======================================================================================================================
# ===== IBED =====
# ======================================================================================================================

class IBED_WilhelmWahl_2024(IBED.InactivityBasedEmergencyDetection):
    def __init__(self,
                 time: datetime,
                 tick_interval: int = 1,
                 buffer_history: bool = False,
                 score_buffer_length: Optional[int] = 30 * 24 * 60 * 60,  # Default: 30 days in seconds,
                 score_buffer_init: Optional[FiFo_ScoreBuffer] = None,
                 sensor_impact_weighting_function: WeightingFunction = WeightingFunction.LINEAR,
                 sensor_impact_max_sensor_release_time: int = 6 * 60 * 60,  # Default: 6 hours in seconds
                 sensor_last_activation_init: Optional[dict] = None,
                 threshold_comparison_type: ThresholdComparisonType = ThresholdComparisonType.WEEKDAYS_AND_WEEKEND,
                 threshold_comparison_range: Tuple[int, int] = (60 * 60, 60 * 60),  # Default: 1 hour in seconds
                 threshold_calculation_method: ThresholdCalculationMethod = ThresholdCalculationMethod.MAXIMUM_FILTERED,
                 threshold_factor: float = 1.1
                 ):
        """
        Implementation of the abstract method '__init__' defined in BaseClass.

        This method overrides the abstract method from BaseClass and provides the specific implementation details unique
        to DerivedClass. Please refer to the BaseClass's documentation for the intended use and requirements of
        '__init__'.

        :param time: Please refer to the BaseClass's documentation
        :param tick_interval: Please refer to the BaseClass's documentation
        :param buffer_history: Please refer to the BaseClass's documentation
        :param score_buffer_length: Optional length of the score buffer in seconds. This is only needed/used if no
        score_buffer_init is provided. (Default is 30 days)
        :param score_buffer_init: Optional initial score buffer. If not provided, a new Buffer will be created.
        :param sensor_impact_weighting_function: Weighting function used for sensor impact. (Default is LINEAR)
        :param sensor_impact_max_sensor_release_time: Maximum sensor release time in seconds. (Default is 6 hours)
        :param sensor_last_activation_init: Optional initial last activation of sensors. If not provided, an empty
        dictionary will be initialized.
        :param threshold_comparison_type: Type of threshold comparison. (Default is SAME_WEEKDAY)
        :param threshold_comparison_range: Range of threshold comparison in seconds (before and after). (Default is
        (1 hour, 1 hour))
        :param threshold_calculation_method: Method used for threshold calculation. (Default is MAX)
        """
        super().__init__(time=time, tick_interval=tick_interval, buffer_history=buffer_history)

        if score_buffer_length <= 0 and score_buffer_init is None:
            raise ValueError("Either score_buffer_length must be positive or score_buffer_init must be provided")
        if sensor_impact_max_sensor_release_time <= 0:
            raise ValueError(f'_sensor_impact_max_sensor_release_time {sensor_impact_max_sensor_release_time} '
                             f'must be positive')
        if threshold_comparison_range[0] <= 0 or threshold_comparison_range[1] <= 0:
            raise ValueError(f'Both elements of threshold_comparison_range {threshold_comparison_range} must be '
                             f'positive')

        self._slope = self._tick_interval  # slope of 1 per second, derived from the tick interval

        # Initialize the score buffer used for threshold calculation
        self._score_buffer = score_buffer_init if score_buffer_init is not None else FiFo_ScoreBuffer(
            int(score_buffer_length/tick_interval))
        self._score = self._score_buffer.get_current_score()

        # Parameters which are required for sensor impact weighting
        self._sensor_impact_weighting_function = sensor_impact_weighting_function
        self._sensor_impact_max_sensor_release_time = sensor_impact_max_sensor_release_time
        self._sensor_last_activation = sensor_last_activation_init if sensor_last_activation_init is not None else {}

        self._threshold_comparison_type = threshold_comparison_type
        self._threshold_comparison_range = threshold_comparison_range
        self._threshold_calculation_method = threshold_calculation_method

        self._threshold_factor = threshold_factor

        # check on init if there is an alarm state
        emergency_threshold = self._get_threshold()
        self.alarm = self._score > emergency_threshold if emergency_threshold is not None else False

        # Initialize the history buffer if required
        if buffer_history:
            self._buffer_history_score_buffer_length = score_buffer_length
            self._buffer_history_scores = {}
            self._buffer_history_sensor_last_activation = {}

    def __repr__(self):
        """
        Returns the official string representation of the IBED instance.

         This method provides a concise summary of the IBED object, including its current time, score, and alarm status,
         which can be particularly useful for debugging purposes.

        :return: string representation of the IBED instance.
        :rtype: str
        """
        class_name = self.__class__.__name__
        return f"{class_name}: \t t={self.IBED_time}, \t score={self._score}, \t alarm={self.alarm}"

    def tick(self, events: List[Event]) -> Tuple[float, float, bool]:
        """
        Implementation of the abstract method 'tick' defined in BaseClass.

        This method overrides the abstract method from BaseClass and provides the specific implementation details unique
        to DerivedClass. Please refer to the BaseClass's documentation for the intended use and requirements of 'tick'.
        """
        # increase current timestamp
        self.IBED_time += timedelta(seconds=self._tick_interval)

        # calculate the new score for the current tick
        self._score = self._score + self._slope if self._score is not None else 0
        for event in events:
            self._process_event(event)

        # get the current threshold
        emergency_threshold = self._get_threshold()

        # Store the updated score to the buffer for threshold calculation
        self._score_buffer.add(self.IBED_time, self._score)

        # Store the updated score to the history buffer for IBED cloning
        if self._buffer_history:
            self._buffer_history_scores[self.IBED_time] = self._score
            self._buffer_history_sensor_last_activation[self.IBED_time] = self._sensor_last_activation.copy()

        # check against alarm
        self.alarm = self._score > emergency_threshold if emergency_threshold is not None else False

        return self._score, emergency_threshold, self.alarm

    def _process_event(self, event: Event) -> NoReturn:
        """
        Processes an event and updates the score and sensor_last_activation.

        This method processes an event, which typically includes information about a sensor activation, and updates
        the score based on the impact of the event. It also records the timestamp of the sensor activation for future
        reference. The sensor impact is calculated using the `_get_sensor_impact` method, and the score is updated
        accordingly. The sensor's activation timestamp is recorded to track the time of the most recent activation.

        :param event: An instance of the Event class containing information about the event.
        - event.sensor_id: The ID of the sensor that generated the event.
        - event.certainty: The certainty level of the event.
        :type event: Event

        :return: None

        :raises ValueError: If the sensor impact is outside the allowed range [0, 1].
        """
        sensor_impact = self._get_sensor_impact(event.sensor_id, event.certainty)
        if not 0 <= sensor_impact <= 1:
            raise ValueError(f"Sensor impact {sensor_impact} is out of the allowed range [0, 1].")
        self._score = (1 - sensor_impact) * self._score
        self._sensor_last_activation[event.sensor_id] = self.IBED_time  # add to sensor activation history

    def _get_sensor_impact(self, sensor_id: str, sensor_certainty: float) -> float:
        """
        Calculate the weighted impact of a sensor event on the overall score.

        This function calculates the impact of a sensor event on the score, taking into account the certainty of the
        event and the time elapsed since the last activation of the sensor. It ensures that the provided parameters are
        in the expected format and range. The sensor impact is determined based on the selected weighting function.

        :param sensor_id: The ID of the sensor.
        :type sensor_id: str
        :param sensor_certainty: The certainty or confidence in the sensor event (in [0, 1])
        :type sensor_certainty: float

        :return: The weighted sensor impact.
        :rtype: float

        :raises ValueError: If sensor ID is None or certainty is not in the range [0, 1].
        """
        # check if handed parameters are in the expected format/range
        if sensor_id is None:
            raise ValueError("Sensor ID must not be None.")
        if not 0 <= sensor_certainty <= 1:
            raise ValueError(f'Sensor certainty {sensor_certainty} must be in the range [0, 1].')

        last_sensor_activation = self._sensor_last_activation.get(sensor_id)  # may None
        weighting = 1  # default sensor weighting

        # Check if the sensor has been activated before
        if last_sensor_activation:
            time_since_last_activation = self.IBED_time - last_sensor_activation
            sensor_release_time = self._sensor_impact_max_sensor_release_time * (1 - sensor_certainty)

            # Type = LINEAR
            if self._sensor_impact_weighting_function == WeightingFunction.LINEAR:
                # Check if the time since the last activation is less than the release time
                if time_since_last_activation.total_seconds() < sensor_release_time:
                    weighting = (time_since_last_activation.total_seconds()  / sensor_release_time)
            else:
                raise NotImplementedError(f'Wighting_function {self._sensor_impact_weighting_function}')

        # Calculate the final weighted sensor impact
        return weighting * sensor_certainty

    def _get_threshold(self) -> float:
        """
        Returns the calculated threshold based on the reference time periods and threshold calculation method.

        :return: The calculated threshold as a float value. Returns None if no values are available.
        :rtype: float
        """
        reference_time_periods = self._get_reference_time_periods()

        if self._threshold_calculation_method == ThresholdCalculationMethod.MAXIMUM:
            values = [self._score_buffer.get_max_value_in_range(t_from, t_to)
                      for t_from, t_to in reference_time_periods]
            return max(values) * self._threshold_factor if values else None

        elif self._threshold_calculation_method == ThresholdCalculationMethod.MAXIMUM_FILTERED:
            values = [self._score_buffer.get_max_value_in_range(t_from, t_to)
                      for t_from, t_to in reference_time_periods]

            # Convert values to a numpy array
            values = np.array(values)

            # check if values is empty
            if len(values) == 0:
                return None

            # Calculate Q1 (25th percentile) and Q3 (75th percentile)
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1

            upper_bound = Q3 + 1.5 * IQR

            return min(max(values), upper_bound) * self._threshold_factor


        elif self._threshold_calculation_method == ThresholdCalculationMethod.INTER_QUARTILE_RANGE:
            values = [self._score_buffer.get_max_value_in_range(t_from, t_to)
                      for t_from, t_to in reference_time_periods]

            # Convert values to a numpy array
            values = np.array(values)

            # check if values is empty
            if len(values) == 0:
                return None

            # Calculate Q1 (25th percentile) and Q3 (75th percentile)
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1

            return (Q3 + 1.5 * IQR) * self._threshold_factor

        elif self._threshold_calculation_method == ThresholdCalculationMethod.STANDARD_DEVIATION:
            values = [self._score_buffer.get_max_value_in_range(t_from, t_to)
                      for t_from, t_to in reference_time_periods]

            # Convert values to a numpy array
            values = np.array(values)

            # check if values is empty
            if len(values) == 0:
                return None

            mean = np.mean(values)
            std = np.std(values)

            return (mean + 2 * std) * self._threshold_factor

        else:
            raise NotImplementedError(f'Adjustment type {self._threshold_calculation_method}')

    def _get_reference_time_periods(self) -> list[tuple[datetime, datetime]]:
        """
        Returns a list of time periods to be used for comparison based on the selected comparison type.

        :return: A list of tuples representing the start and end datetime of each time period.
        :rtype: list[tuple[datetime, datetime]]
        """
        reference_date = self.IBED_time

        # Get all days since the start of the history score buffer and the day before the reference day.
        # The reference day itself must be excluded so that it is not included in the threshold calculation.
        score_min_timestamp = self._score_buffer.get_min_timestamp()
        if score_min_timestamp is not None:
            days_to_be_checked = set(pd.date_range(
                start=score_min_timestamp.replace(hour=0, minute=0, second=0, microsecond=0),
                end=(reference_date - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0),
                freq='D'
            ))
        else:
            days_to_be_checked = set()

        # Filter all relevant days based on the selected comparison Type
        filtered_days = set()
        if self._threshold_comparison_type == ThresholdComparisonType.SAME_WEEKDAY:
            # Only include days that have the same weekday as the reference day
            filtered_days = {day for day in days_to_be_checked if day.weekday() == reference_date.weekday()}

        elif self._threshold_comparison_type == ThresholdComparisonType.EVERY_DAY:
            # Include every day in the range for comparison
            filtered_days = days_to_be_checked

        elif self._threshold_comparison_type == ThresholdComparisonType.WEEKDAYS_AND_WEEKEND:
            # Separate the filtering of weekdays and weekends.
            if reference_date.weekday() <= Weekdays.FRIDAY.value:
                filtered_days = {day for day in days_to_be_checked if day.weekday() <= Weekdays.FRIDAY.value}
            else:
                filtered_days = {day for day in days_to_be_checked if day.weekday() >= Weekdays.SATURDAY.value}

        elif self._threshold_comparison_type == ThresholdComparisonType.WEEKDAYS_AND_SATURDAY_AND_SUNDAY:
            # Separate weekdays, Saturdays, and Sundays into different comparison groups.
            if reference_date.weekday() <= Weekdays.FRIDAY.value:
                filtered_days = {day for day in days_to_be_checked if day.weekday() <= Weekdays.FRIDAY.value}
            elif reference_date.weekday() == Weekdays.SATURDAY.value:
                filtered_days = {day for day in days_to_be_checked if day.weekday() == Weekdays.SATURDAY.value}
            elif reference_date.weekday() == Weekdays.SUNDAY.value:
                filtered_days = {day for day in days_to_be_checked if day.weekday() == Weekdays.SUNDAY.value}

        # Convert filtered days into a list of datetime set at the same hour, minute, and second as the reference_date.
        midpoint_dates = [day.replace(hour=reference_date.hour,
                                      minute=reference_date.minute,
                                      second=reference_date.second) for day in filtered_days]

        # Create time periods for each filtered day, based on the predefined range.
        relevant_time_periods = [(midpoint - timedelta(seconds=self._threshold_comparison_range[0]),
                                  midpoint + timedelta(seconds=self._threshold_comparison_range[1])) for midpoint in
                                 midpoint_dates]

        return relevant_time_periods

    def clone(self, time: datetime, buffer_history: bool = False) -> 'IBED_WilhelmWahl_2024':
        """
        Implementation of the abstract method 'clone' defined in BaseClass.

        This method overrides the abstract method from BaseClass and provides the specific implementation details unique
        to DerivedClass. Please refer to the Base Class's documentation for the intended use.
        """
        if not self._buffer_history:
            raise IBEDCloneError(f"Cannot clone IBED history because buffer history attribute is not set")

        # Create a new FIFO storage and copy relevant history scores up to the specified time
        new_score_buffer = FiFo_ScoreBuffer(self._buffer_history_score_buffer_length)
        # Assuming _buffer_history_score_buffer is a dict-like structure; correct as needed
        for key, value in self._buffer_history_scores.items():
            if key <= time:
                new_score_buffer.add(key, value)
            else:
                break  # Exit early since keys are out of required time scope

        return IBED_WilhelmWahl_2024(time=new_score_buffer.get_max_timestamp(),
                                     tick_interval=self._tick_interval,
                                     buffer_history=buffer_history,
                                     score_buffer_init=new_score_buffer,
                                     sensor_impact_weighting_function=self._sensor_impact_weighting_function,
                                     sensor_impact_max_sensor_release_time=self._sensor_impact_max_sensor_release_time,
                                     sensor_last_activation_init=self._buffer_history_sensor_last_activation[new_score_buffer.get_max_timestamp()].copy(),
                                     threshold_comparison_type=self._threshold_comparison_type,
                                     threshold_comparison_range=self._threshold_comparison_range,
                                     threshold_calculation_method=self._threshold_calculation_method
                                     )