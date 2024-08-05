# ======================================================================================================================
# This class represents the implementation of the IBED according to Floeck et al. in 2011.
#
# Original Source:
# Floeck, M., Litz, L., & Rodner, T. (2011). An Ambient Approach to Emergency Detection Based on Location Tracking. In
# Toward Useful Services for Elderly and People with Disabilities (pp. 296–302). Springer Berlin Heidelberg.
# https://doi.org/10.1007/978-3-642-21535-3_45
# ======================================================================================================================

# Imports
from datetime import datetime, timedelta
from typing import List, Tuple
from IBED import InactivityBasedEmergencyDetection, IBEDCloneError
from Objects import Event
import numpy as np
from collections import deque


# ======================================================================================================================
# ===== Helper Classes =====
# ======================================================================================================================
class FiFo_DurationOfStay_Buffer:
    def __init__(self, max_length: int):
        self._storage = deque(maxlen=max_length)
        self._current_date = None

    def add(self, time: datetime, DurationOfStay: int):
        if self._current_date == time.date():
            self._storage[-1][time.time()] = DurationOfStay
        else:
            self._current_date = time.date()
            self._storage.append({time.time(): DurationOfStay})

    def get_values_at_time(self, time: datetime):
        return [day[time.time()] for day in self._storage if time.time() in day]

    def get_values(self):
        return [value for day in self._storage for value in day.values()]


# ======================================================================================================================
# ===== IBED =====
# ======================================================================================================================

class IBED_FloeckEtAl_2011(InactivityBasedEmergencyDetection):

    def __init__(self,
                 time: datetime,
                 rooms: list,
                 tick_interval: int = 1,
                 buffer_history: bool = False,
                 duration_of_stay_buffer_size: int = 21,  # days
                 durationOfStay_min_lower=60 * 20,  # 20 min in seconds – according to (Floeck et al., 2011)
                 durationOfStay_min_upper=60 * 60,  # 60 min – according to (Floeck et al., 2011)
                 probabilityThreshold_DoS_lower=0.05,  # 5 % – according to (Floeck et al., 2011)
                 probabilityThreshold_DoS_upper=0.05,  # 5 % – according to (Floeck et al., 2011)
                 init_current_room: str = None,
                 init_current_room_entry_time: datetime = None,
                 init_duration_of_stay_buffer: dict = None,
                 ):
        """
        Implementation of the abstract method '__init__' defined in BaseClass.

        This method overrides the abstract method from BaseClass and provides the specific implementation details unique
        to DerivedClass. Please refer to the BaseClass's documentation for the intended use and requirements of
        '__init__'.

        The  hyperparameters were set analogous to the recommendation from the original paper.

        :param time: Please refer to the BaseClass's documentation
        :param tick_interval: Please refer to the BaseClass's documentation
        :param buffer_history: Please refer to the BaseClass's documentation
        :param rooms: List of all possible rooms
        :param duration_of_stay_buffer_size: Length of the DurationOfStay buffer (in days)
        :param durationOfStay_min_lower: Minimum Duration of Stay (lower value)
        :param durationOfStay_min_upper: Minimum Duration of Stay (upper value)
        :param probabilityThreshold_DoS_lower: Threshold for P(DoS) probability (lower DoS)
        :param probabilityThreshold_DoS_upper: Threshold for P(DoS) probability (upper DoS)
        :param init_current_room: Initial room (optional)
        :param init_current_room_entry_time: Initial room entry time (optional)
        :param init_duration_of_stay_buffer: Initial buffer of DurationOfStays (optional)
        """
        super().__init__(time=time, tick_interval=tick_interval, buffer_history=buffer_history)

        self.DurationOfStay_min_lower = durationOfStay_min_lower
        self.DurationOfStay_min_upper = durationOfStay_min_upper
        self.ProbabilityThreshold_DoS_lower = probabilityThreshold_DoS_lower
        self.ProbabilityThreshold_DoS_upper = probabilityThreshold_DoS_upper

        # Initialize DurationOfStay (DoS) buffer
        self._rooms = rooms
        if init_duration_of_stay_buffer is None:
            self._Duration_of_Stay_Buffer = {}
            for room in self._rooms:
                self._Duration_of_Stay_Buffer[room] = FiFo_DurationOfStay_Buffer(max_length=duration_of_stay_buffer_size)
        else:
            self._Duration_of_Stay_Buffer = init_duration_of_stay_buffer

        # IBED state parameter
        self._current_room = init_current_room
        if init_current_room_entry_time is not None:
            self._current_room_entry_time = init_current_room_entry_time
        else:
            self._current_room_entry_time = time

        # check on init if there is an alarm state
        self.alarm = self._check_alarm()

        # Initialize the history buffer if required
        if buffer_history:
            self.buffer_history_current_room = {}
            self.buffer_history_current_room_entry_time = {}
            self.buffer_history_duration_of_stay_buffer = {}

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

        # Update the internal state based on the events processed
        for event in events:
            # update internal state
            if self._current_room != event.room_id:
                self._current_room = event.room_id
                self._current_room_entry_time = event.timestamp

        # Add score to DoS buffer (TP)
        for room in self._rooms:
            if self._current_room == room:
                self._Duration_of_Stay_Buffer[room].add(self.IBED_time, self._get_DurationOfStay())
            else:
                self._Duration_of_Stay_Buffer[room].add(self.IBED_time, 0)

        # Store the updates to the history buffer for IBED cloning
        if self._buffer_history:
            self.buffer_history_current_room[self.IBED_time] = self._current_room
            self.buffer_history_current_room_entry_time[self.IBED_time] = self._current_room_entry_time
            self.buffer_history_duration_of_stay_buffer[self.IBED_time] = self._Duration_of_Stay_Buffer.copy()

        # check against alarm
        if self._current_room is not None:
            self.alarm = self._check_alarm()

        return self._get_DurationOfStay(), self._get_threshold(), self.alarm

    def _get_DurationOfStay(self) -> int:
        """
        Calculates the duration of stay in the current room by subtracting the entry time of the room from the current
        detector time and converting the result to seconds.

        :return: The duration of stay in seconds.
        :rtype: int
        """
        return int((self.IBED_time - self._current_room_entry_time).total_seconds())

    def _get_NumberOfOccurrences_accumulated(self, room_id: str, DurationOfStay_minimum: int) -> int:
        """
        Calculates the accumulated number of occurrences (NO_acc) where the duration of stay (DoS) exceeds a specified
        minimum threshold 'DurationOfStay_minimum' (DoS_min).

        The NO_acc is an indicator of how frequently the tenant has stayed in a room longer than a typical minimum
        duration. Short durations are generally non-critical as they involve recent movement, which suggests the tenant
        is active. Conversely, long durations where the tenant is stationary might indicate potential emergencies, such
        as falls, if they are uncommon within the recorded historical context.

        (According to Equation 2 of the original paper (Floeck et al., 2011))

        :param room_id: The identifier for the room whose durations of stay are to be evaluated.
        :param DurationOfStay_minimum: The minimum duration of stay in seconds that is considered for counting in the
        accumulated number. Durations less than this value are disregarded as they typically indicate normal activity and
        movement.

        :return: The number of times the duration of stay in the specified room has exceeded the specified minimum
        duration, suggesting periods of potential inactivity or concern.
        :rtype: int

        """
        return len([DurationOfStay for DurationOfStay in
                    self._Duration_of_Stay_Buffer[room_id].get_values_at_time(self.IBED_time) if DurationOfStay > DurationOfStay_minimum])

    def _get_Probability(self, room_id: str, DurationOfStay_minimum: int) -> float:
        """
        Calculates the probability P(R, t, DoS_min, TP) that represents the likelihood of encountering a duration of
        stay (DoS) of 'DurationOfStay_minimum' seconds or more in a specific room during a given time period - which is
        the history-buffer time. This probability is used to assess whether a particular DoS is typical or indicative of
        a potential emergency situation based on historical data.

        This method converts the count of occurrences where the DoS exceeds a minimum threshold (NO_acc)
        into a probability by dividing it by the total number of observations in the training period (|TP|).
        This transformation provides a normalized measure of how common a long DoS is, thereby helping to
        determine the criticality of the situation.

        :param: room_id: The identifier for the room whose durations of stay are being evaluated.
        :param: DurationOfStay_minimum: The minimum duration of stay in minutes that is considered significant or
        potentially critical.

        :return: The calculated probability that a DoS of 'DurationOfStay_minimum' minutes or more is observed in the
        specified room, normalized by the total number of observations in the training period.
        :rtype: float
        """
        if len(self._Duration_of_Stay_Buffer[room_id].get_values_at_time(self.IBED_time)) == 0: return np.NaN

        return (self._get_NumberOfOccurrences_accumulated(room_id, DurationOfStay_minimum)
                / len(self._Duration_of_Stay_Buffer[room_id].get_values_at_time(self.IBED_time)))

    def _check_alarm(self):
        """
        Evaluates whether an alarm should be raised based on the duration of stay (DoS) in the current room
        according to predefined rules and thresholds. This method utilizes the duration of stay and the computed
        probabilities to assess if the situation is typical or if it indicates a potential emergency.

        The rules implemented in this method are as follows:
        - Rule 1: If the duration of stay is between 0 and `DurationOfStay_min_lower` seconds (inclusive), the function
          returns False,
          indicating that no alarm should be raised, as the duration is considered normal and non-alarming.
        - Rule 2: If the duration of stay is more than `DurationOfStay_min_lower` seconds but less than or equal to
          `DurationOfStay_min_upper` seconds,the system checks if the probability that such a duration is critical is
          below `ProbabilityThreshold_DoS_lower`. If it is, an alarm is raised by returning True.
        - Rule 3: If the duration of stay exceeds `DurationOfStay_min_upper` seconds, the system checks if the
          probability that such a duration is critical is below `ProbabilityThreshold_DoS_upper`. If so, an alarm is
          raised by returning True.
        - Rule 4: If none of the above conditions apply, the method raises a NotImplementedError, indicating that the
          method has not been fully implemented for other cases.

        (Roles are taken directly from the original paper (Floeck et al., 2011))

        Returns:
        - bool: True if an alarm should be raised, False otherwise.
        """
        # Rule 1
        if 0 <= self._get_DurationOfStay() <= self.DurationOfStay_min_lower:
            return False

        # Rule 2
        elif self.DurationOfStay_min_lower < self._get_DurationOfStay() <= self.DurationOfStay_min_upper:
            return (self._get_Probability(self._current_room, self.DurationOfStay_min_lower)
                    < self.ProbabilityThreshold_DoS_lower)

        # Rule 3
        elif self.DurationOfStay_min_upper < self._get_DurationOfStay():
            return (self._get_Probability(self._current_room, self.DurationOfStay_min_upper)
                    < self.ProbabilityThreshold_DoS_upper)

        # Rule 4
        else:
            raise NotImplementedError("Status is not specified in detail. According to the paper: “QUIT” ")

    def _get_threshold(self) -> int:
        """
        Calculate and return the threshold value for the duration of stay.

        Returns:
        - int: Threshold value for the duration of stay.
        """
        p_lower = self._get_Probability(self._current_room, self.DurationOfStay_min_lower)
        p_upper = self._get_Probability(self._current_room, self.DurationOfStay_min_upper)

        if p_lower < self.ProbabilityThreshold_DoS_lower:
            return self.DurationOfStay_min_lower

        if p_upper < self.ProbabilityThreshold_DoS_upper:
            return self.DurationOfStay_min_upper

        return -1

    def clone(self, time: datetime, buffer_history: bool = False) -> 'IBED_FloeckEtAl_2011':
        """
        Implementation of the abstract method 'clone' defined in BaseClass.

        This method overrides the abstract method from BaseClass and provides the specific implementation details unique
        to DerivedClass. Please refer to the Base Class's documentation for the intended use.
        """
        if not self._buffer_history:
            raise IBEDCloneError(f"Cannot clone IBED history because buffer history attribute is not set")

        reset_time = max((key for key in self.buffer_history_current_room.keys() if key <= time), default=None)

        return IBED_FloeckEtAl_2011(time=reset_time,
                                    tick_interval=self._tick_interval,
                                    buffer_history=buffer_history,
                                    rooms = self._rooms,
                                    durationOfStay_min_lower = self.DurationOfStay_min_lower,
                                    durationOfStay_min_upper = self.DurationOfStay_min_upper,
                                    probabilityThreshold_DoS_lower = self.ProbabilityThreshold_DoS_lower,
                                    probabilityThreshold_DoS_upper = self.ProbabilityThreshold_DoS_upper,
                                    init_current_room = self.buffer_history_current_room[reset_time],
                                    init_current_room_entry_time = self.buffer_history_current_room_entry_time[reset_time],
                                    init_duration_of_stay_buffer = self.buffer_history_duration_of_stay_buffer[reset_time]
        )