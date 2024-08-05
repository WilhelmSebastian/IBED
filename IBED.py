from abc import ABC, abstractmethod
from typing import Tuple, List
from Objects import Event
from datetime import datetime


class IBEDCloneError(Exception):
    def __init__(self, message, extra_info=None):
        super().__init__(message)
        self.extra_info = extra_info


class InactivityBasedEmergencyDetection(ABC):

    @abstractmethod
    def __init__(self, time: datetime, tick_interval: int = 1, buffer_history: bool = False, *args, **kwargs):
        """
        This method is an abstract method that must be implemented by the subclass.
        It is responsible for initializing a new instance of an InactivityBasedEmergencyDetection object.

        :param time: The timestamp at which the detector should be initiated.
        :param tick_interval: The time interval in seconds at which the detector should be ticked
        :param buffer_history: Whether the history of the detector should be stored for the purpose of cloning the
        detector to a history state (default = False)
        """
        self.IBED_time = time  # stores the timestamp of the current detector state
        self._tick_interval = tick_interval
        self._buffer_history = buffer_history  # defines whether the history of the IBED should be buffered

        self.alarm = True  # initialize in an alarm state as default

    @abstractmethod
    def tick(self, events: List[Event]) -> Tuple[float, float, bool]:
        """
        This method is an abstract method that must be implemented by the subclass.
        It is responsible for processing a single tick and updating the detectors state accordingly.

        :param events: A list of events to process during the tick (empty lists are possible)

        :return: A tuple containing:
                 - Score (int): The score/inactivity time of the detector after the current tick
                 - Threshold (int): The threshold for the inactivity detector after the current tick
                 - Alarm (bool): True if the detector in an alarm-state, False otherwise

        :raises NotImplementedError: The method tick must be implemented by the subclass.
        """
        raise NotImplementedError("The method tick must be implemented by the subclass.")

    @abstractmethod
    def clone(self, time: datetime, buffer_history: bool = False) -> 'InactivityBasedEmergencyDetection':
        """
        This method is an abstract method that must be implemented by the subclass.
        It allows cloning of the InactivityBasedEmergencyDetection object at a given time.

        :param time: The timestamp at which the detector should be cloned.
        :param buffer_history: Flag if the history of the detector should be constantly stored (required for cloning)

        :return: Returns an instance of the subclass of InactivityBasedEmergencyDetection.

        :raises NotImplementedError: The method clone must be implemented by the subclass.
        """
        raise NotImplementedError("The method clone must be implemented by the subclass.")