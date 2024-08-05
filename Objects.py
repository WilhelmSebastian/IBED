# Objects.py

from datetime import datetime
from typing import List

import pandas as pd


class Event:
    """
    The Event class represents a sensor event with essential attributes such as its unique identifier (sensor_id),
    the timestamp at which the event occurred (timestamp), and a measure of the certainty or confidence of the event
    (certainty). This class is used to store information about individual sensor events.

    :param sensor_id: A string representing the ID of the sensor.
    :param timestamp: A datetime object representing the timestamp of the data.
    :param certainty: A float indicating the certainty level of the data (must be between 0 and 1, inclusive).

    :raises ValueError: If the certainty is outside the valid range.
    """

    def __init__(self,
                 sensor_id: str,
                 timestamp: datetime,
                 certainty: float,
                 room_id: str):
        """
        Initialize a new instance of an Event.

        :param sensor_id: A string representing the ID of the sensor.
        :param timestamp: A datetime object representing the timestamp of the data.
        :param certainty: A float indicating the certainty level of the data (must be between 0 and 1, inclusive).
        :param room_id: A string representing the ID of the room where the event occurred

        :raises ValueError: If the certainty is outside the valid range.

        Example usage:

        .. code-block:: python

            sensor = Sensor(sensor_id="123456", timestamp=datetime.now(), certainty=0.8)

        """
        if not (0 <= certainty <= 1):
            raise ValueError("Certainty must be in the range [0, 1].")

        self.sensor_id = sensor_id
        self.timestamp = timestamp
        self.certainty = certainty
        self.room_id = room_id

    def __repr__(self):
        """
        Returns a string representation of the Event object, displaying its timestamp, sensor_id, and certainty.

        :return: A string representation of the object following the format "t={timestamp}, sensor_id={sensor_id},
        certainty={certainty}"
        """
        return f"t={self.timestamp}, sensor_id={self.sensor_id}, certainty={self.certainty}, room_id={self.room_id}"


class EventSet:
    """
    The EventTable class represents a collection of sensor events organized in a tabular format. It allows for the
    storage and retrieval of sensor events. If a file path is provided, the events are loaded from a serialized
    format (e.g., a pickled DataFrame). Each event is indexed by its timestamp, making it convenient for querying
    events at specific time points.
    """

    def __init__(self, file: str = None):
        """
        Initialize an EventTable object and read in the pickle file of events.

        :param file: (str) Path to a pickle file containing sensor events (default is None).
        """
        self.events = {}
        for index, row in pd.read_pickle(file).iterrows():
            self.events[index] = Event(row.sensor_id, row.timestamp, row.certainty, row.room_id)

    def get_t_init(self) -> datetime:
        """
        Gets the min timestamp in the event table

        :return: (int) the minimum UNIX-Timestamp
        """
        return min(e.timestamp for e in self.events.values())

    def get_t_max(self) -> datetime:
        """
        Return the maximum timestamp from the events.

        :return: The maximum timestamp.
        """
        return max(e.timestamp for e in self.events.values())

    def get_events(self,
                   t_from: datetime,
                   t_to: datetime) -> List[Event]:
        return [event for event in self.events.values()
                if t_from <= event.timestamp <= t_to]

    def get_rooms(self) -> List[str]:
        return list(set(event.room_id for event in self.events.values()))

    def __repr__(self):
        """
        Provide a string representation of the EventTable object.

        :return: (str) A string representation of the EventTable object
        """
        return str(self.events)