# ======================================================================================================================
# This class represents the implementation of the IBED according to Floeck and Litz in 2009.
#
# Original Source:
# Floeck, M., & Litz, L. (2009). Inactivity patterns and alarm generation in senior citizens’ houses. In 2009 European
# Control Conference (ECC). 2009 European Control Conference (ECC). IEEE. https://doi.org/10.23919/ecc.2009.7074979
# ======================================================================================================================

# Imports
import IBED
from datetime import timedelta, datetime
import numpy as np
from typing import Tuple, List
from Objects import Event
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# ======================================================================================================================
# ===== IBED =====
# ======================================================================================================================
class IBED_FloeckLitz_2009(IBED.InactivityBasedEmergencyDetection):
    def __init__(self,
                 time: datetime,
                 tick_interval: int = 1,
                 buffer_history: bool = False,
                 tolerance_margin: float = 0.1,  # M=0,1 - according to (Floeck & Litz, 2009)
                 datapoints_for_polynom_calculation: int = 1000,  # N=1000 - according to (Floeck & Litz, 2009)
                 init_duration_of_inactivity: int = 0,
                 init_duration_of_stay_history: dict = None
                 ):
        """
        Implementation of the constructor defined in the BaseClass.

        This constructor initializes the IBED with the given parameters and calculates the initial polynomial
        coefficients for inactivity data fitting.

        :param time: The initial time for the IBED instance.
        :param tick_interval: The interval in seconds for each tick.
        :param buffer_history: Whether to buffer history for cloning.
        :param tolerance_margin: The tolerance margin M as defined in Equation (10).
        :param datapoints_for_polynom_calculation: The number of data points N for polynomial fitting as per Equation (9).
        :param init_duration_of_inactivity: Initial duration of inactivity.
        :param init_duration_of_stay_history: Initial history of inactivity durations.
        """
        super().__init__(time, tick_interval, buffer_history)

        if init_duration_of_stay_history is None:
            init_duration_of_stay_history = {}
        self._durationOfInactivity = init_duration_of_inactivity
        self._durationOfInactivity_history = init_duration_of_stay_history

        self._tolerance_margin = tolerance_margin
        self._datapoints_for_polynom_calculation = datapoints_for_polynom_calculation

        # Calculate initial polynomial coefficients
        self._coefficients = []
        self._calculate_polynomial_coefficients()

        # Check on init if there is an alarm state
        self.alarm = self._check_alarm()

    def __repr__(self):
        """
        Returns the official string representation of the IBED instance.

        This method provides a concise summary of the IBED object, including its current time, DI, and alarm status,
        which can be particularly useful for debugging purposes.

        :return: string representation of the IBED instance.
        :rtype: str
        """
        class_name = self.__class__.__name__
        return f"{class_name}: \t t={self.IBED_time}, \t score={self._durationOfInactivity}, \t alarm={self.alarm}"

    def tick(self, events: List[Event]) -> Tuple[float, float, bool]:
        """
        Implementation of the abstract method 'tick' defined in BaseClass.

        This method overrides the abstract method from BaseClass and provides the specific implementation details unique
        to DerivedClass. Please refer to the BaseClass's documentation for the intended use and requirements of 'tick'.
        """
        # Increase current timestamp
        self.IBED_time += timedelta(seconds=self._tick_interval)

        # Update the internal state based on the events processed
        if not events:
            self._durationOfInactivity += self._tick_interval
        else:
            self._durationOfInactivity = 0

        # Add DurationOfStay to history Buffer
        self._durationOfInactivity_history[self.IBED_time] = self._durationOfInactivity

        # Recalculate the coefficients for the polynomial
        self._calculate_polynomial_coefficients()

        # Check against alarm
        self.alarm = self._check_alarm()

        return self._durationOfInactivity, self._get_threshold(self.IBED_time), self.alarm

    def _check_alarm(self) -> bool:
        """
        Checks if the current duration of inactivity exceeds the threshold and should trigger an alarm.

        :return: True if an alarm should be triggered, otherwise False.
        :rtype: bool
        """
        _current_threshold = self._get_threshold(self.IBED_time)
        return self._durationOfInactivity > _current_threshold

    def _get_threshold(self, current_time: datetime) -> float:
        """
        Calculates the threshold Th2(t) as per Equation (10) by evaluating the polynomial fit function and adding the
        necessary shifts and tolerance margin.

        :param current_time: The current time for which the threshold is calculated.
        :param N: The number of data points to consider.
        :return: The calculated threshold.
        :rtype: float
        """
        # Normalize time to a fraction of the day
        normalized_time = (current_time.hour * 3600 + current_time.minute * 60 + current_time.second) / 86400
        f = sum(coef * (normalized_time ** exp) for exp, coef in enumerate(self._coefficients))

        # Shift by the minimal value A (assuming A is pre-calculated)
        A = self._calculate_shift_A()
        # Tolerance margin M
        M = self._tolerance_margin  # M=0.1 as specified in the paper
        threshold = (1 + M) * A + f

        return threshold

    def _calculate_shift_A(self) -> float:
        """
        Calculates the minimal shift A such that no inactivity spikes cause an alarm, as described before Equation (10).

        :return: The calculated shift A.
        :rtype: float
        """
        N = self._datapoints_for_polynom_calculation

        if len(self._durationOfInactivity_history) < N:
            N = len(self._durationOfInactivity_history)
        if N == 0:
            return 0.0  # If no data is available, the shift is 0
        history_values = list(self._durationOfInactivity_history.values())[-N:]
        history_keys = list(self._durationOfInactivity_history.keys())[-N:]
        max_inactivity = max(history_values, default=0)
        if not history_keys:
            return 0.0
        max_poly_value = max(
            sum(coef * (self._normalize_time(t) ** exp) for exp, coef in enumerate(self._coefficients))
            for t in history_keys
        )
        A = max_inactivity - max_poly_value
        return max(A, 0)

    def _normalize_time(self, time: datetime) -> float:
        """
        Normalizes the time to a fraction of the day.

        :param time: The time to be normalized.
        :return: The normalized time as a fraction of the day.
        :rtype: float
        """
        total_seconds = time.hour * 3600 + time.minute * 60 + time.second
        return total_seconds / 86400

    def _calculate_polynomial_coefficients(self) -> List[float]:
        """
        Calculates the polynomial coefficients B_j for fitting the inactivity data as per Equation (7).

        :return: A list of polynomial coefficients.
        :rtype: list
        """
        if len(self._durationOfInactivity_history) < self._datapoints_for_polynom_calculation:
            return [.0, .0, .0, .0, 1]  # Default coefficients

        # Use only the last N points from history (N=self._datapoints_for_polynom_calculation)
        history_keys = list(self._durationOfInactivity_history.keys())[-self._datapoints_for_polynom_calculation:]
        times = np.array([((t.hour * 3600 + t.minute * 60 + t.second) / 86400) for t in history_keys])
        inactivity_durations = np.array([self._durationOfInactivity_history[t] for t in history_keys])

        # Fit a 4th-degree polynomial (Equation 7)
        poly = PolynomialFeatures(degree=4)
        X_poly = poly.fit_transform(times.reshape(-1, 1))
        model = LinearRegression().fit(X_poly, inactivity_durations)

        self._coefficients = model.coef_.tolist()

    def clone(self, time: datetime, buffer_history: bool = False) -> 'IBED_FloeckLitz_2009':
        """
        Implementation of the abstract method 'clone' defined in BaseClass.

        This method overrides the abstract method from BaseClass and provides the specific implementation details unique
        to DerivedClass. Please refer to the Base Class's documentation for the intended use.
        """
        if not self._buffer_history:
            raise IBED.IBEDCloneError(f"Cannot clone IBED history because buffer history attribute is not set")

        reset_time = max((key for key in self._durationOfInactivity_history.keys() if key <= time), default=None)

        # Create a new history dictionary up to the reset_time
        init_duration_of_stay_history = {k: v for k, v in self._durationOfInactivity_history.items() if k <= reset_time}

        return IBED_FloeckLitz_2009(time=reset_time,
                                    tick_interval=self._tick_interval,
                                    buffer_history=buffer_history,
                                    tolerance_margin=self._tolerance_margin,
                                    datapoints_for_polynom_calculation=self._datapoints_for_polynom_calculation,
                                    init_duration_of_inactivity=self._durationOfInactivity_history[reset_time],
                                    init_duration_of_stay_history=init_duration_of_stay_history)