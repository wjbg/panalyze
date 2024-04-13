"""
Contains core of panalyze: Measurement, Segment

"""

from __future__ import annotations
from typing import Callable, Union, Optional
import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from io import StringIO


vector = matrix = npt.NDArray[np.float_]  # Annotation shorthand

# Sapphire reference heat capacity data
SAPPHIRE_CP_DATA = np.loadtxt("sapphire_cp.csv", dtype=np.float32,
                              delimiter=",", skiprows=1, usecols=(0, 2))
SAPPHIRE_CP = CubicSpline(SAPPHIRE_CP_DATA[:, 0],
                          SAPPHIRE_CP_DATA[:, 1])

class Measurement():
    """Class to represent a DSC measurement.

    Attributes
    ----------
    sample : str
        Sample name.
    mass : float
        Sample mass in mg.
    segments : list[Segment]
        List with measurement segments.
    operator : str
        Name of the operator.
    date : datetime.date
        Run date.
    instrument : str
        Instrument name.
    fn : str
        Filename.

    Methods
    -------
    read_csv(fn)
        Reads TRIOS csv file and updates attributes.
    plot()
        Plots T vs. ΔH for all segments.

    """

    def __init__(self, fn: str):
        """Initialize Measurement instance.

        Parameters
        ----------
        fn : str
            Filename.

        Note: For now only CSV files exported from TRIOS are supported.

        """
        self.read_csv(fn)

    def read_csv(self, fn: str):
        """Read file.

        Parameters
        ----------
        fn : str
            Filename.

        Note
        ----
        Currently only supports CSV files exported from TRIOS. The datafile should
        include the parameters and the steps. The following columns (in order) need to
        be exported: time [s], temperature [°C], heat flow [mW], norm. heat flow [W/g].

        """
        with open(fn, 'r', encoding='utf-8-sig') as f:
            head, *segments = f.read().split("[step]\n")
            for info in head.strip().split("\n"):
                keyw, *val = info.lower().split(",")
                if keyw == "filename":
                    self.fn = val[0] + ".csv"
                elif keyw == "operator":
                    self.operator = val[0]
                elif keyw == "rundate":
                    self.date = datetime.strptime(val[0], "%m/%d/%Y")
                elif keyw == "sample name":
                    self.sample = val[0]
                elif keyw == "sample mass":
                    self.mass = float(val[0].split(" ")[0])  # mg
                elif keyw == "instrument name":
                    self.instrument = val[0]
            self.segments = []
            for segment in segments:
                name = segment.split("\n")[0]
                data = pd.read_csv(StringIO(segment), sep=",", header=2)
                data.columns = ["Time [s]",
                                "Temperature [°C]",
                                "Heat flow [mW]",
                                "Norm. heat flow [W/g]"]
                self.segments.append(Segment(data, name, self.mass))

    def plot(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot T versus ΔH for all segments.

        Parameters
        ----------
        ax : plt.Axes (optional)
            An axes where to plot.

        Returns
        -------
        ax : plt.Axes
            Axes handle.

        Note
        ----
        Uses the pandas plot function, which also means that all the pandas plot options
        are available. More information can be found here:

        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html

        """
        if ax == None:
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.18, 0.7, 0.72])
        for segment in self.segments:
            segment.plot(x="Temperature [°C]", y="Heat flow [mW]", ax=ax, legend=None)
        ax.set_title(self.sample)
        return ax

    def __str__(self) -> str:
        """Return string for printing."""
        meta = (f"\n{self.sample}\n" +
                "------------------------------------\n" +
                f"Mass:               {self.mass} mg\n" +
                f"Date:               {self.date.date()}\n" +
                f"Operator:           {self.operator}\n" +
                f"Instrument:         {self.instrument}\n" +
                f"Filename:           {self.fn}\n" +
                f"Number of segments: {len(self.segments):<2}\n\n")
        segm = ("Segments\n"
                "-----------------------------------\n")
        for i, segment in enumerate(self.segments):
            segm += f"{i:<2} {segment.name}\n"
        return (meta + segm)

    def __getitem__(self, index) -> Segment:
        """Return segment using index."""
        return self.segments[index]


class Segment():
    """Class to represent a DSC measurement segment.

    Attributes
    ----------
    data : pd.DataFrame
        Pandas DataFrame with DSC data.
    name : str
        Description of segment
    N : int
        Number of data points.
    t_interval : (float, float)
        Start and end time.
    m : float
        Specimen mass in mg.

    Methods
    -------
    plot()
        Plot Segment data.
    calculate_cp(baseline, sapphire, cp_sapphire)
        Calculate heat capacity based on Segment data.

    """

    def __init__(self, data: pd.DataFrame, name: str="", mass: float=0.0):
        """Initialize Segment instance.

        Parameters
        ----------
        data : Pandas DataFrame with:
            - Time [s]
            - Temperature [°C]
            - Heat flow [W]
            - Norm. heat flow [W/g]
        name : str (defaults to "")
            Description of the segment.
        mass : float (defaults to 0.0)
            Specimen mass in mg.

        """
        self.name = name
        self.data = data
        self.mass = mass
        self.N = len(data)
        self.t_interval = (data.iloc[0, 0], data.iloc[-1, 0])

    def plot(self, *args, **kwargs) -> plt.Axes:
        """Plot data.

        Parameters
        ----------
        x : str | int (defaults to None)
            Label or column position of data to plot on x-axis.
        y : str | int (defaults to None)
            Label or column position of data to plot on y-axis.
        ax : plt.Axes (optional)
            An axes where to plot.

        Returns
        -------
        ax : plt.Axes
            Axes handle.

        Note
        ----
        Uses the pandas plot function, which also means that all the pandas plot options
        are available. More can be found here:

        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html

        """
        ax = self.data.plot(*args, **kwargs)
        return ax

    def calculate_cp(self,
                     baseline: Segment, sapphire: Segment,
                     cp_sapphire: Callable=SAPPHIRE_CP) -> matrix:
        """Calculate heat capacity.

        Parameters
        ----------
        baseline : Segment
            Baseline DSC data.
        sapphire : Segment
            Sapphire DSC data.
        cp_sapphire : Callable
            Function that returns cp of sapphire at T.

        Returns
        -------
        cp : pd.DataFrame
            Dataframe with temperature and heat capacity.

        Note: the cp data is also added to self.data.

        """
        T = self.data.iloc[:, 1]
        Dst = sapphire._H(T) - baseline._H(T)
        Ds = self.data.iloc[:, 2] - baseline._H(T)
        cp = cp_sapphire(T) * (Ds/Dst) * (sapphire.mass/self.mass)
        if new_column_name in df.columns:
            self.data["Specific heat [J/g°C]"] = cp
        else:
            self.data.insert(4, "Specific heat [J/g°C]", cp)
        return pd.concat((T, cp.rename("Specific heat [J/g°C]")), axis=1)

    def _H(self, T: float) -> float:
        """Return heat flow at T based on cubic spline interpolation.

        Parameter
        ---------
        T : float | np.ndarray(dim=1 dtype=float)
            Temperature in °C.

        Return
        ------
        H : np.ndarray(dim=1, dtype=float)
            Heat flow at T in mW.

        """
        if all(np.diff(self.data.iloc[:, 1])) > 0:
            cs = CubicSpline(self.data.iloc[:, 1], self.data.iloc[:, 2])
        elif all(np.diff(self.data.iloc[:, 1])) < 0:
            cs = CubicSpline(self.data.iloc[::-1, 1], self.data.iloc[::-1, 2])
        else:
            raise Exception("Temperature data must be strictly " +
                            "increasing or decreasing for interpolation.")
        return cs(T)

    def __str__(self) -> str:
        """Return string for printing."""
        meta = (f"\n{self.name}\n" +
                "-------------------------------\n" +
                f"Number of data points:  {self.N:>5}\n" +
                f"Start time:             {int(self.t_interval[0]):>5} s\n"
                f"End time:               {int(self.t_interval[1]):>5} s\n\n")
        return (meta + self.data.__str__())

    def __getitem__(self, i):
        """Return segment using index."""
        return self.data[i]