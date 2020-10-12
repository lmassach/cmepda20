import numpy
from matplotlib import pyplot as plt


class VoltageData:
    """Class for handling a set of measurements of the voltage at
    different times.
    """
    def __init__(self, times, voltages):
        """Takes as input two iterables (times and voltage readings)."""
        times = numpy.array(times, dtype=numpy.float64)
        voltages = numpy.array(voltages, dtype=numpy.float64)
        self._data = numpy.column_stack((times, voltages))

    @property
    def times(self):
        """Returns the times of the readings as 1D numpy array."""
        return self._data[:, 0]

    @property
    def timestamps(self):
        """Returns the times of the readings as 1D numpy array."""
        return self._data[:, 0]

    @property
    def voltages(self):
        """Returns the voltage readings as 1D numpy array."""
        return self._data[:, 1]

    def __len__(self):
        """Returns the number of entries/measurements."""
        return len(self._data)

    def __getitem__(self, index):
        """Applying operator [i, j] returns times[i] if j == 0,
        voltages[i] if j == 1. Slicing can be used.
        """
        return self._data[index]

    def __iter__(self):
        """At each iteration, returns a numpy array [time, voltage]."""
        return iter(self._data)

    def __str__(self):
        """Returns a string with index, time and voltage for each entry
        (one entry per line).
        """
        nd = int(numpy.ceil(numpy.log10(len(self))))
        return "\n".join(f"[{i:{nd}d}] t = {itm[0]:6.3g}, v = {itm[1]:6.3g}"
                         for i, itm in enumerate(self))

    def __repr__(self):
        return f"{type(self).__name__}({self._data})"

    def plot(self, ax=None, fmt='.', **kwargs):
        """Plots the data in ax (matplotlib.axes.Axes instance) or in a
        new figure if no ax is provided. Additional keyword arguments
        are passed on to the matplotlib.pyplot.plot method.
        """
        if ax is None:
            plt.figure()
            ax = plt.gca()
        ax.plot(self.times, self.voltages, fmt, **kwargs)


if __name__ == '__main__':
    """Here we test the functionalities of our class. These are not
    proper UnitTest - which you will se in a future lesson.
    """
    # Load some data
    t, v = numpy.loadtxt('sample_data_file.txt', unpack=True)
    # Thest the constructor
    v_data = VoltageData(t, v)
    # Test len()
    assert len(v_data) == len(t)
    # Test the timestamps attribute
    assert numpy.all(v_data.voltages == v)
    # Test the voltages attribute
    assert numpy.all(v_data.timestamps == t)
    # Test square parenthesis
    assert v_data[3, 1] == v[3]
    assert v_data[-1, 0] == t[-1]
    # Test slicing
    assert numpy.all(v_data[1:5, 1] == v[1:5])
    # Test iteration
    for i, entry in enumerate(v_data):
        assert entry[0] == t[i]
        assert entry[1] == v[i]
    # Test printing
    print(v_data)
    # Test plotting
    plt.figure('voltage vs time')
    v_data.plot(plt.gca(), fmt='r+')
    plt.show()
