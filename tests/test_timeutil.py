from datetime import time

from ats.timeutil import parse_hhmm, time_in_window, time_splits
import pandas as pd


def test_ny_window_wraps_midnight():
    start = parse_hhmm("18:30")
    end = parse_hhmm("03:00")
    assert time_in_window(time(22, 0), start, end)
    assert time_in_window(time(1, 0), start, end)
    assert not time_in_window(time(12, 0), start, end)


def test_time_splits_are_ordered():
    idx = pd.date_range("2023-01-01", periods=100, freq="D", tz="UTC")
    split = time_splits(idx, 0.6, 0.2)
    assert split.train_end < split.val_end
    assert split.val_end <= split.oos_start
