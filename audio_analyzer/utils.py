import numpy as np

MAX_SIGNAL_PLOT_POINTS = 200000


def downsample_signal_for_plot(time, samples, max_points=MAX_SIGNAL_PLOT_POINTS):
    sample_count = len(samples)
    if sample_count <= max_points:
        return time, samples
    step = max(1, int(np.ceil(sample_count / max_points)))
    return time[::step], samples[::step]


def true_segments(flags, frame_duration_s, total_duration_s):
    start_idx = None
    for idx, flag in enumerate(flags):
        if flag and start_idx is None:
            start_idx = idx
        if not flag and start_idx is not None:
            yield start_idx * frame_duration_s, min(idx * frame_duration_s, total_duration_s)
            start_idx = None

    if start_idx is not None:
        yield start_idx * frame_duration_s, total_duration_s


def label_segments(labels, frame_duration_s, total_duration_s):
    if not labels:
        return

    current_label = labels[0]
    start_idx = 0
    for idx in range(1, len(labels)):
        if labels[idx] != current_label:
            yield current_label, start_idx * frame_duration_s, min(idx * frame_duration_s, total_duration_s)
            current_label = labels[idx]
            start_idx = idx

    yield current_label, start_idx * frame_duration_s, total_duration_s
