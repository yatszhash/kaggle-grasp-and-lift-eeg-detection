from pathlib import Path

from feature.plot.label_window import write_labels_for_window_plot

if __name__ == '__main__':
    plot_root_dir = Path(__file__).parent.parent.parent.parent.joinpath("output/features/train/window_300_stride_75/")
    data_root_dir = Path(__file__).parent.parent.parent.parent.joinpath("data/raw/train/")
    assert plot_root_dir.exists()
    assert data_root_dir.exists()
    write_labels_for_window_plot(str(plot_root_dir), str(data_root_dir))
