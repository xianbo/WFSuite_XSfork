#!/usr/bin/env python
"""Interactive viewer and TIFF extractor for NumPy ``.npz`` archives.

Examples
--------
python tests/npz_viewer.py
python tests/npz_viewer.py "C:\\data\\Diamond14_50um\\simulated\\simulated_ref_stack.npz"

For a 3-D array, choose the slicing axis and index to view or export one
2-D plane.  "Save entire array" writes a 3-D array as a multipage TIFF.
TIFF output retains the source NumPy dtype; display contrast scaling affects
only the on-screen preview.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError as exc:
    raise SystemExit(
        "This utility requires tifffile. Install it with: "
        "python -m pip install tifffile"
    ) from exc

try:
    from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
    from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QTableView,
        QVBoxLayout,
        QWidget,
    )
    QT6 = True
except ImportError as exc:
    try:
        from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt
        from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
        from PyQt5.QtWidgets import (
            QApplication,
            QComboBox,
            QFileDialog,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSpinBox,
            QTableView,
            QVBoxLayout,
            QWidget,
        )
        QT6 = False
    except ImportError:
        raise SystemExit(
            "This utility requires PyQt6 or PyQt5. Install one with: "
            "python -m pip install PyQt6"
        ) from exc


ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter if QT6 else Qt.AlignCenter
KEEP_ASPECT_RATIO = Qt.AspectRatioMode.KeepAspectRatio if QT6 else Qt.KeepAspectRatio
SMOOTH_TRANSFORMATION = (
    Qt.TransformationMode.SmoothTransformation if QT6 else Qt.SmoothTransformation
)
GRAYSCALE8 = QImage.Format.Format_Grayscale8 if QT6 else QImage.Format_Grayscale8
DISPLAY_ROLE = Qt.ItemDataRole.DisplayRole if QT6 else Qt.DisplayRole
HORIZONTAL = Qt.Orientation.Horizontal if QT6 else Qt.Horizontal


class ScaledImageLabel(QLabel):
    """A label that keeps the current image fitted to the available area."""

    def __init__(self) -> None:
        super().__init__("Open an NPZ file to begin")
        self._source_pixmap: QPixmap | None = None
        self.setAlignment(ALIGN_CENTER)
        self.setMinimumSize(600, 500)
        self.setStyleSheet("background: #202020; color: #dddddd;")

    def set_source_pixmap(self, pixmap: QPixmap | None) -> None:
        self._source_pixmap = pixmap
        self._fit_pixmap()

    def _fit_pixmap(self) -> None:
        if self._source_pixmap is None:
            self.setPixmap(QPixmap())
            return
        self.setPixmap(
            self._source_pixmap.scaled(
                self.size(),
                KEEP_ASPECT_RATIO,
                SMOOTH_TRANSFORMATION,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        self._fit_pixmap()


class Array1DTableModel(QAbstractTableModel):
    """Lazy index/value table model that remains responsive for large arrays."""

    def __init__(self) -> None:
        super().__init__()
        self._data: np.ndarray | None = None

    def set_array(self, data: np.ndarray | None) -> None:
        self.beginResetModel()
        self._data = data
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API name
        return 0 if parent.isValid() or self._data is None else int(self._data.size)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt API name
        return 0 if parent.isValid() else 2

    def data(self, index, role=DISPLAY_ROLE):
        if role != DISPLAY_ROLE or not index.isValid() or self._data is None:
            return None
        if index.column() == 0:
            return str(index.row())
        value = self._data[index.row()]
        return str(value.item() if hasattr(value, "item") else value)

    def headerData(self, section, orientation, role=DISPLAY_ROLE):  # noqa: N802 - Qt API name
        if role != DISPLAY_ROLE:
            return None
        if orientation == HORIZONTAL:
            return "Index" if section == 0 else "Value"
        return str(section)


class LinePlotWidget(QWidget):
    """Lightweight line plot for real-valued or complex 1-D NumPy data."""

    def __init__(self) -> None:
        super().__init__()
        self._data: np.ndarray | None = None
        self._plot_data: np.ndarray | None = None
        self._minimum = 0.0
        self._maximum = 1.0
        self.setMinimumHeight(260)
        self.setStyleSheet("background: white;")

    def set_array(self, data: np.ndarray | None) -> None:
        self._data = data
        if data is None or data.size == 0:
            self._plot_data = None
            self._minimum, self._maximum = 0.0, 1.0
        else:
            values = np.abs(data) if np.iscomplexobj(data) else data
            self._plot_data = np.asarray(values, dtype=np.float64)
            finite = self._plot_data[np.isfinite(self._plot_data)]
            if finite.size:
                self._minimum = float(finite.min())
                self._maximum = float(finite.max())
                if self._maximum <= self._minimum:
                    self._maximum = self._minimum + 1.0
            else:
                self._minimum, self._maximum = 0.0, 1.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("white"))

        left, top, right, bottom = 70, 20, 20, 45
        plot_width = max(1, self.width() - left - right)
        plot_height = max(1, self.height() - top - bottom)
        x0, y0 = left, top

        painter.setPen(QPen(QColor("#555555"), 1))
        painter.drawRect(x0, y0, plot_width, plot_height)
        painter.drawText(5, y0 + 10, f"{self._maximum:.6g}")
        painter.drawText(5, y0 + plot_height, f"{self._minimum:.6g}")

        if self._plot_data is None or self._plot_data.size == 0:
            painter.drawText(x0 + 10, y0 + 25, "No 1-D data")
            return

        count = self._plot_data.size
        painter.drawText(x0, y0 + plot_height + 25, "0")
        painter.drawText(x0 + plot_width - 60, y0 + plot_height + 25, str(count - 1))
        if self._data is not None and np.iscomplexobj(self._data):
            painter.drawText(x0 + 10, y0 + 20, "Complex data: displaying magnitude")

        # Draw at most about two points per horizontal pixel. The original
        # array remains fully available in the table and TIFF export.
        step = max(1, int(np.ceil(count / max(2, plot_width * 2))))
        indices = np.arange(0, count, step, dtype=np.int64)
        if indices[-1] != count - 1:
            indices = np.append(indices, count - 1)
        values = self._plot_data[indices]
        finite = np.isfinite(values)

        path = QPainterPath()
        started = False
        denominator_x = max(1, count - 1)
        denominator_y = self._maximum - self._minimum
        for index, value, is_finite in zip(indices, values, finite):
            if not is_finite:
                started = False
                continue
            x = x0 + plot_width * float(index) / denominator_x
            y = y0 + plot_height * (self._maximum - float(value)) / denominator_y
            if started:
                path.lineTo(x, y)
            else:
                path.moveTo(x, y)
                started = True

        painter.setRenderHint(QPainter.RenderHint.Antialiasing if QT6 else QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#1565c0"), 1.5))
        painter.drawPath(path)


class NpzViewer(QMainWindow):
    def __init__(self, initial_file: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("NPZ Viewer and TIFF Extractor")
        self.resize(1050, 800)

        self._archive: np.lib.npyio.NpzFile | None = None
        self._array: np.ndarray | None = None
        self._slice: np.ndarray | None = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        open_button = QPushButton("Open NPZ...")
        open_button.clicked.connect(self.choose_file)
        file_row.addWidget(QLabel("File:"))
        file_row.addWidget(self.file_edit, 1)
        file_row.addWidget(open_button)
        layout.addLayout(file_row)

        controls = QGridLayout()
        self.key_combo = QComboBox()
        self.key_combo.currentTextChanged.connect(self.load_selected_key)
        self.axis_combo = QComboBox()
        self.axis_combo.currentIndexChanged.connect(self.axis_changed)
        self.index_spin = QSpinBox()
        self.index_spin.valueChanged.connect(self.update_view)
        self.array_info = QLabel("No data selected")

        controls.addWidget(QLabel("Data key:"), 0, 0)
        controls.addWidget(self.key_combo, 0, 1)
        controls.addWidget(QLabel("Slice axis:"), 0, 2)
        controls.addWidget(self.axis_combo, 0, 3)
        controls.addWidget(QLabel("Slice index:"), 0, 4)
        controls.addWidget(self.index_spin, 0, 5)
        controls.addWidget(self.array_info, 1, 0, 1, 6)
        layout.addLayout(controls)

        self.image_label = ScaledImageLabel()
        layout.addWidget(self.image_label, 1)

        self.scalar_label = QLabel()
        self.scalar_label.setAlignment(ALIGN_CENTER)
        self.scalar_label.setStyleSheet(
            "font-size: 32px; font-weight: bold; background: white; color: #202020;"
        )
        self.scalar_label.hide()
        layout.addWidget(self.scalar_label, 1)

        self.one_d_panel = QWidget()
        one_d_layout = QVBoxLayout(self.one_d_panel)
        self.line_plot = LinePlotWidget()
        self.table_model = Array1DTableModel()
        self.data_table = QTableView()
        self.data_table.setModel(self.table_model)
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setColumnWidth(0, 120)
        self.data_table.horizontalHeader().setStretchLastSection(True)
        one_d_layout.addWidget(self.line_plot, 1)
        one_d_layout.addWidget(self.data_table, 1)
        self.one_d_panel.hide()
        layout.addWidget(self.one_d_panel, 1)

        save_row = QHBoxLayout()
        self.save_slice_button = QPushButton("Save current slice as TIFF...")
        self.save_slice_button.clicked.connect(self.save_current_slice)
        self.save_array_button = QPushButton("Save entire array as TIFF...")
        self.save_array_button.clicked.connect(self.save_entire_array)
        self.save_slice_button.setEnabled(False)
        self.save_array_button.setEnabled(False)
        save_row.addStretch(1)
        save_row.addWidget(self.save_slice_button)
        save_row.addWidget(self.save_array_button)
        layout.addLayout(save_row)

        self.statusBar().showMessage("Ready")

        if initial_file:
            self.open_npz(initial_file)

    def choose_file(self) -> None:
        start = self.file_edit.text() or os.getcwd()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open NumPy archive", start, "NumPy archives (*.npz)"
        )
        if filename:
            self.open_npz(filename)

    def open_npz(self, filename: str) -> None:
        try:
            if self._archive is not None:
                self._archive.close()
            self._archive = np.load(filename, allow_pickle=False)
            keys = self._archive.files
            if not keys:
                raise ValueError("The archive contains no arrays")

            self._array = None
            self._slice = None
            self.file_edit.setText(str(Path(filename).resolve()))
            self.key_combo.blockSignals(True)
            self.key_combo.clear()
            self.key_combo.addItems(keys)
            self.key_combo.blockSignals(False)
            self.statusBar().showMessage(f"Archive contains {len(keys)} data keys")
            self.load_selected_key(keys[0])
        except Exception as exc:
            self._archive = None
            self._show_error("Could not open NPZ file", exc)

    def load_selected_key(self, key: str) -> None:
        if self._archive is None or not key:
            return
        try:
            self.statusBar().showMessage(f"Loading {key}...")
            QApplication.processEvents()
            # NPZ members are decompressed as complete arrays. Release the
            # previous member first to avoid holding two large stacks at once.
            self.table_model.set_array(None)
            self.line_plot.set_array(None)
            self._slice = None
            self._array = None
            gc.collect()
            self._array = np.asarray(self._archive[key])
            self.array_info.setText(
                f"Key: {key}    Shape: {self._array.shape}    "
                f"Dimensions: {self._array.ndim}    Dtype: {self._array.dtype}    "
                f"Size: {self._array.nbytes / (1024 ** 2):.1f} MiB"
            )

            self.axis_combo.blockSignals(True)
            self.axis_combo.clear()
            if self._array.ndim == 3:
                self.axis_combo.addItems(
                    [f"Axis {axis} (length {length})" for axis, length in enumerate(self._array.shape)]
                )
                self.axis_combo.setCurrentIndex(0)
            else:
                self.axis_combo.addItem("Not applicable")
            self.axis_combo.setEnabled(self._array.ndim == 3)
            self.axis_combo.blockSignals(False)

            self.save_array_button.setEnabled(self._array.ndim <= 3)
            self.axis_changed()
        except Exception as exc:
            self._array = None
            self._slice = None
            self.save_slice_button.setEnabled(False)
            self.save_array_button.setEnabled(False)
            self._show_error(f"Could not load data key '{key}'", exc)

    def axis_changed(self) -> None:
        if self._array is None:
            return
        self.index_spin.blockSignals(True)
        if self._array.ndim == 3:
            axis = max(0, self.axis_combo.currentIndex())
            self.index_spin.setRange(0, self._array.shape[axis] - 1)
            self.index_spin.setValue(min(self.index_spin.value(), self._array.shape[axis] - 1))
            self.index_spin.setEnabled(True)
        else:
            self.index_spin.setRange(0, 0)
            self.index_spin.setValue(0)
            self.index_spin.setEnabled(False)
        self.index_spin.blockSignals(False)
        self.update_view()

    def current_slice(self) -> np.ndarray:
        if self._array is None:
            raise ValueError("No data is selected")
        if self._array.ndim == 0:
            return self._array.reshape(1, 1)
        if self._array.ndim == 1:
            return self._array.reshape(1, -1)
        if self._array.ndim == 2:
            return self._array
        if self._array.ndim == 3:
            return np.take(
                self._array,
                self.index_spin.value(),
                axis=self.axis_combo.currentIndex(),
            )
        raise ValueError("Viewing is supported for arrays with at most 3 dimensions")

    def update_view(self) -> None:
        if self._array is None:
            return
        try:
            self._slice = np.asarray(self.current_slice())

            if self._array.ndim == 0:
                value = self._array.item()
                self.image_label.hide()
                self.one_d_panel.hide()
                self.scalar_label.setText(str(value))
                self.scalar_label.show()
                self.table_model.set_array(None)
                self.line_plot.set_array(None)
                self.save_slice_button.setEnabled(True)
                self.statusBar().showMessage(
                    f"Displayed scalar value {value} ({self._array.dtype})"
                )
                return

            if self._array.ndim == 1:
                self.image_label.hide()
                self.scalar_label.hide()
                self.one_d_panel.show()
                self.line_plot.set_array(self._array)
                self.table_model.set_array(self._array)
                self.save_slice_button.setEnabled(True)
                finite = self._array[np.isfinite(self._array)]
                range_text = "no finite values"
                if finite.size:
                    if np.iscomplexobj(finite):
                        magnitude = np.abs(finite)
                        range_text = (
                            f"magnitude {float(magnitude.min()):.6g} to "
                            f"{float(magnitude.max()):.6g}"
                        )
                    else:
                        range_text = (
                            f"range {float(finite.min()):.6g} to "
                            f"{float(finite.max()):.6g}"
                        )
                self.statusBar().showMessage(
                    f"Displayed 1-D array with {self._array.size} values; {range_text}"
                )
                return

            self.one_d_panel.hide()
            self.scalar_label.hide()
            self.image_label.show()
            self.table_model.set_array(None)
            self.line_plot.set_array(None)
            preview, low, high = self._make_preview(self._slice)
            height, width = preview.shape
            qimage = QImage(
                preview.data,
                width,
                height,
                preview.strides[0],
                GRAYSCALE8,
            ).copy()
            self.image_label.set_source_pixmap(QPixmap.fromImage(qimage))
            self.save_slice_button.setEnabled(True)
            self.statusBar().showMessage(
                f"Displayed shape {self._slice.shape}; preview range {low:.6g} to {high:.6g}"
            )
        except Exception as exc:
            self._slice = None
            self.table_model.set_array(None)
            self.line_plot.set_array(None)
            self.scalar_label.hide()
            self.image_label.set_source_pixmap(None)
            self.save_slice_button.setEnabled(False)
            self._show_error("Could not display the selected data", exc)

    @staticmethod
    def _make_preview(data: np.ndarray) -> tuple[np.ndarray, float, float]:
        values = np.asarray(data, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros(values.shape, dtype=np.uint8), 0.0, 0.0

        low, high = np.percentile(finite, [1.0, 99.0])
        if high <= low:
            low = float(finite.min())
            high = float(finite.max())
        if high <= low:
            return np.zeros(values.shape, dtype=np.uint8), float(low), float(high)

        scaled = (values - low) * (255.0 / (high - low))
        scaled[~np.isfinite(scaled)] = 0
        return np.clip(scaled, 0, 255).astype(np.uint8), float(low), float(high)

    def save_current_slice(self) -> None:
        if self._slice is None:
            return
        key = self.key_combo.currentText() or "data"
        suffix = ""
        if self._array is not None and self._array.ndim == 3:
            suffix = f"_axis{self.axis_combo.currentIndex()}_index{self.index_spin.value():04d}"
        self._save_tiff(self._slice, f"{key}{suffix}.tif", "Save current slice")

    def save_entire_array(self) -> None:
        if self._array is None:
            return
        data = self._array
        if data.ndim == 0:
            data = data.reshape(1, 1)
        elif data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim > 3:
            self._show_error(
                "Cannot save entire array",
                ValueError("TIFF export is supported for arrays with at most 3 dimensions"),
            )
            return
        self._save_tiff(data, f"{self.key_combo.currentText() or 'data'}.tif", "Save entire array")

    def _save_tiff(self, data: np.ndarray, suggested_name: str, title: str) -> None:
        base = Path(self.file_edit.text()).parent if self.file_edit.text() else Path.cwd()
        filename, _ = QFileDialog.getSaveFileName(
            self, title, str(base / suggested_name), "TIFF images (*.tif *.tiff)"
        )
        if not filename:
            return
        try:
            output = Path(filename)
            if output.suffix.lower() not in {".tif", ".tiff"}:
                output = output.with_suffix(".tif")
            array = np.asarray(data)
            tifffile.imwrite(
                output,
                array,
                photometric="minisblack",
                bigtiff=array.nbytes >= (4 * 1024 ** 3 - 32 * 1024 ** 2),
            )
            self.statusBar().showMessage(
                f"Saved {array.shape} {array.dtype} data to {output}"
            )
        except Exception as exc:
            self._show_error("Could not save TIFF", exc)

    def _show_error(self, title: str, error: Exception) -> None:
        QMessageBox.critical(self, title, str(error))
        self.statusBar().showMessage(str(error))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self._archive is not None:
            self._archive.close()
        super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View arrays in an NPZ archive and export arrays or 2-D slices to TIFF."
    )
    parser.add_argument("npz_file", nargs="?", help="NPZ archive to open initially")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv[:1])
    viewer = NpzViewer(args.npz_file)
    viewer.show()
    return app.exec() if QT6 else app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
