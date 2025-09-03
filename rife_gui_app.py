# file: gui.py
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QLineEdit, QHBoxLayout, QProgressBar, QSpinBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from rife_engine import RIFEEngine

class Worker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool)

    def __init__(self, input_path, output_path, factor):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.factor = factor

    def run(self):
        engine = RIFEEngine()
        success = engine.interpolate_video(self.input_path, self.output_path, self.factor)
        self.finished.emit(success)

class RIFEGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RIFE Interpolator")
        self.resize(400, 250)

        layout = QVBoxLayout()

        # Input
        input_layout = QHBoxLayout()
        self.input_edit = QLineEdit()
        btn_browse_in = QPushButton("Browse Input")
        btn_browse_in.clicked.connect(self.browse_input)
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(btn_browse_in)
        layout.addLayout(input_layout)

        # Output
        output_layout = QHBoxLayout()
        self.output_edit = QLineEdit()
        btn_browse_out = QPushButton("Browse Output")
        btn_browse_out.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(btn_browse_out)
        layout.addLayout(output_layout)

        # Factor
        self.factor_spin = QSpinBox()
        self.factor_spin.setValue(2)
        self.factor_spin.setMinimum(2)
        self.factor_spin.setMaximum(8)
        layout.addWidget(QLabel("Interpolation Factor"))
        layout.addWidget(self.factor_spin)

        # Run Button
        self.btn_run = QPushButton("Run Interpolation")
        self.btn_run.clicked.connect(self.run_interpolation)
        layout.addWidget(self.btn_run)

        # Progress bar
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.setLayout(layout)

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Input Video")
        if path:
            self.input_edit.setText(path)

    def browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Output Video", filter="*.mp4")
        if path:
            self.output_edit.setText(path)

    def run_interpolation(self):
        input_path = self.input_edit.text()
        output_path = self.output_edit.text()
        
        if not input_path or not output_path:
            return  # Could add error message here
            
        factor = self.factor_spin.value()

        self.worker = Worker(input_path, output_path, factor)
        self.worker.finished.connect(self.on_finished)
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.worker.start()

    def on_finished(self, success):
        self.btn_run.setEnabled(True)
        if success:
            self.progress.setValue(100)
        else:
            self.progress.setValue(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RIFEGUI()
    gui.show()
    sys.exit(app.exec_())