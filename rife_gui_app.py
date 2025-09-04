import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QLineEdit, QHBoxLayout, QProgressBar, QSpinBox, QMessageBox, QComboBox,
    QTextEdit, QCheckBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from rife_engine import RIFEEngine

class Worker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)

    def __init__(self, input_path, output_path, factor, model):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.factor = factor
        self.model = model

    def run(self):
        engine = RIFEEngine(model=self.model)
        
        # Test RIFE executable first
        if not engine.test_rife_executable():
            self.finished.emit(False, "RIFE executable test failed. Check if RIFE is properly installed.")
            return
        
        try:
            self.log_message.emit(f"Starting interpolation with model: {self.model}")
            self.log_message.emit(f"Input: {self.input_path}")
            self.log_message.emit(f"Output: {self.output_path}")
            self.log_message.emit(f"Factor: {self.factor}")
            
            success = engine.interpolate(self.input_path, self.output_path, self.factor, self.progress.emit)
            message = f"Interpolation complete! Check the output folder." if success else "Interpolation failed. Check the log for details."
        except Exception as e:
            success = False
            message = f"Error: {str(e)}"
            self.log_message.emit(f"Exception occurred: {str(e)}")
        
        self.finished.emit(success, message)

class RIFEGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RIFE Interpolator - Blender Viewport Enhancement")
        self.resize(600, 500)

        layout = QVBoxLayout()

        # Title
        title = QLabel("RIFE Video Frame Interpolation")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)

        # Input
        layout.addWidget(QLabel("Input Video File:"))
        input_layout = QHBoxLayout()
        self.input_edit = QLineEdit()
        btn_browse_in = QPushButton("Browse Input")
        btn_browse_in.clicked.connect(self.browse_input)
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(btn_browse_in)
        layout.addLayout(input_layout)

        # Output
        layout.addWidget(QLabel("Output Folder:"))
        output_layout = QHBoxLayout()
        self.output_edit = QLineEdit()
        # Set default output to current directory
        self.output_edit.setText(os.getcwd())
        btn_browse_out = QPushButton("Browse Output Folder")
        btn_browse_out.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(btn_browse_out)
        layout.addLayout(output_layout)

        # Settings layout
        settings_layout = QHBoxLayout()
        
        # Interpolation Factor
        factor_layout = QVBoxLayout()
        factor_layout.addWidget(QLabel("Interpolation Factor:"))
        self.factor_spin = QSpinBox()
        self.factor_spin.setValue(2)
        self.factor_spin.setMinimum(2)
        self.factor_spin.setMaximum(8)
        factor_layout.addWidget(self.factor_spin)
        settings_layout.addLayout(factor_layout)
        
        # Model Selection
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("Select RIFE Model:"))
        self.model_combo = QComboBox()
        
        # Initialize engine and find models
        try:
            engine = RIFEEngine()
            models = engine.find_models()
            print(f"Found models: {models}")
            
            if not models or models == ['rife']:
                # Check if the basic rife folder exists
                rife_path = os.path.join(engine.base_dir, 'rife')
                if os.path.exists(rife_path):
                    self.model_combo.addItem("rife")
                    self.model_combo.setEnabled(True)
                else:
                    self.model_combo.addItem("No models found - Check installation")
                    self.model_combo.setEnabled(False)
            else:
                self.model_combo.addItems(models)
                self.model_combo.setEnabled(True)
                
            # Set default to 'rife' if available
            rife_index = self.model_combo.findText('rife')
            if rife_index >= 0:
                self.model_combo.setCurrentIndex(rife_index)
                
        except Exception as e:
            print(f"Error initializing engine: {e}")
            self.model_combo.addItem("Error finding models")
            self.model_combo.setEnabled(False)

        model_layout.addWidget(self.model_combo)
        settings_layout.addLayout(model_layout)
        layout.addLayout(settings_layout)

        # Debug checkbox
        self.debug_checkbox = QCheckBox("Show detailed log")
        self.debug_checkbox.setChecked(True)
        layout.addWidget(self.debug_checkbox)

        # Run Button
        self.btn_run = QPushButton("Run Interpolation")
        self.btn_run.clicked.connect(self.run_interpolation)
        self.btn_run.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        layout.addWidget(self.btn_run)

        # Progress bar
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # Log area
        self.log_area = QTextEdit()
        self.log_area.setMaximumHeight(150)
        self.log_area.setVisible(self.debug_checkbox.isChecked())
        self.debug_checkbox.stateChanged.connect(lambda state: self.log_area.setVisible(state == 2))
        layout.addWidget(self.log_area)

        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.setLayout(layout)
        print("GUI initialized")

    def log_message(self, message):
        """Add message to log area"""
        self.log_area.append(message)
        print(message)  # Also print to console

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Input Video", 
            "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*)"
        )
        if path:
            self.input_edit.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_edit.setText(path)

    def run_interpolation(self):
        input_path = self.input_edit.text().strip()
        output_path = self.output_edit.text().strip()
        
        # Validation
        if not input_path:
            QMessageBox.warning(self, "Error", "Please select an input video file.")
            return
            
        if not output_path:
            QMessageBox.warning(self, "Error", "Please select an output folder.")
            return
            
        if not os.path.exists(input_path):
            QMessageBox.warning(self, "Error", "Input file does not exist.")
            return
            
        if not os.path.exists(output_path):
            QMessageBox.warning(self, "Error", "Output folder does not exist.")
            return
            
        factor = self.factor_spin.value()
        model = self.model_combo.currentText()
        
        if model in ["No models found - Check installation", "Error finding models"]:
            QMessageBox.warning(self, "Error", "No RIFE models were found. Please ensure RIFE is properly installed.")
            return

        # Clear log
        self.log_area.clear()
        
        # Start processing
        self.worker = Worker(input_path, output_path, factor, model)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.log_message.connect(self.log_message)
        
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Processing...")
        self.progress.setValue(0)
        self.status_label.setText("Starting interpolation...")
        
        self.worker.start()

    def on_finished(self, success, message):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Interpolation")
        self.progress.setValue(100 if success else 0)
        self.status_label.setText("Complete" if success else "Failed")
        
        # Show result
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Result")
        msg_box.setText(message)
        
        if success:
            msg_box.setIcon(QMessageBox.Information)
        else:
            msg_box.setIcon(QMessageBox.Warning)
            
        msg_box.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RIFEGUI()
    gui.show()
    sys.exit(app.exec_())