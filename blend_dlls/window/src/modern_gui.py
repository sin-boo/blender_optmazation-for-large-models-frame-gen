#!/usr/bin/env python3
"""
Modern Screen Capture GUI Application
Features a modern dark theme and improved user experience
"""

import sys
import os
import json
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import font as tkFont

# Import capture functionality
try:
    import win32gui
    import win32con
    import win32ui
    import win32api
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

try:
    from PIL import Image, ImageGrab, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

import requests
import io

class ModernStyle:
    """Modern dark theme styling for tkinter"""
    
    # Color palette
    COLORS = {
        'bg_primary': '#1e1e1e',        # Dark background
        'bg_secondary': '#2d2d2d',      # Lighter dark background
        'bg_tertiary': '#3c3c3c',       # Even lighter background
        'fg_primary': '#ffffff',        # White text
        'fg_secondary': '#cccccc',      # Light gray text
        'fg_disabled': '#666666',       # Disabled text
        'accent_primary': '#0078d4',    # Blue accent
        'accent_hover': '#106ebe',      # Darker blue
        'success': '#107c10',           # Green
        'warning': '#ff8c00',           # Orange
        'error': '#d13438',             # Red
        'border': '#484848'             # Border color
    }
    
    @classmethod
    def configure_style(cls, root):
        """Configure the modern style"""
        # Configure ttk styles
        style = ttk.Style(root)
        style.theme_use('clam')
        
        # Configure root window
        root.configure(bg=cls.COLORS['bg_primary'])
        
        # Frame styles
        style.configure(
            'Modern.TFrame',
            background=cls.COLORS['bg_primary'],
            borderwidth=0
        )
        
        style.configure(
            'Card.TFrame',
            background=cls.COLORS['bg_secondary'],
            relief='flat',
            borderwidth=1
        )
        
        # Label styles
        style.configure(
            'Modern.TLabel',
            background=cls.COLORS['bg_primary'],
            foreground=cls.COLORS['fg_primary'],
            font=('Segoe UI', 10)
        )
        
        style.configure(
            'Title.TLabel',
            background=cls.COLORS['bg_primary'],
            foreground=cls.COLORS['fg_primary'],
            font=('Segoe UI', 14, 'bold')
        )
        
        style.configure(
            'Subtitle.TLabel',
            background=cls.COLORS['bg_primary'],
            foreground=cls.COLORS['fg_secondary'],
            font=('Segoe UI', 9)
        )
        
        # Button styles
        style.configure(
            'Modern.TButton',
            font=('Segoe UI', 10),
            borderwidth=0,
            focuscolor='none'
        )
        
        style.map(
            'Modern.TButton',
            background=[
                ('active', cls.COLORS['accent_hover']),
                ('pressed', cls.COLORS['accent_hover']),
                ('!active', cls.COLORS['accent_primary'])
            ],
            foreground=[('!active', cls.COLORS['fg_primary'])]
        )
        
        style.configure(
            'Success.TButton',
            font=('Segoe UI', 10, 'bold')
        )
        
        style.map(
            'Success.TButton',
            background=[
                ('active', '#0e6b0e'),
                ('!active', cls.COLORS['success'])
            ],
            foreground=[('!active', cls.COLORS['fg_primary'])]
        )
        
        style.configure(
            'Warning.TButton',
            font=('Segoe UI', 10, 'bold')
        )
        
        style.map(
            'Warning.TButton',
            background=[
                ('active', '#e67c00'),
                ('!active', cls.COLORS['warning'])
            ],
            foreground=[('!active', cls.COLORS['fg_primary'])]
        )
        
        # Entry styles
        style.configure(
            'Modern.TEntry',
            fieldbackground=cls.COLORS['bg_tertiary'],
            borderwidth=1,
            insertcolor=cls.COLORS['fg_primary'],
            font=('Segoe UI', 10)
        )
        
        style.map(
            'Modern.TEntry',
            focuscolor=[('focus', cls.COLORS['accent_primary'])],
            bordercolor=[('focus', cls.COLORS['accent_primary'])]
        )
        
        # Checkbutton styles
        style.configure(
            'Modern.TCheckbutton',
            background=cls.COLORS['bg_primary'],
            foreground=cls.COLORS['fg_primary'],
            focuscolor='none',
            font=('Segoe UI', 10)
        )
        
        # Radiobutton styles
        style.configure(
            'Modern.TRadiobutton',
            background=cls.COLORS['bg_primary'],
            foreground=cls.COLORS['fg_primary'],
            focuscolor='none',
            font=('Segoe UI', 10)
        )
        
        # LabelFrame styles
        style.configure(
            'Modern.TLabelframe',
            background=cls.COLORS['bg_primary'],
            borderwidth=1,
            relief='solid',
            bordercolor=cls.COLORS['border']
        )
        
        style.configure(
            'Modern.TLabelframe.Label',
            background=cls.COLORS['bg_primary'],
            foreground=cls.COLORS['fg_primary'],
            font=('Segoe UI', 11, 'bold')
        )

class WindowInfo:
    """Container for window information"""
    def __init__(self, hwnd: int, title: str, rect: Tuple[int, int, int, int], visible: bool = True):
        self.hwnd = hwnd
        self.title = title
        self.rect = rect  # (left, top, right, bottom)
        self.visible = visible
        self.width = rect[2] - rect[0]
        self.height = rect[3] - rect[1]

class ModernScreenCaptureApp:
    def __init__(self):
        self.app_dir = Path(__file__).parent.parent
        self.config_dir = self.app_dir / "config"
        self.config_file = self.config_dir / "app_config.json"
        self.config = self.load_config()
        
        # Capture settings
        self.capture_active = False
        self.capture_thread = None
        self.selected_window = None
        self.capture_mode = "window"
        self.windows = []
        
        # Initialize GUI
        self.root = tk.Tk()
        self.setup_modern_gui()
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        self.config_dir.mkdir(exist_ok=True)
        
        default_config = {
            "server_url": "http://127.0.0.1:8000/stream",
            "capture_interval": 0.1,
            "output_directory": "./captured_frames",
            "save_locally": True,
            "stream_to_server": True,
            "window_title_filter": "",
            "capture_mouse": True,
            "window_geometry": "900x700",
            "theme": "dark"
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"Error loading config: {e}")
        
        return default_config
    
    def save_config(self):
        """Save current configuration"""
        try:
            # Update window geometry
            self.config["window_geometry"] = self.root.geometry()
            
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def setup_modern_gui(self):
        """Setup the modern GUI"""
        self.root.title("🖥️ Screen Capture Pro")
        self.root.geometry(self.config["window_geometry"])
        self.root.minsize(800, 600)
        
        # Configure modern style
        ModernStyle.configure_style(self.root)
        
        # Create main container
        self.main_container = ttk.Frame(self.root, style='Modern.TFrame')
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.create_header()
        self.create_capture_section()
        self.create_settings_section()
        self.create_status_section()
        self.create_control_section()
        
        # Load windows on startup
        self.refresh_windows()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_header(self):
        """Create application header"""
        header_frame = ttk.Frame(self.main_container, style='Modern.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # App title
        title_label = ttk.Label(
            header_frame, 
            text="🖥️ Screen Capture Pro", 
            style='Title.TLabel'
        )
        title_label.pack(side=tk.LEFT)
        
        # Version info
        version_label = ttk.Label(
            header_frame,
            text="v2.0 | Self-Sufficient Edition",
            style='Subtitle.TLabel'
        )
        version_label.pack(side=tk.RIGHT)
    
    def create_capture_section(self):
        """Create capture source selection section"""
        capture_frame = ttk.LabelFrame(
            self.main_container,
            text="📹 Capture Source",
            style='Modern.TLabelframe',
            padding=15
        )
        capture_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Mode selection
        mode_frame = ttk.Frame(capture_frame, style='Modern.TFrame')
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="window")
        
        ttk.Radiobutton(
            mode_frame,
            text="🪟 Capture Window",
            variable=self.mode_var,
            value="window",
            style='Modern.TRadiobutton',
            command=self.on_mode_change
        ).pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Radiobutton(
            mode_frame,
            text="🖥️ Capture Screen",
            variable=self.mode_var,
            value="screen",
            style='Modern.TRadiobutton',
            command=self.on_mode_change
        ).pack(side=tk.LEFT)
        
        # Window selection frame
        self.window_frame = ttk.Frame(capture_frame, style='Card.TFrame')
        self.window_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Window list header
        list_header = ttk.Frame(self.window_frame, style='Modern.TFrame')
        list_header.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        ttk.Label(
            list_header,
            text="Available Windows:",
            style='Modern.TLabel'
        ).pack(side=tk.LEFT)
        
        refresh_btn = ttk.Button(
            list_header,
            text="🔄 Refresh",
            command=self.refresh_windows,
            style='Modern.TButton'
        )
        refresh_btn.pack(side=tk.RIGHT)
        
        # Window listbox with scrollbar
        list_frame = ttk.Frame(self.window_frame, style='Modern.TFrame')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Custom listbox styling
        self.window_listbox = tk.Listbox(
            list_frame,
            font=('Segoe UI', 10),
            bg=ModernStyle.COLORS['bg_tertiary'],
            fg=ModernStyle.COLORS['fg_primary'],
            selectbackground=ModernStyle.COLORS['accent_primary'],
            selectforeground=ModernStyle.COLORS['fg_primary'],
            borderwidth=0,
            highlightthickness=1,
            highlightcolor=ModernStyle.COLORS['accent_primary'],
            activestyle='none'
        )
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.window_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.window_listbox.yview)
        
        self.window_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_settings_section(self):
        """Create settings section"""
        settings_frame = ttk.LabelFrame(
            self.main_container,
            text="⚙️ Settings",
            style='Modern.TLabelframe',
            padding=15
        )
        settings_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Create two columns
        left_col = ttk.Frame(settings_frame, style='Modern.TFrame')
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        right_col = ttk.Frame(settings_frame, style='Modern.TFrame')
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Left column settings
        ttk.Label(left_col, text="🌐 Server URL:", style='Modern.TLabel').pack(anchor=tk.W)
        self.server_url_var = tk.StringVar(value=self.config["server_url"])
        server_entry = ttk.Entry(
            left_col,
            textvariable=self.server_url_var,
            font=('Segoe UI', 10),
            style='Modern.TEntry'
        )
        server_entry.pack(fill=tk.X, pady=(2, 10))
        
        ttk.Label(left_col, text="⏱️ Capture Interval (seconds):", style='Modern.TLabel').pack(anchor=tk.W)
        self.interval_var = tk.StringVar(value=str(self.config["capture_interval"]))
        interval_entry = ttk.Entry(
            left_col,
            textvariable=self.interval_var,
            font=('Segoe UI', 10),
            style='Modern.TEntry',
            width=10
        )
        interval_entry.pack(anchor=tk.W, pady=(2, 10))
        
        # Right column settings
        ttk.Label(right_col, text="📁 Output Directory:", style='Modern.TLabel').pack(anchor=tk.W)
        
        dir_frame = ttk.Frame(right_col, style='Modern.TFrame')
        dir_frame.pack(fill=tk.X, pady=(2, 10))
        
        self.output_dir_var = tk.StringVar(value=self.config["output_directory"])
        dir_entry = ttk.Entry(
            dir_frame,
            textvariable=self.output_dir_var,
            font=('Segoe UI', 10),
            style='Modern.TEntry'
        )
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        browse_btn = ttk.Button(
            dir_frame,
            text="📂",
            command=self.browse_output_dir,
            style='Modern.TButton',
            width=3
        )
        browse_btn.pack(side=tk.RIGHT)
        
        # Checkboxes
        self.save_local_var = tk.BooleanVar(value=self.config["save_locally"])
        save_check = ttk.Checkbutton(
            right_col,
            text="💾 Save Locally",
            variable=self.save_local_var,
            style='Modern.TCheckbutton'
        )
        save_check.pack(anchor=tk.W, pady=2)
        
        self.stream_var = tk.BooleanVar(value=self.config["stream_to_server"])
        stream_check = ttk.Checkbutton(
            right_col,
            text="🌐 Stream to Server",
            variable=self.stream_var,
            style='Modern.TCheckbutton'
        )
        stream_check.pack(anchor=tk.W, pady=2)
    
    def create_status_section(self):
        """Create status section"""
        status_frame = ttk.LabelFrame(
            self.main_container,
            text="📊 Status",
            style='Modern.TLabelframe',
            padding=15
        )
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Status indicators
        status_grid = ttk.Frame(status_frame, style='Modern.TFrame')
        status_grid.pack(fill=tk.X)
        
        # Capture status
        self.capture_status_label = ttk.Label(
            status_grid,
            text="⏹️ Stopped",
            style='Modern.TLabel'
        )
        self.capture_status_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        # Frame count
        self.frame_count_label = ttk.Label(
            status_grid,
            text="📸 Frames: 0",
            style='Modern.TLabel'
        )
        self.frame_count_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        # Selected source
        self.source_label = ttk.Label(
            status_grid,
            text="🎯 Source: None",
            style='Modern.TLabel'
        )
        self.source_label.grid(row=0, column=2, sticky=tk.W)
    
    def create_control_section(self):
        """Create control buttons section"""
        control_frame = ttk.Frame(self.main_container, style='Modern.TFrame')
        control_frame.pack(fill=tk.X, pady=10)
        
        # Left side buttons
        left_controls = ttk.Frame(control_frame, style='Modern.TFrame')
        left_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.start_btn = ttk.Button(
            left_controls,
            text="▶️ Start Capture",
            command=self.on_start_capture,
            style='Success.TButton'
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10), ipadx=10, ipady=5)
        
        self.stop_btn = ttk.Button(
            left_controls,
            text="⏹️ Stop Capture",
            command=self.on_stop_capture,
            style='Warning.TButton',
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10), ipadx=10, ipady=5)
        
        # Right side buttons
        right_controls = ttk.Frame(control_frame, style='Modern.TFrame')
        right_controls.pack(side=tk.RIGHT)
        
        ttk.Button(
            right_controls,
            text="💾 Save Settings",
            command=self.save_settings,
            style='Modern.TButton'
        ).pack(side=tk.RIGHT, padx=(10, 0), ipadx=10, ipady=5)
        
        ttk.Button(
            right_controls,
            text="🌐 Open Server",
            command=self.open_server_interface,
            style='Modern.TButton'
        ).pack(side=tk.RIGHT, padx=(10, 0), ipadx=10, ipady=5)
    
    def get_windows(self) -> List[WindowInfo]:
        """Get list of all visible windows"""
        windows = []
        
        if not HAS_WIN32:
            return windows
        
        def enum_windows_proc(hwnd, lParam):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and len(title) > 1:
                    rect = win32gui.GetWindowRect(hwnd)
                    # Filter out very small windows
                    width, height = rect[2] - rect[0], rect[3] - rect[1]
                    if width > 100 and height > 50:
                        windows.append(WindowInfo(hwnd, title, rect))
            return True
        
        try:
            win32gui.EnumWindows(enum_windows_proc, 0)
        except Exception as e:
            print(f"Error enumerating windows: {e}")
        
        return windows
    
    def refresh_windows(self):
        """Refresh the window list"""
        self.window_listbox.delete(0, tk.END)
        self.windows = self.get_windows()
        
        for i, window in enumerate(self.windows):
            # Create formatted display text
            display_text = f"🪟 {window.title}"
            if len(display_text) > 60:
                display_text = display_text[:57] + "..."
            display_text += f" ({window.width}×{window.height})"
            
            self.window_listbox.insert(tk.END, display_text)
        
        # Update status
        if self.windows:
            self.source_label.config(text=f"🎯 Source: {len(self.windows)} windows available")
        else:
            self.source_label.config(text="🎯 Source: No windows found")
    
    def on_mode_change(self):
        """Handle capture mode change"""
        mode = self.mode_var.get()
        
        if mode == "window":
            self.window_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
            self.source_label.config(text="🎯 Source: Window mode")
        else:
            self.window_frame.pack_forget()
            self.source_label.config(text="🎯 Source: Full screen")
        
        self.capture_mode = mode
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.config["output_directory"]
        )
        if directory:
            self.output_dir_var.set(directory)
    
    def update_config_from_gui(self):
        """Update configuration from GUI values"""
        self.config["server_url"] = self.server_url_var.get()
        self.config["output_directory"] = self.output_dir_var.get()
        self.config["save_locally"] = self.save_local_var.get()
        self.config["stream_to_server"] = self.stream_var.get()
        
        try:
            self.config["capture_interval"] = float(self.interval_var.get())
        except ValueError:
            self.config["capture_interval"] = 0.1
            self.interval_var.set("0.1")
    
    def capture_window(self, window_info: WindowInfo) -> Optional[Image.Image]:
        """Capture a specific window (same as before but with error handling)"""
        if not HAS_WIN32 or not HAS_PIL:
            return None
        
        try:
            # Method 1: Try PrintWindow
            hwndDC = win32gui.GetWindowDC(window_info.hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, window_info.width, window_info.height)
            saveDC.SelectObject(saveBitMap)
            
            result = win32gui.PrintWindow(window_info.hwnd, saveDC.GetSafeHdc(), 2)
            
            if result == 1:
                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)
                img = Image.frombuffer(
                    'RGB',
                    (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                    bmpstr, 'raw', 'BGRX', 0, 1
                )
                
                # Cleanup
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(window_info.hwnd, hwndDC)
                
                return img
            else:
                # Method 2: Screenshot fallback
                win32gui.SetForegroundWindow(window_info.hwnd)
                time.sleep(0.2)
                screenshot = ImageGrab.grab(bbox=window_info.rect)
                
                # Cleanup
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(window_info.hwnd, hwndDC)
                
                return screenshot
                
        except Exception as e:
            print(f"Window capture error: {e}")
            return None
    
    def capture_screen(self) -> Optional[Image.Image]:
        """Capture entire screen"""
        try:
            if HAS_MSS:
                with mss.mss() as sct:
                    screenshot = sct.grab(sct.monitors[0])
                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                    return img
            elif HAS_PIL:
                return ImageGrab.grab(all_screens=True)
        except Exception as e:
            print(f"Screen capture error: {e}")
            return None
    
    def save_image(self, image: Image.Image) -> str:
        """Save image locally"""
        if not image:
            return None
        
        output_dir = Path(self.config["output_directory"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time() * 1000)
        filename = f"capture_{timestamp}.png"
        filepath = output_dir / filename
        
        try:
            image.save(filepath, "PNG")
            return str(filepath)
        except Exception as e:
            print(f"Save error: {e}")
            return None
    
    def send_to_server(self, image: Image.Image) -> bool:
        """Send image to server"""
        if not image:
            return False
        
        try:
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            response = requests.post(
                self.config["server_url"],
                data=img_bytes.read(),
                headers={"Content-Type": "image/png"},
                timeout=5
            )
            
            return response.status_code == 200
        except Exception as e:
            print(f"Server send error: {e}")
            return False
    
    def capture_loop(self):
        """Main capture loop"""
        frame_count = 0
        
        while self.capture_active:
            try:
                image = None
                
                if self.capture_mode == "window" and self.selected_window:
                    image = self.capture_window(self.selected_window)
                elif self.capture_mode == "screen":
                    image = self.capture_screen()
                
                if image:
                    frame_count += 1
                    
                    # Update GUI from main thread
                    self.root.after(0, lambda: self.frame_count_label.config(
                        text=f"📸 Frames: {frame_count}"
                    ))
                    
                    if self.config["save_locally"]:
                        self.save_image(image)
                    
                    if self.config["stream_to_server"]:
                        self.send_to_server(image)
                
                time.sleep(self.config["capture_interval"])
                
            except Exception as e:
                print(f"Capture loop error: {e}")
                break
        
        # Update status when loop ends
        self.root.after(0, lambda: self.capture_status_label.config(text="⏹️ Stopped"))
    
    def on_start_capture(self):
        """Start capture"""
        if self.capture_active:
            return
        
        self.update_config_from_gui()
        
        if self.capture_mode == "window":
            selection = self.window_listbox.curselection()
            if not selection:
                messagebox.showerror("Error", "Please select a window to capture")
                return
            self.selected_window = self.windows[selection[0]]
        
        self.capture_active = True
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Update GUI
        self.capture_status_label.config(text="▶️ Recording")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        if self.selected_window:
            self.source_label.config(text=f"🎯 Source: {self.selected_window.title}")
        else:
            self.source_label.config(text="🎯 Source: Full Screen")
    
    def on_stop_capture(self):
        """Stop capture"""
        self.capture_active = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        
        # Update GUI
        self.capture_status_label.config(text="⏹️ Stopped")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def save_settings(self):
        """Save current settings"""
        self.update_config_from_gui()
        self.save_config()
        messagebox.showinfo("Settings", "✅ Settings saved successfully!")
    
    def open_server_interface(self):
        """Open server web interface"""
        import webbrowser
        server_url = self.server_url_var.get().replace('/stream', '')
        try:
            webbrowser.open(server_url)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open browser: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.capture_active:
            self.on_stop_capture()
        
        self.save_config()
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    try:
        app = ModernScreenCaptureApp()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
