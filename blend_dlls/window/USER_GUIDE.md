# 🖥️ Screen Capture Application - User Guide

## ✨ Ultimate Self-Sufficient Edition

This screen capture application is designed to be **completely self-sufficient** - it can automatically install Python and all dependencies without any manual setup required!

---

## 🚀 **Quick Start (Recommended)**

### Windows Users
**Double-click:** `LAUNCH_APP.bat`

That's it! The application will:
- ✅ Check if Python is installed
- ✅ Download and install Python if needed (portable, no system changes)
- ✅ Install all required dependencies automatically
- ✅ Launch the modern GUI interface
- ✅ Remember your setup for future launches

### macOS/Linux Users
**Run in terminal:** `python3 launch_app.py`

---

## 📋 **What This App Does**

- **🪟 Window Capture**: Capture specific application windows (OBS-style)
- **🖥️ Screen Capture**: Full screen or region capture
- **📱 Modern GUI**: Dark theme interface with intuitive controls
- **🌐 HTTP Streaming**: Stream captures over HTTP for remote viewing
- **💾 Local Storage**: Save captures as images locally
- **⚙️ Zero Configuration**: Works out of the box with sensible defaults

---

## 🛠️ **Available Launchers**

You have multiple ways to start the app, but **`LAUNCH_APP.bat`** is recommended for most users:

### Primary Launchers ⭐
- **`LAUNCH_APP.bat`** - Ultimate Windows launcher (RECOMMENDED)
- **`launch_app.py`** - Ultimate cross-platform Python launcher

### Alternative Launchers
- **`START_HERE.bat`** - Simple Windows launcher
- **`INSTALL_AND_RUN.bat`** - First-time setup focused launcher
- **`bootstrap.py`** - Advanced Python-based bootstrap
- **`run.py`** - Original comprehensive launcher

---

## 🔧 **First Run Experience**

On your **first run**, the app will:

1. **🔍 System Check**: Detect your operating system and architecture
2. **🐍 Python Setup**: Download and install Python 3.11.9 if needed (~25MB download)
3. **📦 Dependencies**: Install required packages (Pillow, Flask, MSS, etc.)
4. **🔬 Verification**: Test that everything is working correctly
5. **🚀 Launch**: Start the modern GUI interface

**Total time**: 2-5 minutes depending on internet speed.

### What Gets Installed?
- **Portable Python** (Windows only): Downloaded to `tools/python/` folder
- **Python Packages**: Pillow, Flask, MSS, Requests, PyWin32 (Windows)
- **No system changes**: Everything is self-contained in the app folder

---

## ⚡ **Subsequent Runs**

After the first setup, launching is **instant**:
- The app remembers your setup
- Checks that Python and dependencies are still working
- Launches immediately if everything is ready
- Shows troubleshooting options if issues are detected

---

## 🛠️ **Troubleshooting**

### If the app won't start:
1. **Try again**: First runs can sometimes have temporary issues
2. **Check internet**: Required for downloading Python/dependencies
3. **Run as Administrator**: May help with Windows permission issues
4. **Use advanced menu**: Run `launch_app.py` directly for more options

### Advanced Options
Run `launch_app.py` and select from the advanced menu:
- **Force Reinstall Python**: Clean install if Python is corrupted
- **Reinstall Dependencies**: Update packages if they're broken
- **System Diagnostics**: Check what's installed and working
- **Clean Environment**: Reset everything to start fresh

### Manual Python Installation
If automatic installation fails:
1. Visit: https://www.python.org/downloads/
2. Download Python 3.7 or newer
3. **Important**: Check "Add Python to PATH" during installation
4. Run the app launcher again

---

## 📁 **Project Structure**

```
window/
├── LAUNCH_APP.bat          ⭐ Start here (Windows)
├── launch_app.py           ⭐ Ultimate launcher (all platforms)
├── src/
│   ├── main_app.py         📱 Main application entry point
│   ├── modern_gui.py       🎨 Modern GUI interface
│   └── capture_server.py   🌐 HTTP server component
├── tools/
│   ├── python/             🐍 Portable Python (auto-installed)
│   └── downloads/          📥 Temporary download cache
├── config/
│   └── config_template.json ⚙️ Configuration template
├── assets/                 🖼️ App resources (empty)
├── captured_frames/        📷 Screenshot outputs (auto-created)
└── requirements.txt        📋 Python dependencies list
```

---

## 🎯 **Key Features**

### Window Capture
- **Live Window List**: Refreshes automatically to show all open windows
- **OBS-Style Interface**: Familiar window selection UI
- **Smart Filtering**: Shows only relevant capturable windows
- **Preview**: See window thumbnails before capture

### Screen Capture
- **Full Screen**: Capture entire screen or specific monitor
- **Region Capture**: Select custom areas to capture
- **High Quality**: Lossless PNG output by default
- **Batch Capture**: Multiple screenshots with timestamps

### HTTP Streaming
- **Live Streaming**: Stream captures over HTTP for remote viewing
- **Real-time**: Low-latency capture streaming
- **Cross-platform**: Access from any device with a web browser
- **Port Configuration**: Customizable port settings

### Modern Interface
- **Dark Theme**: Easy on the eyes for extended use
- **Responsive**: Works on different screen sizes
- **Intuitive**: Clear icons and organized controls
- **Status Feedback**: Real-time status updates

---

## 🚨 **Common Issues & Solutions**

| Issue | Solution |
|-------|----------|
| "Python not found" | Use `INSTALL_AND_RUN.bat` or install Python manually |
| "Dependencies missing" | Run the launcher again - it will install them |
| "Permission denied" | Run as Administrator on Windows |
| "Download failed" | Check internet connection and antivirus settings |
| "App won't start" | Try the advanced menu in `launch_app.py` |
| "GUI looks broken" | Dependencies might be incomplete - reinstall them |

---

## 🎉 **Sharing This App**

To share this app with others:

1. **Zip the entire `window` folder**
2. **Send the zip file**
3. **Recipients just double-click `LAUNCH_APP.bat`** (Windows) or run `python3 launch_app.py`

**No installation instructions needed** - the app handles everything automatically!

---

## 📞 **Support**

The app includes built-in diagnostics accessible through the advanced menu. If you encounter issues:

1. Run `launch_app.py` directly
2. Choose option 5 (System Diagnostics)
3. This will show you exactly what's installed and working

---

## 🎨 **Customization**

- **Configuration**: Edit files in the `config/` folder
- **Themes**: The modern GUI supports theme customization
- **Output Folder**: Screenshots are saved to `captured_frames/`
- **Server Settings**: HTTP streaming can be configured in the GUI

---

**Enjoy your self-sufficient screen capture application!** 🎉
