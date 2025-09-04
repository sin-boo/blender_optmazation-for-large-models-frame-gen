# 🚀 QUICK START GUIDE

## For Immediate Use (Recommended)

**Windows Users**: Double-click `START_HERE.bat` 
**Mac/Linux Users**: Run `python3 run.py`

That's it! The application will automatically:
- ✅ Check for Python (install if missing)
- ✅ Install all dependencies 
- ✅ Launch the modern GUI

---

## 🖥️ What You Get

### Modern GUI Features
- 🌙 **Dark Theme** - Easy on the eyes
- 🪟 **OBS-Style Window Selection** - Choose any window from a list
- 📺 **Screen Capture** - Capture entire screen or monitors  
- ⚡ **Real-time Streaming** - Stream to HTTP server
- 💾 **Local Storage** - Save frames to disk
- ⚙️ **Zero Configuration** - Works out of the box

### Self-Sufficient
- 🐍 **Auto Python Install** - Downloads Python if not found
- 📦 **Auto Dependencies** - Installs all required packages
- 🔄 **Portable** - Works on any Windows computer
- 💼 **Shareable** - Copy folder to any computer and run

---

## 🎯 How to Use

1. **Launch**: Double-click `START_HERE.bat`
2. **Select Source**: 
   - Choose "Capture Window" and select a window from the list
   - Or choose "Capture Screen" for full screen
3. **Configure** (optional):
   - Server URL (default: http://127.0.0.1:8000/stream)
   - Capture rate (default: 10 FPS)
   - Output folder
4. **Start**: Click "▶️ Start Capture"

---

## 🌐 Built-in Server

The app includes its own HTTP server:
- **Start Server**: Use menu option or run `python src/capture_server.py`
- **Web Interface**: Open http://127.0.0.1:8000 in browser
- **View Frames**: See all captured frames in real-time

---

## 📁 File Structure

```
📁 Screen Capture App/
├── 🚀 START_HERE.bat          ← DOUBLE-CLICK THIS
├── 🚀 run.py                  ← Main launcher  
├── 📖 README.md               ← Full documentation
├── 📖 QUICK_START.md          ← This file
├── 📄 requirements.txt        ← Dependencies list
│
├── 📁 src/                    ← Application code
│   ├── main_app.py           ← Main app entry point
│   ├── modern_gui.py         ← Modern GUI interface  
│   └── capture_server.py     ← HTTP server
│
├── 📁 config/                 ← Configuration files
├── 📁 tools/                  ← Downloaded tools (Python, etc.)
└── 📁 captured_frames/        ← Your captured images
```

---

## 🔧 Troubleshooting

**"Python not found"**
- Double-click `START_HERE.bat` (it will install Python)
- Or download Python from https://python.org

**"No windows found"**  
- Click "🔄 Refresh" button
- Make sure windows are visible and not minimized

**"Connection refused"**
- Start the server first (menu option 3)
- Check server URL is correct

**"Import errors"**
- Run again - first run installs dependencies
- All dependencies install automatically

---

## 🎉 Ready to Share

This entire folder can be copied to any computer and will work immediately!
Just share the whole folder and tell people to double-click `START_HERE.bat`.

**No installation required - it's completely self-sufficient!** 🚀
