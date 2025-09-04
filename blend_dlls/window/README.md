# Screen Capture Application

A standalone screen/window capture application that replaces the Blender addon functionality with a cross-platform GUI application, similar to OBS window capture.

## Features

- **Window Capture**: Capture specific windows by selecting from a list (similar to OBS)
- **Screen Capture**: Capture entire screen or multiple monitors
- **Live Streaming**: Stream captured frames to an HTTP server in real-time
- **Local Saving**: Save captured frames locally in PNG/JPEG/WebP formats
- **Cross-platform**: Works on Windows, macOS, and Linux (best support on Windows)
- **Configurable**: Adjustable capture intervals, server URLs, and output settings
- **Web Interface**: Built-in web interface to view captured frames and server status

## Quick Start

### Option 1: Simple Launch (Recommended)
```bash
python launcher.py
```
This will automatically check dependencies, install missing packages, and show a menu.

### Option 2: Direct Launch
```bash
python launcher.py --direct
```
Launches the application directly after dependency checks.

### Option 3: Manual Launch
```bash
# Install dependencies first
pip install -r requirements.txt

# Launch the GUI application
python screen_capture_app.py

# Or launch just the server
python capture_server.py
```

## System Requirements

- **Python**: 3.7 or higher
- **Operating System**: 
  - Windows 10/11 (recommended for full window capture support)
  - macOS 10.14+ (limited window capture)
  - Linux (limited window capture)
- **Memory**: 2GB RAM minimum
- **Storage**: 100MB free space for application + space for captured frames

## Installation

1. **Download/Clone** the application to a folder on your computer
2. **Run the launcher**: Double-click `launcher.py` or run `python launcher.py`
3. **Install dependencies**: The launcher will automatically detect and install missing packages
4. **Launch**: Choose "Launch Screen Capture App" from the menu

### Dependencies

The application will automatically install these packages if missing:
- `Pillow` - Image processing
- `mss` - Fast screen capture
- `pywin32` - Windows API access (Windows only)
- `requests` - HTTP communication
- `flask` - Web server framework

## Usage

### Capturing Windows
1. Launch the application
2. Click "Refresh Windows" to populate the window list
3. Select a window from the list
4. Choose "Capture Window" mode
5. Configure server URL and capture settings
6. Click "Start Capture"

### Capturing Screen
1. Launch the application
2. Choose "Capture Screen" mode
3. Configure settings as needed
4. Click "Start Capture"

### Server Setup
The application includes a built-in HTTP server to receive captured frames:

1. **Launch Server**: 
   ```bash
   python capture_server.py
   ```
   Or use the launcher menu option "Launch Server Only"

2. **Access Web Interface**: Open http://127.0.0.1:8000 in your browser

3. **Configure Client**: Set the server URL in the capture application to `http://127.0.0.1:8000/stream`

## Configuration

### Application Settings
- **Server URL**: Where to send captured frames (default: `http://127.0.0.1:8000/stream`)
- **Capture Interval**: Time between captures in seconds (default: 0.1s = 10 FPS)
- **Save Locally**: Whether to save frames to disk
- **Stream to Server**: Whether to send frames to the HTTP server
- **Output Directory**: Where to save captured frames locally

### Server Settings
The server can be configured via `server_config.json`:
```json
{
  "host": "127.0.0.1",
  "port": 8000,
  "output_directory": "./received_frames",
  "save_frames": true,
  "max_file_size_mb": 10,
  "frame_format": "png"
}
```

## Troubleshooting

### Common Issues

**"No windows found"**
- On non-Windows systems, window capture support is limited
- Try using "Capture Screen" mode instead
- Ensure you have the required permissions

**"Connection refused" errors**
- Make sure the server is running
- Check that the server URL is correct
- Verify firewall settings aren't blocking the connection

**"Import Error" or missing modules**
- Run `python launcher.py` to automatically install dependencies
- Or manually install with `pip install -r requirements.txt`

**Poor capture performance**
- Increase the capture interval (reduce FPS)
- Reduce the size of the captured window/screen
- Close unnecessary applications

### Platform-Specific Notes

**Windows**
- Full window capture support using Windows API
- Individual window selection works best
- Requires `pywin32` package

**macOS/Linux**
- Limited window capture (falls back to screen region capture)
- Screen capture works normally
- Some features may require additional permissions

## File Structure

```
dllsstff/
├── launcher.py                 # Main launcher script
├── screen_capture_app.py      # GUI application
├── capture_server.py          # HTTP server
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── config.json               # App configuration (auto-generated)
├── server_config.json        # Server configuration (auto-generated)
├── captured_frames/          # Local frame storage
└── received_frames/          # Server-received frames
```

## API Reference

### Server Endpoints

- `GET /` - Web interface homepage
- `POST /stream` - Receive captured frames
- `GET /frames` - List saved frames
- `GET /frames/<filename>` - View specific frame
- `GET /config` - Get server configuration
- `GET /stats` - Get server statistics

### Frame Streaming

Send PNG image data to `/stream` endpoint:
```python
import requests

# Send image data
with open('image.png', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:8000/stream',
        data=f.read(),
        headers={'Content-Type': 'image/png'}
    )
```

## Migration from Blender Addon

If you were using the original Blender addon:

1. **Remove hardcoded paths**: The new application uses relative paths and configurable directories
2. **Update server URL**: Change from any hardcoded URLs to the configurable server URL
3. **Use GUI instead of Blender**: Launch the standalone application instead of the Blender addon
4. **Configure capture source**: Select windows or screen capture instead of Blender viewport

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on multiple platforms if possible
5. Submit a pull request

## License

This project is provided as-is for educational and personal use.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Check that your Python version is 3.7 or higher
4. Verify system compatibility

---

**Note**: This application replaces the original Blender addon and removes all hardcoded paths, making it portable across different computers and systems.
