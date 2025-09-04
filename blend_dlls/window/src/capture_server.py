#!/usr/bin/env python3
"""
Capture Server - HTTP server to receive and process captured frames
Simple Flask-based server that replaces the hardcoded server endpoint
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from flask import Flask, request, jsonify, send_from_directory
    HAS_FLASK = True
except ImportError:
    print("Flask not available. Install with: pip install flask")
    HAS_FLASK = False
    # Fallback to basic HTTP server
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse
        HAS_HTTP_SERVER = True
    except ImportError:
        HAS_HTTP_SERVER = False

try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

class CaptureServerConfig:
    def __init__(self, config_file: str = "server_config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        default_config = {
            "host": "127.0.0.1",
            "port": 8000,
            "output_directory": "./received_frames",
            "save_frames": True,
            "max_file_size_mb": 10,
            "enable_web_interface": True,
            "frame_format": "png",  # png, jpg, webp
            "keep_frames_days": 7,  # auto-cleanup after N days
            "enable_logging": True,
            "log_file": "./server.log"
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"Error loading server config: {e}")
                return default_config
        
        return default_config
    
    def save_config(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving server config: {e}")

class FlaskCaptureServer:
    def __init__(self, config: CaptureServerConfig):
        self.config = config.config
        self.app = Flask(__name__)
        self.frame_count = 0
        self.setup_routes()
        self.setup_output_directory()
    
    def setup_output_directory(self):
        """Create output directory if it doesn't exist"""
        output_dir = Path(self.config["output_directory"])
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        if self.config["enable_logging"]:
            try:
                with open(self.config["log_file"], 'a') as f:
                    f.write(log_message + "\\n")
            except Exception as e:
                print(f"Error writing to log file: {e}")
    
    def save_frame(self, image_data: bytes, filename: str = None) -> str:
        """Save received frame to disk"""
        if not self.config["save_frames"]:
            return None
        
        output_dir = Path(self.config["output_directory"])
        
        if filename is None:
            timestamp = int(time.time() * 1000)
            filename = f"frame_{timestamp}_{self.frame_count:06d}.{self.config['frame_format']}"
        
        filepath = output_dir / filename
        
        try:
            if HAS_PIL:
                # Process image with PIL for format conversion if needed
                img = Image.open(io.BytesIO(image_data))
                
                # Convert format if necessary
                if self.config["frame_format"].lower() == "jpg":
                    # Convert RGBA to RGB for JPEG
                    if img.mode == "RGBA":
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1] if len(img.split()) == 4 else None)
                        img = background
                    img.save(filepath, "JPEG", quality=85)
                elif self.config["frame_format"].lower() == "webp":
                    img.save(filepath, "WEBP", quality=85)
                else:
                    img.save(filepath, "PNG")
            else:
                # Direct save without processing
                with open(filepath, 'wb') as f:
                    f.write(image_data)
            
            self.frame_count += 1
            return str(filepath)
        
        except Exception as e:
            self.log(f"Error saving frame: {e}")
            return None
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            if not self.config["enable_web_interface"]:
                return "Server is running", 200
            
            return f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Screen Capture Server</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .status {{ background: #e8f5e8; padding: 10px; border-radius: 5px; }}
                    .config {{ background: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Screen Capture Server</h1>
                <div class="status">
                    <h3>Status: Running</h3>
                    <p>Frames received: {self.frame_count}</p>
                    <p>Server endpoint: http://{self.config["host"]}:{self.config["port"]}/stream</p>
                </div>
                
                <div class="config">
                    <h3>Configuration</h3>
                    <table>
                        <tr><th>Setting</th><th>Value</th></tr>
                        <tr><td>Output Directory</td><td>{self.config["output_directory"]}</td></tr>
                        <tr><td>Save Frames</td><td>{self.config["save_frames"]}</td></tr>
                        <tr><td>Frame Format</td><td>{self.config["frame_format"]}</td></tr>
                        <tr><td>Max File Size</td><td>{self.config["max_file_size_mb"]} MB</td></tr>
                    </table>
                </div>
                
                <div style="margin-top: 20px;">
                    <h3>Recent Frames</h3>
                    <a href="/frames">View Frames Directory</a>
                </div>
            </body>
            </html>
            '''
        
        @self.app.route('/stream', methods=['POST'])
        def receive_stream():
            """Receive streamed frames"""
            try:
                # Check content type
                content_type = request.headers.get('Content-Type', '')
                if not content_type.startswith('image/'):
                    return jsonify({"error": "Invalid content type. Expected image data."}), 400
                
                # Get image data
                image_data = request.get_data()
                
                # Check file size
                size_mb = len(image_data) / (1024 * 1024)
                if size_mb > self.config["max_file_size_mb"]:
                    return jsonify({"error": f"File too large. Max size: {self.config['max_file_size_mb']} MB"}), 413
                
                # Save frame
                saved_path = self.save_frame(image_data)
                
                if saved_path:
                    self.log(f"Frame saved: {saved_path} ({size_mb:.2f} MB)")
                    return jsonify({
                        "status": "success",
                        "message": "Frame received and saved",
                        "path": saved_path,
                        "frame_count": self.frame_count
                    }), 200
                else:
                    self.log("Frame received but not saved (save_frames disabled)")
                    return jsonify({
                        "status": "success",
                        "message": "Frame received",
                        "frame_count": self.frame_count
                    }), 200
            
            except Exception as e:
                self.log(f"Error processing stream: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/frames')
        def list_frames():
            """List saved frames"""
            if not self.config["enable_web_interface"]:
                return "Web interface disabled", 403
            
            try:
                output_dir = Path(self.config["output_directory"])
                if not output_dir.exists():
                    return "No frames directory found", 404
                
                # Get all image files
                image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'}
                frames = []
                
                for file_path in output_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                        stat = file_path.stat()
                        frames.append({
                            'name': file_path.name,
                            'size': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        })
                
                # Sort by modification time (newest first)
                frames.sort(key=lambda x: x['modified'], reverse=True)
                
                html = '''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Captured Frames</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; }
                        table { border-collapse: collapse; width: 100%; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                        .size { text-align: right; }
                        a { text-decoration: none; color: #0066cc; }
                        a:hover { text-decoration: underline; }
                    </style>
                </head>
                <body>
                    <h1>Captured Frames</h1>
                    <p><a href="/">&larr; Back to Server Status</a></p>
                    <table>
                        <tr>
                            <th>Filename</th>
                            <th>Size (KB)</th>
                            <th>Modified</th>
                            <th>Actions</th>
                        </tr>
                '''
                
                for frame in frames:
                    size_kb = frame['size'] / 1024
                    html += f'''
                        <tr>
                            <td>{frame['name']}</td>
                            <td class="size">{size_kb:.1f}</td>
                            <td>{frame['modified']}</td>
                            <td><a href="/frames/{frame['name']}" target="_blank">View</a></td>
                        </tr>
                    '''
                
                html += '''
                    </table>
                </body>
                </html>
                '''
                
                return html
            
            except Exception as e:
                return f"Error listing frames: {e}", 500
        
        @self.app.route('/frames/<filename>')
        def serve_frame(filename):
            """Serve individual frame files"""
            if not self.config["enable_web_interface"]:
                return "Web interface disabled", 403
            
            try:
                output_dir = Path(self.config["output_directory"])
                return send_from_directory(output_dir, filename)
            except Exception as e:
                return f"Error serving frame: {e}", 404
        
        @self.app.route('/config', methods=['GET'])
        def get_config():
            """Get server configuration"""
            return jsonify(self.config)
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """Get server statistics"""
            try:
                output_dir = Path(self.config["output_directory"])
                frame_count = 0
                total_size = 0
                
                if output_dir.exists():
                    for file_path in output_dir.iterdir():
                        if file_path.is_file():
                            frame_count += 1
                            total_size += file_path.stat().st_size
                
                return jsonify({
                    "frames_received": self.frame_count,
                    "frames_stored": frame_count,
                    "total_size_mb": total_size / (1024 * 1024),
                    "server_uptime": "N/A",  # Could be implemented
                    "output_directory": str(output_dir.absolute())
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    def run(self):
        """Run the Flask server"""
        self.log(f"Starting capture server on http://{self.config['host']}:{self.config['port']}")
        self.log(f"Stream endpoint: http://{self.config['host']}:{self.config['port']}/stream")
        self.log(f"Output directory: {Path(self.config['output_directory']).absolute()}")
        
        try:
            self.app.run(
                host=self.config["host"],
                port=self.config["port"],
                debug=False,
                threaded=True
            )
        except Exception as e:
            self.log(f"Server error: {e}")

class BasicHTTPCaptureServer:
    """Fallback HTTP server implementation if Flask is not available"""
    
    def __init__(self, config: CaptureServerConfig):
        self.config = config.config
        self.frame_count = 0
        self.setup_output_directory()
    
    def setup_output_directory(self):
        output_dir = Path(self.config["output_directory"])
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_frame(self, image_data: bytes) -> str:
        if not self.config["save_frames"]:
            return None
        
        output_dir = Path(self.config["output_directory"])
        timestamp = int(time.time() * 1000)
        filename = f"frame_{timestamp}_{self.frame_count:06d}.png"
        filepath = output_dir / filename
        
        try:
            with open(filepath, 'wb') as f:
                f.write(image_data)
            self.frame_count += 1
            return str(filepath)
        except Exception as e:
            print(f"Error saving frame: {e}")
            return None
    
    class RequestHandler(BaseHTTPRequestHandler):
        def __init__(self, server_instance, *args, **kwargs):
            self.server_instance = server_instance
            super().__init__(*args, **kwargs)
        
        def do_POST(self):
            if self.path == '/stream':
                try:
                    content_length = int(self.headers.get('Content-Length', 0))
                    if content_length == 0:
                        self.send_error(400, "No content")
                        return
                    
                    image_data = self.rfile.read(content_length)
                    saved_path = self.server_instance.save_frame(image_data)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    response = {
                        "status": "success",
                        "message": "Frame received",
                        "frame_count": self.server_instance.frame_count
                    }
                    
                    if saved_path:
                        response["path"] = saved_path
                    
                    self.wfile.write(json.dumps(response).encode())
                    
                except Exception as e:
                    self.send_error(500, str(e))
            else:
                self.send_error(404, "Not found")
        
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                html = f'''
                <html>
                <body>
                    <h1>Basic Capture Server</h1>
                    <p>Server is running on port {self.server_instance.config["port"]}</p>
                    <p>Stream endpoint: POST /stream</p>
                    <p>Frames received: {self.server_instance.frame_count}</p>
                </body>
                </html>
                '''
                self.wfile.write(html.encode())
            else:
                self.send_error(404, "Not found")
    
    def run(self):
        print(f"Starting basic HTTP server on {self.config['host']}:{self.config['port']}")
        
        def handler(*args, **kwargs):
            return self.RequestHandler(self, *args, **kwargs)
        
        try:
            server = HTTPServer((self.config["host"], self.config["port"]), handler)
            server.serve_forever()
        except Exception as e:
            print(f"Server error: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Screen Capture Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--output-dir', default='./received_frames', help='Output directory for frames')
    parser.add_argument('--no-save', action='store_true', help='Disable saving frames to disk')
    parser.add_argument('--config', default='server_config.json', help='Config file path')
    
    args = parser.parse_args()
    
    # Load config
    config = CaptureServerConfig(args.config)
    
    # Override with command line arguments
    if args.host != '127.0.0.1':
        config.config['host'] = args.host
    if args.port != 8000:
        config.config['port'] = args.port
    if args.output_dir != './received_frames':
        config.config['output_directory'] = args.output_dir
    if args.no_save:
        config.config['save_frames'] = False
    
    # Save updated config
    config.save_config()
    
    # Start appropriate server
    if HAS_FLASK:
        print("Using Flask server (recommended)")
        server = FlaskCaptureServer(config)
    elif HAS_HTTP_SERVER:
        print("Using basic HTTP server (Flask not available)")
        server = BasicHTTPCaptureServer(config)
    else:
        print("No HTTP server implementation available!")
        return
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\\nShutting down server...")

if __name__ == '__main__':
    main()
