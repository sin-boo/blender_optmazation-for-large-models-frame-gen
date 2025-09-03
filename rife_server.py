import http.server
import socketserver
import json

class RIFEHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        # In a later step, we'll process the data
        print(f"Received {content_length} bytes of data.")
        
        # Placeholder for now
        response_data = {"status": "success", "message": "Frame received and processed."}
        
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

PORT = 8000

# Create a simple server
with socketserver.TCPServer(("", PORT), RIFEHandler) as httpd:
    print(f"Server is running at http://localhost:{PORT}")
    httpd.serve_forever()
