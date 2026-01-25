#!/usr/bin/env python3
"""
Simple HTTP server to run the Recommendation Systems course viewer.
"""
import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def main():
    # Change to the directory containing this script
    os.chdir(Path(__file__).parent)

    Handler = MyHTTPRequestHandler

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/index.html"
        print(f"\n" + "="*60)
        print(f"  CS 329R: Recommendation Systems Course Viewer")
        print(f"="*60)
        print(f"\n  Server running at: {url}")
        print(f"\n  Press Ctrl+C to stop the server\n")
        print("="*60 + "\n")

        # Open browser automatically
        webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
            httpd.shutdown()
            print("Server stopped.\n")

if __name__ == "__main__":
    main()
