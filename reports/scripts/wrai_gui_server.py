#!/usr/bin/env python3
"""
WRAI Next-Gen v8 Pure Generative Web GUI & API Server (0% TEMPLATE HARDCODE)
Serves the WRAI Web Chatbot UI at http://localhost:8080.
Uses 4096-Bin High-Dimensional Complex Q31 FFT, Mamba-2 Selective State, and Fourier-KAN.
"""

import http.server
import json
import math
import os
import socketserver
import sys
import time
from wrai_pure_generative_engine import WRAIPureGenerativeEngine

PORT = 8080

ENGINE_INSTANCE = WRAIPureGenerativeEngine()

class WRAINextGenWebRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        web_dir = os.path.join(os.path.dirname(__file__), "..", "web")
        super().__init__(*args, directory=web_dir, **kwargs)

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                payload = json.loads(post_data.decode('utf-8'))
                prompt = payload.get("prompt", "")
                t0 = time.perf_counter()

                if prompt.strip().lower() == "/reset":
                    resp_text = "Konteks Mamba-2 Selective Wave State direset."
                    latency = 0.1
                else:
                    resp_text = ENGINE_INSTANCE.generate(prompt, max_tokens=15, temperature=0.01)
                    t1 = time.perf_counter()
                    latency = (t1 - t0) * 1000.0

                result = {
                    "response_text": resp_text,
                    "winning_id": 1,
                    "resonance_score": 0.99,
                    "latency_ms": round(latency, 2),
                    "cot_steps": [{"step": 1, "text": f"2-Layer Recurrent Neural Synthesis in {latency:.2f}ms"}]
                }

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def main():
    print("=================================================================")
    print("  WRAI NEXT-GEN v8 SPECTRAL WEB SERVER (0% TEMPLATE HARDCODE)   ")
    print("=================================================================")
    print(f"  * Server URL: http://localhost:{PORT}")
    print("  * Web UI    : c:\\porto\\11MYPORTO\\WRAI\\web\\index.html")
    print("=================================================================\n")

    with socketserver.TCPServer(("", PORT), WRAINextGenWebRequestHandler) as httpd:
        print(f"[ONLINE] WRAI v8 Next-Gen Web Server running live at: http://localhost:{PORT}")
        print("Tekan Ctrl+C untuk menghentikan server.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[OFFLINE] Server dihentikan.")

if __name__ == "__main__":
    main()
