#!/usr/bin/env python3
"""
LoRA Metadata Viewer Flask Server
Serves safetensors files from a specified directory and provides a web interface to view their metadata.
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template_string, jsonify, send_file, request
from flask_cors import CORS
import argparse

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the files directory
FILES_DIR = None

def get_file_info(file_path):
    """Get basic file information"""
    try:
        stat = os.stat(file_path)
        return {
            'name': os.path.basename(file_path),
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'path': str(file_path)
        }
    except Exception as e:
        print(f"Error getting file info for {file_path}: {e}")
        return None

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(script_dir, 'index.html')
        
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return f"index.html not found. Looking for it at: {os.path.join(script_dir, 'index.html')}", 404

@app.route('/api/files')
def list_files():
    """API endpoint to list all safetensors files in the configured directory"""
    if not FILES_DIR:
        return jsonify({'error': 'Files directory not configured'}), 500
    
    if not os.path.exists(FILES_DIR):
        return jsonify({'error': f'Directory does not exist: {FILES_DIR}'}), 404
    
    try:
        files = []
        # Walk through directory and subdirectories
        for root, dirs, filenames in os.walk(FILES_DIR):
            for filename in filenames:
                if filename.lower().endswith(('.safetensors', '.gguf')):
                    file_path = os.path.join(root, filename)
                    file_info = get_file_info(file_path)
                    if file_info:
                        # Add relative path from FILES_DIR
                        rel_path = os.path.relpath(file_path, FILES_DIR)
                        file_info['relative_path'] = rel_path
                        files.append(file_info)
        
        # Sort files by name
        files.sort(key=lambda x: x['name'].lower())
        
        return jsonify({
            'files': files,
            'total': len(files),
            'directory': FILES_DIR
        })
    
    except Exception as e:
        return jsonify({'error': f'Error listing files: {str(e)}'}), 500

@app.route('/api/file/<path:filename>')
def serve_file(filename):
    """Serve a specific file"""
    if not FILES_DIR:
        return jsonify({'error': 'Files directory not configured'}), 500
    
    try:
        # Construct full path and ensure it's within FILES_DIR
        file_path = os.path.join(FILES_DIR, filename)
        file_path = os.path.abspath(file_path)
        files_dir_abs = os.path.abspath(FILES_DIR)
        
        # Security check: ensure the file is within the allowed directory
        if not file_path.startswith(files_dir_abs):
            return jsonify({'error': 'Access denied'}), 403
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Serve the file with appropriate headers
        return send_file(
            file_path,
            as_attachment=False,
            download_name=os.path.basename(file_path),
            mimetype='application/octet-stream'
        )
    
    except Exception as e:
        return jsonify({'error': f'Error serving file: {str(e)}'}), 500

@app.route('/api/info')
def server_info():
    """Get server information"""
    return jsonify({
        'files_directory': FILES_DIR,
        'server_mode': True,
        'supported_formats': ['.safetensors', '.gguf']
    })

def main():
    parser = argparse.ArgumentParser(description='LoRA Metadata Viewer Server')
    parser.add_argument(
        '--directory', '-d',
        required=True,
        help='Directory containing safetensors files to serve'
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Validate and set the files directory
    global FILES_DIR
    FILES_DIR = os.path.abspath(args.directory)
    
    if not os.path.exists(FILES_DIR):
        print(f"Error: Directory does not exist: {FILES_DIR}")
        return 1
    
    if not os.path.isdir(FILES_DIR):
        print(f"Error: Path is not a directory: {FILES_DIR}")
        return 1
    
    print(f"Starting LoRA Metadata Viewer Server")
    print(f"Files directory: {FILES_DIR}")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    
    # Count files to serve
    file_count = 0
    for root, dirs, files in os.walk(FILES_DIR):
        file_count += sum(1 for f in files if f.lower().endswith(('.safetensors', '.gguf')))
    print(f"Found {file_count} compatible files")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0

if __name__ == '__main__':
    exit(main())
