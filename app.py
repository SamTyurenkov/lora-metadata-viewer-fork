#!/usr/bin/env python3
"""
LoRA Metadata Viewer Flask Server
Serves safetensors files from a specified directory and provides a web interface to view their metadata.
"""

import os
import json
import struct
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

def extract_safetensors_metadata(file_path):
    """Extract metadata from a safetensors file"""
    try:
        with open(file_path, 'rb') as f:
            # Read the first 8 bytes to get metadata size
            header_bytes = f.read(8)
            if len(header_bytes) < 8:
                return None
            
            # Extract metadata size (little-endian uint32)
            metadata_size = struct.unpack('<I', header_bytes[:4])[0]
            
            # Read the metadata
            metadata_bytes = f.read(metadata_size)
            if len(metadata_bytes) < metadata_size:
                return None
            
            # Parse JSON header
            header = json.loads(metadata_bytes.decode('utf-8'))
            
            # Extract the __metadata__ section
            formatted_metadata = header.get('__metadata__', {})
            
            if not formatted_metadata:
                return None
            
            # Process metadata - convert JSON strings to objects where applicable
            metadata = {}
            for key, value in formatted_metadata.items():
                if isinstance(value, str):
                    try:
                        # Try to parse as JSON
                        metadata[key] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        # If it fails, keep as string
                        metadata[key] = value
                else:
                    metadata[key] = value
            
            return {
                'metadata': metadata,
                'formatted_metadata': formatted_metadata
            }
            
    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")
        return None

def update_safetensors_metadata(file_path, new_metadata):
    """Update metadata in a safetensors file"""
    try:
        # Read the entire file
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # Parse the header to get the structure
        header_bytes = file_data[:8]
        if len(header_bytes) < 8:
            return False
        
        metadata_size = struct.unpack('<I', header_bytes[:4])[0]
        metadata_bytes = file_data[8:8+metadata_size]
        
        if len(metadata_bytes) < metadata_size:
            return False
        
        # Parse the existing header
        header = json.loads(metadata_bytes.decode('utf-8'))
        
        # Convert new metadata to the format expected by safetensors
        formatted_metadata = {}
        for key, value in new_metadata.items():
            if isinstance(value, (dict, list)):
                # Convert complex objects to JSON strings
                formatted_metadata[key] = json.dumps(value)
            else:
                formatted_metadata[key] = value
        
        # Update the __metadata__ section
        header['__metadata__'] = formatted_metadata
        
        # Serialize the new header
        new_header_json = json.dumps(header, separators=(',', ':'))
        new_header_bytes = new_header_json.encode('utf-8')
        new_metadata_size = len(new_header_bytes)
        
        # Create new header with size
        new_header_with_size = struct.pack('<I', new_metadata_size) + b'\x00\x00\x00\x00' + new_header_bytes
        
        # Get the tensor data (everything after the original metadata)
        tensor_data = file_data[8+metadata_size:]
        
        # Create the new file content
        new_file_data = new_header_with_size + tensor_data
        
        # Write the updated file
        with open(file_path, 'wb') as f:
            f.write(new_file_data)
        
        return True
        
    except Exception as e:
        print(f"Error updating metadata in {file_path}: {e}")
        return False

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
    """Serve a specific file with streaming support for large files"""
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
        
        # Get file size for headers
        file_size = os.path.getsize(file_path)
        
        print(f"Serving file: {file_path} ({file_size:,} bytes)")
        
        # Serve the file with appropriate headers for large file streaming
        response = send_file(
            file_path,
            as_attachment=False,
            download_name=os.path.basename(file_path),
            mimetype='application/octet-stream'
        )
        
        # Add headers to help with large file downloads
        response.headers['Content-Length'] = str(file_size)
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'no-cache'
        
        return response
    
    except Exception as e:
        print(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': f'Error serving file: {str(e)}'}), 500

@app.route('/api/metadata/<path:filename>')
def get_file_metadata(filename):
    """Extract and return metadata from a specific safetensors file"""
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
        
        # Check if it's a safetensors file
        if not file_path.lower().endswith('.safetensors'):
            return jsonify({'error': 'Only safetensors files are supported for metadata extraction'}), 400
        
        print(f"Extracting metadata from: {file_path}")
        
        # Extract metadata
        result = extract_safetensors_metadata(file_path)
        
        if result is None:
            return jsonify({'error': 'No metadata found or failed to parse metadata'}), 404
        
        # Add file info
        file_info = get_file_info(file_path)
        result['file_info'] = file_info
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error extracting metadata from {filename}: {str(e)}")
        return jsonify({'error': f'Error extracting metadata: {str(e)}'}), 500

@app.route('/api/metadata/<path:filename>', methods=['PUT'])
def update_file_metadata(filename):
    """Update metadata in a specific safetensors file"""
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
        
        # Check if it's a safetensors file
        if not file_path.lower().endswith('.safetensors'):
            return jsonify({'error': 'Only safetensors files are supported for metadata updates'}), 400
        
        # Get the new metadata from the request
        request_data = request.get_json()
        if not request_data or 'metadata' not in request_data:
            return jsonify({'error': 'No metadata provided'}), 400
        
        new_metadata = request_data['metadata']
        
        print(f"Updating metadata in: {file_path}")
        
        # Update the metadata in the file
        success = update_safetensors_metadata(file_path, new_metadata)
        
        if not success:
            return jsonify({'error': 'Failed to update metadata'}), 500
        
        # Return the updated metadata
        result = extract_safetensors_metadata(file_path)
        if result is None:
            return jsonify({'error': 'Failed to read updated metadata'}), 500
        
        # Add file info
        file_info = get_file_info(file_path)
        result['file_info'] = file_info
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error updating metadata in {filename}: {str(e)}")
        return jsonify({'error': f'Error updating metadata: {str(e)}'}), 500

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
