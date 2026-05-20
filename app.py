#!/usr/bin/env python3
"""
LoRA Metadata Viewer Flask Server
Serves safetensors files from a specified directory and provides a web interface to view their metadata.
"""

import os
import json
import tempfile
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from flask import Flask, render_template_string, jsonify, send_file, request, Response
from flask_cors import CORS
import argparse

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the files directory
FILES_DIR = None

CIVITAI_VERSION_BY_HASH_URL = 'https://civitai.com/api/v1/model-versions/by-hash/'
CIVITAI_MODEL_URL = 'https://civitai.com/api/v1/models/'
PROTECTED_HTML_DESCRIPTIONS = frozenset({'4b14f5e9ff13.html', '7ed6e711b6e5.html'})

def get_html_description_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'html_description')

def _civitai_api_get(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'LoRA-Metadata-Viewer/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        print(f"CivitAI API request failed ({url}): {e}")
        return None

def build_civitai_description_html(version_data, model_data=None):
    """Match frontend civitai_description custom field formatting."""
    version_desc = (version_data or {}).get('description') or ''
    model_desc = (model_data or {}).get('description') or ''
    if version_desc and model_desc:
        return (
            version_desc
            + '<hr style="margin: 1em 0; border: 1px solid #ccc;">'
            + model_desc
        )
    return model_desc or version_desc or None

def lookup_civitai_by_hashes(autov2_hash, autov3_hash):
    for hash_value in (autov2_hash, autov3_hash):
        if not hash_value:
            continue
        data = _civitai_api_get(CIVITAI_VERSION_BY_HASH_URL + hash_value)
        if data:
            return data
    return None

def save_html_description(autov3_hash, html_content):
    if not autov3_hash or not html_content:
        return 'invalid'
    filename = f'{autov3_hash}.html'
    if filename in PROTECTED_HTML_DESCRIPTIONS:
        return 'protected'
    html_desc_dir = get_html_description_dir()
    os.makedirs(html_desc_dir, exist_ok=True)
    file_path = os.path.join(html_desc_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved CivitAI description to {file_path}")
    return 'saved'

def fetch_and_save_civitai_description(autov2_hash, autov3_hash):
    version_data = lookup_civitai_by_hashes(autov2_hash, autov3_hash)
    if not version_data:
        return 'not_found'
    if not autov3_hash:
        return 'no_hash'
    model_data = None
    model_id = version_data.get('modelId')
    if model_id:
        model_data = _civitai_api_get(f'{CIVITAI_MODEL_URL}{model_id}')
    html_content = build_civitai_description_html(version_data, model_data)
    if not html_content:
        return 'no_description'
    return save_html_description(autov3_hash, html_content)

class PrefixMiddleware:
    def __init__(self, app, prefix):
        self.app = app
        self.prefix = prefix.rstrip("/")

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path == self.prefix:
            environ["PATH_INFO"] = "/"
            environ["SCRIPT_NAME"] = self.prefix
        elif path.startswith(self.prefix + "/"):
            environ["PATH_INFO"] = path[len(self.prefix):] or "/"
            environ["SCRIPT_NAME"] = self.prefix
        return self.app(environ, start_response)

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

def calculate_file_hash_autov2(file_path, max_size_gb=2):
    """
    Calculate AutoV2 hash (SHA256 of entire file) with size limit.
    For files > max_size_gb, calculates hash in chunks.
    Returns first 10 characters of the hex hash.
    """
    try:
        file_size = os.path.getsize(file_path)
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        
        sha256_hash = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files
            chunk_size = 8192 * 1024  # 8MB chunks
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()[:10]
    except Exception as e:
        print(f"Error calculating AutoV2 hash for {file_path}: {e}")
        return None

def calculate_file_hash_autov3(file_path):
    """
    Calculate AutoV3 hash (SHA256 of specific 0x100000 byte blocks).
    This matches CivitAI's AutoV3 hash calculation.
    Returns first 12 characters of the hex hash.
    """
    try:
        file_size = os.path.getsize(file_path)
        block_size = 0x100000  # 1MB blocks
        
        sha256_hash = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            # Read first block
            first_block = f.read(block_size)
            sha256_hash.update(first_block)
            
            # If file is larger than 2 blocks, read middle and end blocks
            if file_size > block_size * 2:
                # Seek to middle block
                middle_pos = (file_size // 2) - (block_size // 2)
                f.seek(middle_pos)
                middle_block = f.read(block_size)
                sha256_hash.update(middle_block)
                
                # Seek to last block
                f.seek(-block_size, 2)  # 2 means from end of file
                last_block = f.read(block_size)
                sha256_hash.update(last_block)
        
        return sha256_hash.hexdigest()[:12]
    except Exception as e:
        print(f"Error calculating AutoV3 hash for {file_path}: {e}")
        return None

def extract_safetensors_metadata(file_path):
    """Extract metadata from a safetensors file"""
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            formatted_metadata = f.metadata() or {}
            
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
        # Convert new metadata to the format expected by safetensors
        formatted_metadata = {}
        for key, value in new_metadata.items():
            if isinstance(value, (dict, list)):
                # Convert complex objects to JSON strings with ensure_ascii=False
                formatted_metadata[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, str):
                # Keep strings as-is
                formatted_metadata[key] = value
            else:
                # Convert other types (int, float, bool, None) to strings
                formatted_metadata[key] = str(value) if value is not None else ""
        
        print(f"Formatted metadata keys: {list(formatted_metadata.keys())}")
        print(f"Formatted metadata sample: {json.dumps(dict(list(formatted_metadata.items())[:3]), indent=2, ensure_ascii=False)}")
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensors = {key: f.get_tensor(key) for key in f.keys()}

        file_dir = os.path.dirname(os.path.abspath(file_path)) or "."
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(file_path)}.",
            suffix=".tmp",
            dir=file_dir
        )
        os.close(fd)
        
        try:
            save_file(tensors, temp_path, metadata=formatted_metadata)
            os.replace(temp_path, file_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        
        return True
        
    except Exception as e:
        print(f"Error updating metadata in {file_path}: {e}")
        import traceback
        traceback.print_exc()
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
        
        # Calculate hashes for CivitAI lookup
        print("Calculating file hashes for CivitAI lookup...")
        autov2_hash = calculate_file_hash_autov2(file_path)
        autov3_hash = calculate_file_hash_autov3(file_path)
        
        result['hashes'] = {
            'AutoV2': autov2_hash,
            'AutoV3': autov3_hash
        }
        
        print(f"Hashes calculated - AutoV2: {autov2_hash}, AutoV3: {autov3_hash}")
        
        if autov3_hash:
            print("Looking up CivitAI description for cache...")
            result['html_description_cache'] = fetch_and_save_civitai_description(
                autov2_hash, autov3_hash
            )
        
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
        print(f"Received metadata keys: {list(new_metadata.keys())}")
        print(f"Received metadata: {json.dumps(new_metadata, indent=2, ensure_ascii=False)}")
        
        # Validate that metadata is not empty
        if not new_metadata:
            return jsonify({'error': 'Metadata cannot be empty'}), 400
        
        # Update the metadata in the file
        success = update_safetensors_metadata(file_path, new_metadata)
        
        if not success:
            return jsonify({'error': 'Failed to update metadata'}), 500
        
        print(f"Metadata update successful, reading back...")
        
        # Return the updated metadata
        result = extract_safetensors_metadata(file_path)
        if result is None:
            return jsonify({'error': 'Failed to read updated metadata'}), 500
        
        print(f"Read back metadata keys: {list(result.get('metadata', {}).keys())}")
        
        # Add file info
        file_info = get_file_info(file_path)
        result['file_info'] = file_info
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error updating metadata in {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error updating metadata: {str(e)}'}), 500

@app.route('/api/save-html-description', methods=['POST'])
def save_html_description_api():
    """Save a CivitAI description HTML file keyed by AutoV3 hash."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        autov3_hash = data.get('autov3')
        html_content = data.get('html')
        if not autov3_hash or not html_content:
            return jsonify({'error': 'autov3 and html are required'}), 400
        status = save_html_description(autov3_hash, html_content)
        if status == 'saved':
            return jsonify({'saved': True, 'status': status, 'filename': f'{autov3_hash}.html'})
        return jsonify({'saved': False, 'status': status}), 200
    except Exception as e:
        print(f"Error saving HTML description: {e}")
        return jsonify({'error': f'Error saving HTML description: {str(e)}'}), 500

@app.route('/html_description/<path:filename>')
def serve_html_description(filename):
    """Serve HTML description files from html_description directory"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        html_desc_dir = os.path.join(script_dir, 'html_description')
        file_path = os.path.join(html_desc_dir, filename)
        file_path = os.path.abspath(file_path)
        html_desc_dir_abs = os.path.abspath(html_desc_dir)
        
        # Security check: ensure the file is within the html_description directory
        if not file_path.startswith(html_desc_dir_abs):
            return jsonify({'error': 'Access denied'}), 403
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Only serve .html files
        if not filename.lower().endswith('.html'):
            return jsonify({'error': 'Only HTML files are allowed'}), 400
        
        # Read and return the HTML file
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return Response(html_content, mimetype='text/html')
    
    except Exception as e:
        print(f"Error serving HTML description {filename}: {str(e)}")
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
    parser.add_argument(
        '--middleware',
        metavar='PREFIX',
        default=None,
        help='URL path prefix for reverse-proxy mounting (e.g. lora -> /lora)'
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
    
    if args.middleware:
        prefix = "/" + args.middleware.strip("/")
        app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix)

    print(f"Starting LoRA Metadata Viewer Server")
    print(f"Files directory: {FILES_DIR}")
    base_url = f"http://{args.host}:{args.port}"
    if args.middleware:
        base_url += "/" + args.middleware.strip("/")
    print(f"Server will be available at: {base_url}")
    
    # Count files to serve
    file_count = 0
    for root, dirs, files in os.walk(FILES_DIR):
        file_count += sum(1 for f in files if f.lower().endswith(('.safetensors', '.gguf')))
    print(f"Found {file_count} compatible files")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0

if __name__ == '__main__':
    exit(main())
