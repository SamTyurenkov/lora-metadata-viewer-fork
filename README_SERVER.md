# LoRA Metadata Viewer - Server Edition

This is a Flask-based server version of the LoRA Metadata Viewer that can serve and display safetensors files from a directory on your server.

## Features

- **Server-hosted files**: Browse and view safetensors files from a server directory
- **File listing**: Automatically lists all `.safetensors` and `.gguf` files in the specified directory and subdirectories
- **Original functionality**: All original metadata viewing features are preserved
- **Dual mode**: Works both as a server (for browsing server files) and locally (drag & drop still works)

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your files**: Place your `.safetensors` files in a directory on your server

## Usage

### Basic Usage

Run the server with a directory containing your safetensors files:

```bash
python app.py --directory /path/to/your/safetensors/files
```

Then open your browser to: `http://127.0.0.1:5000`

### Advanced Usage

```bash
python app.py --directory /path/to/files --host 0.0.0.0 --port 8080
```

**Command line options**:
- `--directory, -d`: **Required** - Directory containing safetensors files to serve
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port, -p`: Port to bind to (default: 5000)
- `--debug`: Enable debug mode

### Examples

**Local development**:
```bash
python app.py -d ./my_models
```

**Network accessible**:
```bash
python app.py -d /home/user/models --host 0.0.0.0 --port 8080
```

**Debug mode**:
```bash
python app.py -d ./models --debug
```

## Web Interface

When you open the web interface, you'll see:

1. **Available Files panel**: Lists all compatible files from your server directory
   - Click on any file to load and view its metadata
   - Shows file sizes and relative paths
   - Refresh button to reload the file list

2. **Original functionality**: All existing features work as before
   - Drag & drop local files still works
   - All metadata viewing, editing, and export features
   - CivitAI integration
   - Tag frequency analysis
   - Custom templates

## File Organization

The server will recursively scan the specified directory for compatible files:
- `.safetensors` files (LoRA models)
- `.gguf` files (GGUF models)

Files in subdirectories will be shown with their relative paths.

## Security Notes

- The server only serves files from the specified directory
- Path traversal attacks are prevented
- CORS is enabled for the API endpoints
- Files are served as application/octet-stream

## API Endpoints

The server provides these API endpoints:

- `GET /`: Main web interface
- `GET /api/files`: List all available files
- `GET /api/file/<path>`: Download a specific file
- `GET /api/info`: Server information

## Troubleshooting

**"Directory does not exist"**: Make sure the path you specified exists and is accessible

**"No compatible files found"**: The directory doesn't contain any `.safetensors` or `.gguf` files

**CORS errors**: Make sure you're accessing the server through the correct URL (don't mix http/https or different ports)

**Large files**: Very large files (>2GB) may take longer to load and process

## Original Features

All original LoRA Metadata Viewer features are preserved:
- Metadata display and editing
- CivitAI integration and lookups
- Tag frequency analysis
- Custom field calculations
- Summary templates
- Export functionality

The server version simply adds the ability to browse and load files from a server directory instead of only using local files.
