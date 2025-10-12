#!/usr/bin/env python3
"""
Universal Documentation Builder for PRJ381 Data Preprocessing API
================================================================

A cross-platform documentation builder that works on Windows, macOS, and Linux.
Builds Sphinx documentation and checks the FastAPI server status.

Usage:
    python build_docs.py [options]

Options:
    --serve         Start a local server to view documentation
    --port PORT     Port for local server (default: 8080)
    --no-sphinx     Skip Sphinx documentation build
    --open          Open documentation in browser
    --clean         Clean build directory before building
    --help          Show this help message

Examples:
    python build_docs.py                    # Build docs only
    python build_docs.py --serve --open     # Build, serve, and open in browser
    python build_docs.py --clean --serve    # Clean build, then serve
"""

import os
import sys
import subprocess
import argparse
import webbrowser
import time
import shutil
import platform
from pathlib import Path
import http.server
import socketserver
from threading import Thread


def get_project_paths():
    """Get universal project paths relative to script location."""
    # Script location (works whether in root or docs directory)
    script_dir = Path(__file__).parent.absolute()
    
    # Find project root (contains app directory)
    project_root = script_dir
    while project_root != project_root.parent:
        if (project_root / "app" / "main.py").exists():
            break
        project_root = project_root.parent
    else:
        raise FileNotFoundError("Could not find project root (app/main.py not found)")
    
    docs_dir = project_root / "docs"
    build_dir = docs_dir / "_build" / "html"
    
    return {
        "script_dir": script_dir,
        "project_root": project_root,
        "docs_dir": docs_dir,
        "build_dir": build_dir
    }


def run_command(command, cwd=None, check=True, capture=True):
    """Run a command with cross-platform compatibility."""
    print(f"[INFO] Running: {command}")
    print(f"[INFO] In directory: {cwd or 'current'}")
    
    try:
        # Use shell=True on Windows, False on Unix-like systems for better compatibility
        use_shell = platform.system() == "Windows"
        
        result = subprocess.run(
            command if use_shell else command.split(),
            shell=use_shell,
            cwd=cwd,
            check=check,
            capture_output=capture,
            text=True
        )
        
        if capture and result.stdout:
            print(result.stdout)
        
        return result
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error running command: {e}")
        if capture and e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            raise
        return e


def check_dependencies():
    """Check and install required dependencies."""
    print("[INFO] Checking documentation dependencies...")
    
    required_packages = [
        "sphinx",
        "sphinx-rtd-theme", 
        "sphinx-autodoc-typehints"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"[INFO] Installing missing packages: {', '.join(missing_packages)}")
        pip_command = f"{sys.executable} -m pip install {' '.join(missing_packages)}"
        run_command(pip_command)
        print("[SUCCESS] Dependencies installed successfully!")
    else:
        print("[SUCCESS] All dependencies are already installed!")


def clean_build_directory(build_dir):
    """Clean the build directory."""
    if build_dir.exists():
        print(f"[INFO] Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)
        print("[SUCCESS] Build directory cleaned!")


def build_sphinx_docs(docs_dir, build_dir):
    """Build Sphinx documentation."""
    print("=" * 60)
    print("Building Sphinx Documentation")
    print("=" * 60)
    
    # Ensure build directory parent exists
    build_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Build command
    sphinx_command = f"{sys.executable} -m sphinx -b html . {build_dir}"
    
    try:
        run_command(sphinx_command, cwd=docs_dir)
        print("[SUCCESS] Sphinx documentation built successfully!")
        print(f"[INFO] Location: {build_dir}")
        return True
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to build Sphinx documentation")
        return False


def check_fastapi_server():
    """Check if FastAPI server is running."""
    try:
        # Try importing requests, fallback to urllib if not available
        try:
            import requests
            response = requests.get("http://localhost:8000/docs", timeout=2)
            return response.status_code == 200
        except ImportError:
            import urllib.request
            import urllib.error
            try:
                urllib.request.urlopen("http://localhost:8000/docs", timeout=2)
                return True
            except urllib.error.URLError:
                return False
    except:
        return False


def start_docs_server(port, docs_dir):
    """Start a simple HTTP server for documentation."""
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(docs_dir), **kwargs)
        
        def log_message(self, format, *args):
            # Suppress server logs for cleaner output
            pass
    
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"[INFO] Documentation server running at http://localhost:{port}")
            print("[INFO] Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"[ERROR] Port {port} is already in use. Try a different port with --port")
        else:
            print(f"[ERROR] Failed to start server: {e}")
        return False
    except KeyboardInterrupt:
        print("\n[INFO] Documentation server stopped")
        return True


def open_in_browser(url, delay=2):
    """Open URL in browser after a delay."""
    def open_browser():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            print(f"[INFO] Opened {url} in browser")
        except Exception as e:
            print(f"[WARNING] Could not open browser: {e}")
    
    Thread(target=open_browser, daemon=True).start()


def print_status_summary(paths, fastapi_running, build_success):
    """Print a summary of the documentation status."""
    print("\n" + "=" * 60)
    print("Documentation Status Summary")
    print("=" * 60)
    
    # Sphinx documentation
    if build_success:
        print("[SUCCESS] Sphinx Documentation: Built successfully")
        print(f"[INFO] Location: {paths['build_dir']}")
    else:
        print("[ERROR] Sphinx Documentation: Build failed")
    
    # FastAPI documentation
    if fastapi_running:
        print("[SUCCESS] FastAPI Server: Running")
        print("[INFO] Swagger UI: http://localhost:8000/docs")
        print("[INFO] ReDoc: http://localhost:8000/redoc")
    else:
        print("[WARNING] FastAPI Server: Not running")
        print("[INFO] To start: uvicorn app.main:app --reload")
        print("[INFO] Then access:")
        print("   • Swagger UI: http://localhost:8000/docs")
        print("   • ReDoc: http://localhost:8000/redoc")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Build PRJ381 Data Preprocessing API documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_docs.py                    Build documentation only
  python build_docs.py --serve --open     Build and serve with browser
  python build_docs.py --clean --serve    Clean build then serve
  python build_docs.py --no-sphinx        Skip Sphinx, check FastAPI only
        """
    )
    
    parser.add_argument("--serve", action="store_true",
                       help="Start local server to view documentation")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port for local documentation server (default: 8080)")
    parser.add_argument("--no-sphinx", action="store_true",
                       help="Skip Sphinx documentation build")
    parser.add_argument("--open", action="store_true",
                       help="Open documentation in browser")
    parser.add_argument("--clean", action="store_true",
                       help="Clean build directory before building")
    
    args = parser.parse_args()
    
    try:
        # Get universal paths
        paths = get_project_paths()
        print(f"🏠 Project root: {paths['project_root']}")
        print(f"[INFO] Docs directory: {paths['docs_dir']}")
        
        # Check if docs directory exists
        if not paths["docs_dir"].exists():
            print(f"[ERROR] Documentation directory not found: {paths['docs_dir']}")
            return 1
        
        build_success = True
        
        # Clean build directory if requested
        if args.clean:
            clean_build_directory(paths["build_dir"])
        
        # Build Sphinx documentation
        if not args.no_sphinx:
            check_dependencies()
            build_success = build_sphinx_docs(paths["docs_dir"], paths["build_dir"])
        
        # Check FastAPI server
        print("\n" + "=" * 60)
        print("Checking FastAPI Server Status")
        print("=" * 60)
        fastapi_running = check_fastapi_server()
        
        # Print status summary
        print_status_summary(paths, fastapi_running, build_success)
        
        # Serve documentation
        if args.serve:
            if not paths["build_dir"].exists() or not any(paths["build_dir"].iterdir()):
                print(f"\n[ERROR] No documentation found in {paths['build_dir']}")
                print("[INFO] Build documentation first (without --no-sphinx)")
                return 1
            
            print(f"\n[INFO] Starting documentation server on port {args.port}")
            
            if args.open:
                open_in_browser(f"http://localhost:{args.port}")
            
            # Start server (this blocks until Ctrl+C)
            start_docs_server(args.port, paths["build_dir"])
        
        elif args.open and paths["build_dir"].exists():
            # Just open without serving
            index_file = paths["build_dir"] / "index.html"
            if index_file.exists():
                webbrowser.open(f"file://{index_file.absolute()}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Make sure you're running this script from the project directory")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Build cancelled by user")
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
