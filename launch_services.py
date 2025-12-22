import subprocess
import time
import os
import sys
import threading
import socket
import webbrowser

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def log(msg, emoji="ℹ️ ", color=RESET):
    print(f"{color}{emoji} {msg}{RESET}")

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def stream_output(process, prefix):
    for line in iter(process.stdout.readline, ''):
        pass # We might want to suppress verbose output unless debug is on to keep the terminal clean
    process.stdout.close()

def check_service(name, check_cmd):
    try:
        subprocess.run(check_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False

def start_ollama():
    log("Checking Ollama Service...", "🦙", CYAN)
    
    # Check if port 11434 is in use (Ollama default)
    if is_port_in_use(11434):
        log("Ollama is already running (Port 11434 active).", "�
", GREEN)
        return None
    
    log("Starting Ollama serve...", "🚀", YELLOW)
    try:
        # Assuming 'ollama' is in PATH. 
        proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t = threading.Thread(target=stream_output, args=(proc, "OLLAMA"))
        t.daemon = True
        t.start()
        
        # Wait for it to come up
        for _ in range(10):
            if is_port_in_use(11434):
                log("Ollama started successfully.", "�
", GREEN)
                return proc
            time.sleep(1)
            
        log("Ollama might not have started correctly, but proceeding...", "⚠️ ", YELLOW)
        return proc
    except FileNotFoundError:
        log("Error: 'ollama' command not found. Please install Ollama.", "❌", RED)
        return None

def start_mlflow():
    log("Checking MLflow UI...", "📊", CYAN)
    
    if is_port_in_use(5001):
        log("MLflow is likely already running (Port 5001 active).", "�
", GREEN)
        return None

    log("Starting MLflow UI on Port 5001...", "🚀", YELLOW)
    try:
        env = os.environ.copy()
        venv_bin = os.path.join(os.getcwd(), "venv", "bin")
        env["PATH"] = venv_bin + os.pathsep + env["PATH"]
        
        # Explicit host and port 5001
        proc = subprocess.Popen(["mlflow", "ui", "--host", "127.0.0.1", "--port", "5001"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        
        # Don't capture output to avoid spamming unless we want to debug
        # But we need to read it so buffer doesn't fill if we do capture.
        # Let's just let it run in bg.
        
        t = threading.Thread(target=stream_output, args=(proc, "MLFLOW"))
        t.daemon = True
        t.start()
        
        # Wait for port
        for _ in range(10):
            if is_port_in_use(5001):
                log("MLflow UI started at http://127.0.0.1:5001", "�
", GREEN)
                webbrowser.open("http://127.0.0.1:5001")
                return proc
            time.sleep(1)

        log("MLflow started (background).", "�
", GREEN)
        webbrowser.open("http://127.0.0.1:5001")
        return proc
    except Exception as e:
        log(f"Error starting MLflow: {e}", "❌", RED)
        return None

def setup_env():
    log("Verifying Environment...", "🛠️ ", CYAN)
    if not os.path.exists("venv"):
        log("Creating virtual environment...", "📦", YELLOW)
    else:
        log("Virtual environment found.", "�
", GREEN)

    # We reuse the shell script for consistency, but we could do it in python too.
    # subprocess.run(["bash", "setup_env.sh"], check=True) 
    # To keep it pretty, let's just assume setup_env.sh handles the pip install noise
    # or we suppress it.
    
    log("Syncing dependencies (pip)...", "🔄", YELLOW)
    try:
        subprocess.run(["bash", "setup_env.sh"], check=True) #, stdout=subprocess.DEVNULL)
        log("Environment is ready.", "✨", GREEN)
    except subprocess.CalledProcessError:
        log("Dependency installation failed!", "❌", RED)
        sys.exit(1)

def main():
    print(f"\n{BOLD}{CYAN}🌍  EVOLVING AI BIOSPHERE  🌍{RESET}\n")
    
    setup_env()
    
    ollama_proc = start_ollama()
    mlflow_proc = start_mlflow()

    # Launch services in background threads so they don't block the UI startup
    # We will just fire and forget them for now, but keep references for cleanup if strict management needed.
    # Actually, better pattern: Start them in non-blocking way, then launch game.
    
    print("\n" + "="*40 + "\n")
    log("Services are running. Keep this terminal open.", "ℹ️ ", GREEN)
    log("Run 'python3 simulation.py' in a new terminal to play.", "🎮", CYAN)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("\nInterrupted by user.", "🛑", YELLOW)
    finally:
        print("\n" + "="*40 + "\n")
        log("Shutting down services...", "🔌", CYAN)
        if ollama_proc:
            log("Stopping Ollama instance...", "🛑", YELLOW)
            ollama_proc.terminate()
        if mlflow_proc:
            log("Stopping MLflow UI...", "🛑", YELLOW)
            mlflow_proc.terminate()
        log("Goodbye!", "👋", GREEN)

if __name__ == "__main__":
    main()
