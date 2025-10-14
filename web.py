import os
import sys
import subprocess
import signal
import click
from pathlib import Path
import importlib.resources as pkg_resources

PID_FILE = os.path.expanduser("~/.torlite_web.pid")
PORT = 8000

def get_default_www():
    return str(pkg_resources.files("torlite") / "www")

DEFAULT_DIR = get_default_www()

def _get_pid():
    if not os.path.exists(PID_FILE):
        return None
    with open(PID_FILE) as f:
        return int(f.read().strip())

def _write_pid(pid):
    with open(PID_FILE, "w") as f:
        f.write(str(pid))

def _clear_pid():
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)

@click.group()
def web():
    """Manage TorLite web server"""
    pass

@web.command()
@click.option(
    "--dir", "directory",
    default=None,
    help="Directory to serve (defaults to torlite/www inside package)"
)
def start(directory):
    """Start the TorLite Flask web server"""
    if _get_pid():
        click.echo("🚫 Server already running.")
        return

    if directory is None:
        directory = DEFAULT_DIR

    if not os.path.exists(directory):
        click.echo(f"⚠️ Directory {directory} not found.")
        return

    env = os.environ.copy()
    env["FLASK_APP"] = "torlite.flask_server"
    env["FLASK_RUN_PORT"] = str(PORT)
    env["FLASK_ENV"] = "development"  # or production
    env["TORLITE_WWW"] = directory

    proc = subprocess.Popen(
        [sys.executable, "-m", "torlite.flask_server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        env=env
    )

    _write_pid(proc.pid)
    click.echo(f"✅ Flask serving {directory} at http://localhost:{PORT} (PID {proc.pid})")

@web.command()
def stop():
    """Stop the TorLite web server"""
    pid = _get_pid()
    if not pid:
        click.echo("🚫 No server running.")
        return
    try:
        os.killpg(pid, signal.SIGTERM)
        click.echo(f"🛑 Stopped server (PID {pid})")
    except Exception as e:
        click.echo(f"⚠️ Error stopping server: {e}")
    _clear_pid()

@web.command()
def status():
    """Check TorLite web server status"""
    pid = _get_pid()
    if not pid:
        click.echo("ℹ️ Server not running.")
        return
    try:
        os.kill(pid, 0)
        click.echo(f"✅ Server is running (PID {pid}) at http://localhost:{PORT}")
    except ProcessLookupError:
        click.echo("⚠️ PID file exists but process is not alive.")
        _clear_pid()

@web.command()
@click.option(
    "--dir", "directory",
    default=DEFAULT_DIR,
    show_default=True,
    help="Directory to serve when restarting"
)
def restart(directory):
    """Restart the TorLite web server"""
    stop()
    start.main([f"--dir={directory}"], standalone_mode=False)

