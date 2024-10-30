from RAB.app import create_app
from RAB.celery_worker import celery
import subprocess
import atexit
import socket
import threading

app = create_app()

def is_redis_running(host='127.0.0.1', port=6379):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

redis_process = None
celery_process = None

def terminate_processes():
    if redis_process:
        print("Terminating Redis server...")
        redis_process.terminate()
        try:
            redis_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            redis_process.kill()
    
    if celery_process:
        print("Terminating Celery worker...")
        celery_process.terminate()
        try:
            celery_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            celery_process.kill()

def start_celery_worker():
    global celery_process
    print("Starting Celery worker...")
    celery_process = subprocess.Popen(['celery', '-A', 'RAB.celery_worker.celery', 'worker', '--loglevel=info'])

if __name__ == '__main__':
    try:
        # Check and start Redis if needed
        if not is_redis_running():
            print("Starting Redis server...")
            redis_process = subprocess.Popen(['redis-server'])
        else:
            print("Redis is already running.")

        # Start Celery worker in a separate thread
        celery_thread = threading.Thread(target=start_celery_worker)
        celery_thread.daemon = True
        celery_thread.start()

        # Register cleanup function
        atexit.register(terminate_processes)

        print("Starting Flask server on http://localhost:1111")
        app.run(
            host='127.0.0.1',  # Only listen on localhost
            port=1111,
            debug=True
        )

    except Exception as e:
        print(f"Failed to start server: {e}")
        terminate_processes()