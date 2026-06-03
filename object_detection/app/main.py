from .services.cv_runner import CVRunner
from .utils.logger import app_logger

def main():
    try:
        app_logger.info("Starting AI Proctoring System")
        runner = CVRunner()
        runner.start()
    except Exception as e:
        app_logger.exception(f"Application failed: {e}")
