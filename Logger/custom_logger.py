import logging
import os
from datetime import datetime
import structlog

class CustomLogger:
    def __init__(self, log_dir="logs"):
        #Ensure logs directory exists
        self.log_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        # Timestamped log file (for persistence)
        log_file = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.log_dir, log_file)

    def get_logger(self, name=__file__):
        return structlog.get_logger(os.path.basename(name))
    
if __name__ == "__main__":
    logger = CustomLogger().get_logger(__file__)
    logger.info("This is an info message", user_id=12345, filename="report.pdf")
    logger.error("This is an error message", error="File not found", user_id=12345)
