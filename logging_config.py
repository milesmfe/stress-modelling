from datetime import datetime
import logging
import os


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)-15s %(message)s')

os.makedirs('.logs', exist_ok=True)
log_filename = datetime.now().strftime('.logs/pipeline_%Y%m%d_%H%M%S.log')
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)