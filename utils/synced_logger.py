import time


class SyncedLogger:
    '''
    A class to print messages with a timestamp that is synced across multiple calls.
    
    Attributes:
    start_time (float): The time at which the logger was created.
    
    Methods:
    print_with_timestamp(message): Prints the message with a timestamp.
    reset_start_time(): Resets the start time to the current time.
    
    TODO: Implement logging to file.
    '''
    def __init__(self):
        self.start_time = time.time()
        
    def print_with_timestamp(self, message):
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time))
        print(f"[{elapsed_time}] {message}")
        
    def reset_start_time(self):
        self.start_time = time.time()
        
logger = SyncedLogger()
logger.print_with_timestamp("Logger Initialised...")