import logging


def setup_logger(log_file='app.log'):
    """
    Sets up the logger with the specified configuration.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )