import logging

class LoggerCenter:
    _logger = None

    @staticmethod
    def get_logger(name="multi_tool"):
        if LoggerCenter._logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('multi_tool.log'),
                    logging.StreamHandler()
                ]
            )
            LoggerCenter._logger = logging.getLogger(name)
        return LoggerCenter._logger
    