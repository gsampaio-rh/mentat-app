import os
import logging


class Config:
    """
    Configuration class to hold all the configurations for the project.
    """

    # Directory settings
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(BASE_DIR)  # Parent directory of the config directory
    DATA_DIR = os.path.join(PROJECT_DIR, "data")
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

    # File paths
    OPERATIONAL_FILE = os.path.join(DATA_DIR, "netflix_operational_metrics.csv")
    BUSINESS_FILE = os.path.join(DATA_DIR, "netflix_business_metrics.csv")

    # Logging settings
    LOG_FILE = os.path.join(PROJECT_DIR, "app.log")

    @staticmethod
    def setup_logging():
        """
        Setup logging configuration.
        """
        logging.basicConfig(
            filename=Config.LOG_FILE,
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.getLogger().addHandler(logging.StreamHandler())

    @staticmethod
    def ensure_output_dir_exists():
        """
        Ensure that the output directory exists. Create it if it doesn't.
        """
        output_dir = Config.OUTPUT_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir
