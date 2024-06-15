# config.py

class Config:
    """
    Singleton class to handle configuration settings.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.init_config()
        return cls._instance

    def init_config(self):
        """
        Initialize configuration settings.
        """
        self.operational_file_path = "data/netflix_operational_metrics.csv"
        self.business_file_path = "data/netflix_business_metrics.csv"
        self.output_dir = "output"


# Usage
config = Config()
