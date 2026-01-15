import logging
import sys
from contextlib import contextmanager

# Ensure LOGS_PATH is defined in config.py
from utils.config import LOGS_PATH


class MedicalLogger:
    """
    Custom Logger wrapper to handle indentation, colors, and specific
    data science formatting (dataframes, substeps).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MedicalLogger, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.logger = logging.getLogger("MedicalPipeline")
        self.logger.setLevel(logging.INFO)
        self.indent_level = 0

        # Avoid adding handlers multiple times
        if not self.logger.handlers:
            # 1. Console Handler (Standard output)
            c_handler = logging.StreamHandler(sys.stdout)
            c_format = logging.Formatter("%(message)s")
            c_handler.setFormatter(c_format)
            self.logger.addHandler(c_handler)

            # 2. File Handler (Log file)
            LOGS_PATH.mkdir(parents=True, exist_ok=True)
            f_handler = logging.FileHandler(LOGS_PATH / "pipeline.log", mode="w")
            f_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            f_handler.setFormatter(f_format)
            self.logger.addHandler(f_handler)

    def _format_msg(self, msg, color_code=None):
        indent_str = "  " * self.indent_level
        # Only color if output is a terminal
        if color_code and sys.stdout.isatty():
            return f"{indent_str}\033[{color_code}m{msg}\033[0m"
        return f"{indent_str}{msg}"

    def info(self, msg):
        """Standard info message."""
        self.logger.info(self._format_msg(msg))

    def warning(self, msg):
        """Warning message (Yellow)."""
        # 93m is bright yellow
        self.logger.warning(self._format_msg(f"[WARNING] {msg}", "93"))

    def error(self, msg):
        """Error message (Red)."""
        # 91m is bright red
        self.logger.error(self._format_msg(f"[ERROR] {msg}", "91"))

    def success(self, msg):
        """Success message (Green)."""
        # 92m is bright green
        self.logger.info(self._format_msg(f"✔ {msg}", "92"))

    def substep(self, msg):
        """Highlights a major step in the process (Blue/Bold)."""
        # 94m is bright blue
        self.logger.info("")  # Empty line before
        self.logger.info(self._format_msg(f"➜ {msg}", "94"))

    @contextmanager
    def indent(self):
        """Context manager to indent logs inside a block."""
        self.indent_level += 1
        try:
            yield
        finally:
            self.indent_level -= 1

    def dataframe_info(self, df, name="DataFrame"):
        """Specific helper to log DataFrame stats nicely."""
        rows, cols = df.shape
        missing = df.isnull().sum().sum()

        # Memory usage in MB
        mem = df.memory_usage(deep=True).sum() / 1024**2

        info_str = (
            f"[{name}] Shape: ({rows}, {cols}) | "
            f"Missing: {missing} | "
            f"Mem: {mem:.2f} MB"
        )
        self.info(info_str)


def get_logger():
    """Factory method to get the singleton logger."""
    return MedicalLogger()
