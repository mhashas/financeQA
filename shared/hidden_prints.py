import logging
import os
import sys


class HiddenPrints:
    def __enter__(self):
        # Redirect stdout to suppress prints
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

        # Suppress logging
        self._original_logging_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore stdout
        sys.stdout.close()
        sys.stdout = self._original_stdout

        # Restore logging
        logging.disable(self._original_logging_level)
