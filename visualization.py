# noinspection PyPep8Naming
import tensorboardX as tensorboard
import logging


class Reporting:
    def __init__(self):
        self._global_summary_writer = None
        self._global_logger = None

    @property
    def global_summary_writer(self) -> tensorboard.SummaryWriter:
        if self._global_summary_writer is None:
            self._global_summary_writer = tensorboard.SummaryWriter()
        return self._global_summary_writer

    @property
    def global_logger(self) -> logging.Logger:
        if self._global_logger is None:
            self._global_logger = logging.getLogger("imitation_logger")
            self._global_logger.setLevel(logging.DEBUG)
        return self._global_logger

    def initialize(self, run_name: str, logging_path: str):
        file_handler = logging.FileHandler(f'{logging_path}/{run_name}.log')
        file_handler.setLevel(logging.DEBUG)
        self.global_logger.addHandler(file_handler)

        self._global_summary_writer = tensorboard.SummaryWriter(log_dir=f"runs/{run_name}")


reporting = Reporting()

# global_summary_writer = tensorboard.SummaryWriter()
# global_logger = logging.getLogger("imitation_learning")
# global_logger.setLevel(logging.DEBUG)

