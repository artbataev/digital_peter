import logging
from pathlib import Path
from typing import Optional, Union


def setup_logger(exp_dir: Optional[Union[str, Path]] = None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    c_handler = logging.StreamHandler()
    c_format = logging.Formatter("%(name)s:%(lineno)d - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    if exp_dir is not None:
        exp_dir = Path(exp_dir)
        f_handler = logging.FileHandler(exp_dir / "learning.log", mode="w")
        f_format = logging.Formatter("%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s")
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)
