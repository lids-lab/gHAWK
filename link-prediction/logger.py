import logging
import os
import os.path as osp
import logging, sys, os
from datetime import datetime


# logger.py
import logging
import sys
import os
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    """
    Configure the root logger with one console + file handler (on first call),
    then return a child logger named `name` that inherits those handlers.
    """
    root = logging.getLogger()
    if not root.handlers:
        # Only the very first call will set up handlers
        root.setLevel(logging.INFO)

        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        # Console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        root.addHandler(ch)

        # File
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logdir = os.path.join("logs", timestamp)
        os.makedirs(logdir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(logdir, "run.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        root.addHandler(fh)

        # Prevent messages from being propagated twice
        root.propagate = False

    # Return (or create) a child logger; it reuses the root handlers
    return logging.getLogger(name)

'''def get_logger(name):
    # create folder, handlers, formatters
    logger = logging.getLogger("Hybrid-GNN-PyG (Bloom+TransE+RotatE)")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_folder = os.path.join("logs", current_time)
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(log_folder, "hybrid-gnn-pyg-training.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger'''
