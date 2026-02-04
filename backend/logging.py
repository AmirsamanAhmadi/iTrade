import logging
import logging.config
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'logging.yaml'

def configure_logging():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'rt') as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
