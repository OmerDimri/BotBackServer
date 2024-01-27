import logging.config
from datetime import datetime
import os
import sys

LOGFILE = '/tmp/{0}.{1}.log'.format(
    os.path.basename(__file__),
    datetime.now().strftime('%Y%m%dT%H%M%S'))

DEFAULT_LOGGING = {
    'version': 1,
    'formatters': {
        'standard': {
            'format': '%(asctime)s %(levelname)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'simple': {
            'format': '%(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'DEBUG',
            'stream': sys.stdout,
        }
    },
    'loggers': {
        __name__: {
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': False,
        }#,
    }
}

logging.basicConfig(level=logging.ERROR)
logging.config.dictConfig(DEFAULT_LOGGING)
logger = logging.getLogger(__name__)