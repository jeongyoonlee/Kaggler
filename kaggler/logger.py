from logging import config, getLogger
import logging.config


config.fileConfig('logging.conf')
log = getLogger('console') 
