from logging.handlers import RotatingFileHandler

for h in list(logger.handlers):
    print(type(h), getattr(h, 'baseFilename', '?'))
