import logging


class CustomClassA:
    def __init__(self):
        logger = logging.getLogger(self.__class__.__name__)
        logger.info("Initialize a custom class A!")
