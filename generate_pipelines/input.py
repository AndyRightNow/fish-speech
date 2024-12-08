from loguru import logger
from os import path
from hashlib import sha256
import constants


class Input:
    name = ''

    input_hash = ''

    input_lines = []

    def __init__(self, name):
        logger.info("Reading input file")
        self.name = name
        with open(path.join(constants.base_input_dir, name), "r", encoding="utf-8") as input_file:
            file_content = input_file.read()
            m = sha256()
            self.input_lines = file_content.splitlines()
            m.update(str.encode(file_content))
            self.input_hash = m.hexdigest()
