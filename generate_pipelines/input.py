from os import path
from hashlib import sha256
import constants
from utils import generate_pipelines_logger as logger


class Input:
    name = ''

    input_hash = ''

    input_lines = []

    input_segments = []

    def __init__(self, name, max_sem_input_count, prompt_name):
        logger.info("Reading input file")
        self.name = name
        with open(path.join(constants.base_input_dir, name), "r", encoding="utf-8") as input_file:
            file_content = input_file.read()
            m = sha256()
            self.input_lines = file_content.splitlines()
            m.update(str.encode(file_content))
            m.update(str.encode(str(max_sem_input_count)))
            m.update(str.encode(prompt_name))
            self.input_hash = m.hexdigest()

        next_input_segment = []
        char_count = 0
        segment_index = 0

        for line_index, line in enumerate(self.input_lines):
            next_input_segment.append(
                [line_index, line]
            )
            char_count += len(line)

            if char_count >= max_sem_input_count or line_index == len(self.input_lines) - 1:
                self.input_segments.append([segment_index, next_input_segment])
                next_input_segment = []
                char_count = 0
                segment_index += 1

        logger.info(f"Read {segment_index} segments from {name}")
            
