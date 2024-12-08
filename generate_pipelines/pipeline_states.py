from typing import Any
from loguru import logger
from constants import base_dir
from os import path
import json

pipeline_states_json_path = "states.json"


class PipelineStates:
    __states: Any = {}
    __input_hash = ''

    def __init__(self, input_hash):
        with open(path.join(base_dir, pipeline_states_json_path), 'r', encoding='utf-8') as pipeline_states_json_file:
            try:
                self.__states = json.loads(pipeline_states_json_file.read())
            except json.decoder.JSONDecodeError:
                self.__states = {}

        if input_hash not in self.__states:
            self.__states[input_hash] = {}

        self.__input_hash = input_hash

    def save(self):
        with open(path.join(base_dir, pipeline_states_json_path), 'w', encoding='utf-8') as pipeline_states_json_file:
            logger.info("Updating pipeline states")
            pipeline_states_json_file.write(json.dumps(self.__states))

    def save_processed_lines(self, indices):
        self.__states[self.__input_hash]['processed_lines'] = (
            [] if 'processed_lines' not in self.__states[self.__input_hash] else self.__states[self.__input_hash]['processed_lines']) + indices

        self.__states[self.__input_hash]['processed_segments'] = ([] if 'processed_segments' not in self.__states[self.__input_hash] else self.__states[self.__input_hash]['processed_segments']) + [[
            indices[0], indices[-1]
        ]]

    def is_line_processed(self, index):
        return 'processed_lines' in self.__states[self.__input_hash] and index in self.__states[self.__input_hash]['processed_lines']

    def get_processed_segments(self):
        return self.__states.get(self.__input_hash).get('processed_segments')
