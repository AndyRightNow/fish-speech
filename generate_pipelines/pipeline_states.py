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
        logger.info("Loading pipeline states")
        with open(path.join(base_dir, pipeline_states_json_path), 'r', encoding='utf-8') as pipeline_states_json_file:
            try:
                self.__states = json.loads(pipeline_states_json_file.read())
            except json.decoder.JSONDecodeError:
                self.__states = {}

        if input_hash not in self.__states:
            self.__states[input_hash] = {}

        self.__input_hash = input_hash

        self.__normalize_processed_info()

    def __normalize_processed_info(self):
        logger.info("Normalizing pipeline states")

        self.__states[self.__input_hash]['processed_segments'] = [
        ] if 'processed_segments' not in self.__states[self.__input_hash] else self.__states[self.__input_hash]['processed_segments']

        self.__states[self.__input_hash]['processed_segments'] = sorted(
            list({
                x[0]: x for x in self.__states[self.__input_hash]['processed_segments']
            }.values()),
            key=lambda x: x[0]
        )

    def save(self):
        self.__normalize_processed_info()

        with open(path.join(base_dir, pipeline_states_json_path), 'w', encoding='utf-8') as pipeline_states_json_file:
            logger.info("Updating pipeline states")
            try:
                stringified_json = json.dumps(self.__states)

                if stringified_json:
                    pipeline_states_json_file.write(stringified_json)
            except Exception as e:
                logger.error(f"Unable to save pipeline states: {e}")

    def save_processed_segment(self, input_segment, segment_index):
        self.__states[self.__input_hash]['processed_segments'] = ([] if 'processed_segments' not in self.__states[self.__input_hash] else self.__states[self.__input_hash]['processed_segments']) + [[
            segment_index, [input_segment[0][0], input_segment[-1][0]]
        ]]

    def get_processed_segments(self):
        return self.__states.get(self.__input_hash).get('processed_segments')

    def is_segment_processed(self, input_segment_or_segment, segment_index):
        segment = input_segment_or_segment if isinstance(input_segment_or_segment[0], int) else [
            line_index for (line_index, _) in input_segment_or_segment]

        return 'processed_segments' in self.__states[self.__input_hash] and ([segment_index] + [segment]) in self.__states[self.__input_hash]['processed_segments']
