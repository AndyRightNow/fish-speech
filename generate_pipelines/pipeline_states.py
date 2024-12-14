from typing import Any
from constants import states_dir, base_dir
from os import path
import json
from utils import generate_pipelines_logger as logger

pipeline_states_json_path = "states.json"


class PipelineStates:
    __states: Any = {}
    __input_hash = ''

    def __init__(self, input_hash):
        logger.debug("Loading pipeline states")
        with open(path.join(states_dir, pipeline_states_json_path), 'r', encoding='utf-8') as pipeline_states_json_file:
            try:
                self.__states = json.loads(pipeline_states_json_file.read())
            except json.decoder.JSONDecodeError as e:
                logger.error(f"Unable to parse pipeline states JSON: {e}")
                self.__states = {}

        if input_hash not in self.__states:
            self.__states[input_hash] = {}

        self.__input_hash = input_hash

        self.__normalize_processed_info()

    def __normalize_processed_info(self):
        logger.debug("Normalizing pipeline states")

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
            logger.debug("Updating pipeline states")
            try:
                stringified_json = json.dumps(self.__states)

                if stringified_json:
                    pipeline_states_json_file.write(stringified_json)
                else:
                    raise Exception("Empty states")
            except Exception as e:
                logger.exception(f"Unable to save pipeline states: {e}")

    def save_processed_segment(self, input_segment, segment_index):
        self.__states[self.__input_hash]['processed_segments'] = ([] if 'processed_segments' not in self.__states[self.__input_hash] else self.__states[self.__input_hash]['processed_segments']) + [[
            segment_index, [input_segment[0][0], input_segment[-1][0]]
        ]]

    def get_processed_segments(self):
        return self.__states.get(self.__input_hash).get('processed_segments')

    def is_segment_processed(self, segment_index):
        return 'processed_segments' in self.__states[self.__input_hash] and True in (seg_i == segment_index for (seg_i, _) in self.__states[self.__input_hash]
                                                                                     ['processed_segments'])
