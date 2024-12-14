from pydub import AudioSegment
from os import mkdir
import constants
import click
from os import path
import time
from pathlib import Path
import shelve
import json
import signal
from input import Input
from utils import get_intermediate_output_base_name, use_shared_command_options, generate_pipelines_logger as logger

mp3_output_dir = path.join(constants.base_output_dir, "mp3")


@logger.catch
def debug_print_shelve(prompt_name, input_name, force_segment_index, max_sem_input_count, start_segment_index):
    input = Input(f"{input_name}.txt",
                  max_sem_input_count=max_sem_input_count, prompt_name=prompt_name)

    with shelve.open(path.join(constants.states_dir, f"convert_states_{input.input_hash}")) as convert_states:
        logger.debug(
            f"{None if 'converted_segments' not in convert_states else convert_states['converted_segments']}")


@click.command()
@use_shared_command_options
def main(**kwargs):
    debug_print_shelve(**kwargs)


if __name__ == "__main__":
    main()
