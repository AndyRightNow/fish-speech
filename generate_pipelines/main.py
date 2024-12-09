from utils import use_shared_command_options
from generator import TTSGenerator
from pathlib import Path
from loguru import logger
import time
import click
from datetime import datetime
from pipeline_states import PipelineStates
import sys
import constants
from os import path
from traceback import print_exception
import warnings
from input import Input

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


@click.command()
@click.option(
    "--no-audio",
    type=bool,
    default=False
)
@click.option(
    "--no-semantic-tokens",
    type=bool,
    default=False
)
@use_shared_command_options
def main(no_audio, no_semantic_tokens, prompt_name, input_name, force_segment_index, max_sem_input_count, start_segment_index):
    pipeline_states = None
    try:
        generator = TTSGenerator(
            prompt_text=[
                Path(path.join(constants.base_dir,
                     f"{prompt_name}.txt")).read_text(encoding='utf-8')
            ],
            prompt_tokens=[
                Path(path.join(constants.base_dir, f"{prompt_name}.npy"))
            ],
            no_audio=no_audio,
            no_semantic_tokens=no_semantic_tokens
        )
        input = Input(
            f"{input_name}.txt", max_sem_input_count=max_sem_input_count, prompt_name=prompt_name)
        pipeline_states = PipelineStates(input_hash=input.input_hash)
        current_time_per_line = 0

        try:
            logger.info(f"Start generating from segment index {start_segment_index}")
            for segment_index, segment in input.input_segments[start_segment_index:]:
                t0 = time.perf_counter()
                generator.generate(
                    input_hash=input.input_hash,
                    input_lines=segment,
                    no_audio=pipeline_states.is_segment_processed(
                        segment, segment_index),
                    force=segment_index in force_segment_index
                )
                time_per_line = (
                    current_time_per_line + (time.perf_counter() - t0) / len(segment)) / (1 if current_time_per_line == 0 else 2)
                finish_time = datetime.fromtimestamp(
                    datetime.now().timestamp() + len(segment) * time_per_line
                ).strftime('%y-%m-%d %H:%M:%S')
                logger.info(f"Estimated finish time: {finish_time}")
                current_time_per_line = time_per_line

                pipeline_states.save_processed_segment(segment, segment_index)

                pipeline_states.save()
        except Exception as e:
            logger.error(f"Unable to generate: {e}")
            print_exception(e)
        finally:
            pipeline_states.save()
    except KeyboardInterrupt:
        if pipeline_states is not None:
            pipeline_states.save()
        sys.exit(130)


if __name__ == "__main__":
    main()
