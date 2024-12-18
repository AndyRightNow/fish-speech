from input import Input
import warnings
from traceback import print_exception
from os import path
from convert_mp3 import convert_mp3_async
import constants
import os
import sys
from pipeline_states import PipelineStates
import click
from loguru import logger as global_logger
from pathlib import Path
import importlib
from generator import TTSGenerator
from utils import use_shared_command_options, generate_pipelines_logger as logger
import locale

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def __get_segment_title(segment_title_language, segment_index):
    if segment_title_language == "cn":
        return f"第{segment_index + 1}段!,!,!,"
    if segment_title_language == "en":
        return f"Segment {segment_index + 1}!,!,!,"
    if segment_title_language == "ja":
        return f"第{segment_index + 1}部分!,!,!,"

    return ""


@click.command()
@click.option("--no-audio", type=bool, default=False)
@click.option("--no-semantic-tokens", type=bool, default=False)
@click.option("--insert-segment-title", type=bool, default=False)
@click.option(
    "--segment-title-language",
    type=click.types.Choice(["en", "cn", "ja"]),
    default="cn",
)
@use_shared_command_options
def main(
    no_audio,
    no_semantic_tokens,
    prompt_name,
    input_name,
    force_segment_index,
    max_sem_input_count,
    start_segment_index,
    insert_segment_title,
    segment_title_language,
):
    pipeline_states = None
    try:
        generator = TTSGenerator(
            prompt_text=[
                Path(path.join(constants.base_dir, f"{prompt_name}.txt")).read_text(
                    encoding="utf-8"
                )
            ],
            prompt_tokens=[
                Path(path.join(constants.base_dir, f"{prompt_name}.npy"))],
            no_audio=no_audio,
            no_semantic_tokens=no_semantic_tokens,
        )
        input = Input(
            f"{input_name}.txt",
            max_sem_input_count=max_sem_input_count,
            prompt_name=prompt_name,
        )
        pipeline_states = PipelineStates(input_hash=input.input_hash)

        try:
            logger.info(
                f"Start generating from segment index {start_segment_index}")

            for segment_index, segment in input.input_segments[start_segment_index:]:
                is_segment_processed = pipeline_states.is_segment_processed(
                    segment_index)
                logger.info(
                    f"The segment {segment_index} has{'' if  is_segment_processed else ' not'} been processed")

                output_wav_name = generator.generate(
                    input_hash=input.input_hash,
                    input_lines=segment,
                    no_semantic_tokens=is_segment_processed,
                    no_audio=is_segment_processed,
                    force=segment_index in force_segment_index,
                    segment_title=(
                        ""
                        if not insert_segment_title
                        else __get_segment_title(segment_title_language, segment_index)
                    ),
                )

                if output_wav_name is not None:
                    pipeline_states.save_processed_segment(
                        segment, segment_index)

                    pipeline_states.save()

                    convert_mp3_async(
                        input_hash=input.input_hash,
                        wav_file_path=output_wav_name,
                        segment_index=segment_index
                    )

        except Exception as e:
            logger.exception(f"Unable to generate: {e}")
            print_exception(e)
        finally:
            pipeline_states.save()
    except KeyboardInterrupt:
        if pipeline_states is not None:
            pipeline_states.save()
        sys.exit(130)


if __name__ == "__main__":
    main()
