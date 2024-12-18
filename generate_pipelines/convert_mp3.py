from pydub import AudioSegment
from os import mkdir
import constants
import click
from os import path
import time
from pathlib import Path
import shelve
import json
from multiprocessing import Pool
from pipeline_states import PipelineStates
import signal
from input import Input
from utils import get_intermediate_output_base_name, use_shared_command_options, generate_pipelines_logger as logger

mp3_output_dir = path.join(constants.base_output_dir, "mp3")


def convert(wav_file_path, output_file_path, tags):
    try:
        AudioSegment.from_wav(
            wav_file_path
        ).export(output_file_path, format="mp3", parameters=["-q:a", "0", "-write_xing", "0"], tags=tags)
    except Exception as e:
        raise e


@logger.catch
def convert_mp3_async(input_hash, wav_file_path, segment_index):
    current_input_mp3_output_dir = path.join(
        mp3_output_dir, input_hash)

    Path(current_input_mp3_output_dir).mkdir(parents=True, exist_ok=True)

    with Pool(initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_IGN)) as pool:
        try:
            output_mp3_name = path.join(
                current_input_mp3_output_dir, f"{segment_index + 1}.mp3")

            if not Path(wav_file_path).exists():
                logger.info(
                    f"Segment {segment_index} has no input wav file, skipped.")
                return

            def callback(finished_index):
                logger.success(
                    f"Generated segment {segment_index} to mp3 file.")

            def error_callback(e):
                logger.exception(f"Unable to convert to mp3: {e}")

            logger.info(f"Queue segment {segment_index} for generation.")
            pool.apply_async(convert, (
                input_wav_name,
                output_mp3_name,
                {
                    'title': segment_index
                }
            ), callback=callback, error_callback=error_callback)

        except KeyboardInterrupt:
            pool.close()


@click.command()
@use_shared_command_options
def main(**kwargs):
    convert_mp3(**kwargs)


if __name__ == "__main__":
    main()
