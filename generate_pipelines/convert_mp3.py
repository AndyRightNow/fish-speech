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


def convert(wav_file_path, output_file_path, index, segment, tags):
    AudioSegment.from_wav(
        wav_file_path
    ).export(output_file_path, format="mp3", parameters=["-q:a", "0", "-write_xing", "0"], tags=tags)

    return (index, segment)


@logger.catch
def convert_mp3(prompt_name, input_name, force_segment_index, max_sem_input_count, start_segment_index):
    input = Input(f"{input_name}.txt",
                  max_sem_input_count=max_sem_input_count, prompt_name=prompt_name)
    input_meta = json.loads(Path(path.join(
        constants.base_input_dir, 'meta', f"{input_name}.json")).read_text(encoding='utf-8'))
    pipeline_states = PipelineStates(input.input_hash)
    with shelve.open(path.join(constants.states_dir, f"convert_states_{input.input_hash}")) as convert_states:
        current_input_mp3_output_dir = path.join(
            mp3_output_dir, input.input_hash)

        Path(current_input_mp3_output_dir).mkdir(parents=True, exist_ok=True)

        processed_segments = pipeline_states.get_processed_segments()

        if not len(processed_segments):
            return

        start_segment = next(
            (x for x in processed_segments if x[0] == start_segment_index), processed_segments[0])

        current_processed_segments = processed_segments[processed_segments.index(
            start_segment):]

        convert_states['converted_segments'] = [
        ] if 'converted_segments' not in convert_states else convert_states['converted_segments']

        with Pool(initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_IGN)) as pool:
            async_results = {}
            queued_count = 0

            try:
                for index, segment in current_processed_segments:
                    if segment in convert_states['converted_segments'] and index not in force_segment_index:
                        continue

                    output_mp3_name = path.join(
                        current_input_mp3_output_dir, f"{index + 1}.mp3")

                    input_wav_name = path.join(
                        constants.audio_output_dir,
                        input.input_hash,
                        f"{get_intermediate_output_base_name(segment[0], segment[1])}.wav"
                    )

                    if not Path(input_wav_name).exists():
                        continue

                    def callback(result):
                        (finished_index, finished_segment) = result
                        async_results[finished_index] = True

                        convert_states['converted_segments'] += [finished_segment]

                    def error_callback(e):
                        async_results[finished_index] = True
                        logger.exception(f"Unable to convert to mp3: {e}")

                    pool.apply_async(convert, (
                        input_wav_name,
                        output_mp3_name,
                        index,
                        segment,
                        input_meta['tags']
                    ), callback=callback, error_callback=error_callback)

                    queued_count += 1

                while len(async_results.keys()) != queued_count:
                    logger.debug(
                        f"Progress: {len(async_results.keys())}/{queued_count}")
                    time.sleep(1)

            except KeyboardInterrupt:
                pool.close()


@click.command()
@use_shared_command_options
def main(**kwargs):
    convert_mp3(**kwargs)


if __name__ == "__main__":
    main()
