from loguru import logger
from pydub import AudioSegment
from os import mkdir
import constants
from os import path
import time
import shelve
from multiprocessing import Pool
from pipeline_states import PipelineStates
import signal
from input import Input

mp3_output_dir = path.join(constants.base_output_dir, "mp3")


def convert(wav_file_path, output_file_path, index, segment):
    AudioSegment.from_wav(
        wav_file_path
    ).export(output_file_path, format="mp3", parameters=["-q:a", "0"], tags={
        "artist": "稚楚",
        "album": "恒星时刻"
    })

    return (index, segment)


def main():
    input = Input("input.txt")
    pipeline_states = PipelineStates(input.input_hash)
    with shelve.open(path.join(constants.states_dir, f"convert_states_{input.input_hash}")) as convert_states:
        current_input_mp3_output_dir = path.join(
            mp3_output_dir, input.input_hash)

        try:
            mkdir(current_input_mp3_output_dir)
        except FileExistsError:
            pass

        try:
            convert_states['converted_segments'] = [
            ] if 'converted_segments' not in convert_states else convert_states['converted_segments']

            with Pool(initializer=signal.signal, initargs=(signal.SIGINT, signal.SIG_IGN)) as pool:
                async_results = {}
                queued_count = 0
                processed_segments = pipeline_states.get_processed_segments()

                try:
                    for index, segment in enumerate(processed_segments):
                        if segment in convert_states['converted_segments']:
                            continue

                        output_base_name = f"{input.input_hash}_{segment[0]}_{segment[1]}"

                        output_mp3_name = path.join(
                            current_input_mp3_output_dir, f"{index + 1}.mp3")

                        def callback(result):
                            (finished_index, finished_segment) = result
                            async_results[finished_index] = True

                            convert_states['converted_segments'] += [finished_segment]

                        def error_callback(e):
                            raise e

                        pool.apply_async(convert, (path.join(
                            constants.audio_output_dir,
                            f"{output_base_name}.wav"
                        ), output_mp3_name, index, segment), callback=callback, error_callback=error_callback)

                        queued_count += 1

                    while len(async_results.keys()) != queued_count:
                        logger.info(
                            f"Progress: {len(async_results.keys())}/{queued_count}")
                        time.sleep(1)

                except KeyboardInterrupt:
                    pool.close()
        except Exception as e:
            logger.error(f"Unable to convert mp3: {e}")


if __name__ == "__main__":
    main()
