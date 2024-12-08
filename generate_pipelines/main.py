from tools.llama.generate_class import LlamaSemanticTokensGenerator
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
import warnings
from input import Input

max_sem_input_count = 1000

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
@click.option(
    "--prompt-name",
    type=str,
    default="generic_higher_pitch_male"
)
@click.option(
    "--input-file",
    type=str,
    default="input.txt"
)
def main(no_audio, no_semantic_tokens, prompt_name, input_file):
    pipeline_states = None
    try:
        generator = TTSGenerator(
            prompt_text=[
                Path(path.join(constants.base_dir,
                     f"{prompt_name}.txt")).read_text()
            ],
            prompt_tokens=[
                Path(path.join(constants.base_dir, f"{prompt_name}.npy"))
            ],
            no_audio=no_audio,
            no_semantic_tokens=no_semantic_tokens
        )
        input = Input(input_file)
        pipeline_states = PipelineStates(input_hash=input.input_hash)
        current_time_per_line = 0

        next_sem_tokens_input_lines = []
        try:
            for index, line in enumerate(input.input_lines):
                tmp_next_input_lines = next_sem_tokens_input_lines + \
                    [(index, line)]

                if len("\n".join([line for (_, line) in tmp_next_input_lines])) > max_sem_input_count:
                    t0 = time.perf_counter()
                    generator.generate(
                        input_hash=input.input_hash,
                        input_lines=next_sem_tokens_input_lines
                    )
                    time_per_line = (
                        current_time_per_line + (time.perf_counter() - t0) / len(next_sem_tokens_input_lines)) / (1 if current_time_per_line == 0 else 2)
                    finish_time = datetime.fromtimestamp(
                        datetime.now().timestamp() + (len(input.input_lines) - index - 1) * time_per_line
                    ).strftime('%y-%m-%d %H:%M:%S')
                    logger.info(f"Estimated finish time: {finish_time}")
                    current_time_per_line = time_per_line

                    pipeline_states.save_processed_lines(
                        [pi for (pi, _) in next_sem_tokens_input_lines])

                    pipeline_states.save()

                    next_sem_tokens_input_lines = [(index, line)]
                else:
                    next_sem_tokens_input_lines = tmp_next_input_lines
        except Exception as e:
            logger.error(f"Unable to generate: {e}")
        finally:
            pipeline_states.save()
    except KeyboardInterrupt:
        if pipeline_states is not None:
            pipeline_states.save()
        sys.exit(130)


if __name__ == "__main__":
    main()
