from tools.llama.generate_class import LlamaSemanticTokensGenerator
from pathlib import Path
from loguru import logger
from os import path
import constants
import click
from pipeline_states import PipelineStates
import sys
from input import Input

max_sem_input_count = 1000
sem_tokens_output_dir = path.join(constants.base_output_dir, "semantic_tokens")


@click.command()
@click.option(
    "--dry",
    type=bool,
    default=False
)
def main(
    dry: bool
):
    pipeline_states = None
    try:
        sem_tokens_generator = LlamaSemanticTokensGenerator(
            prompt_text=[
                "人间灯火倒映湖中，她的渴望让静水泛起涟漪。若代价只是孤独，那就让这份愿望肆意流淌。流入她所注视的世间，也流入她如湖水般澄澈的目光。"],
            prompt_tokens=[Path("4_output_trained.npy")],
        )
        input = Input("input.txt")
        pipeline_states = PipelineStates(input_hash=input.input_hash)

        next_sem_tokens_input_lines = []
        try:
            for index, line in enumerate(input.input_lines):
                tmp_next_input_lines = next_sem_tokens_input_lines + \
                    [(index, line)]

                if len("\n".join([line for (_, line) in tmp_next_input_lines])) > max_sem_input_count:
                    if not dry:
                        sem_tokens_generator.generate(
                            text='\n'.join(
                                [line for (_, line) in next_sem_tokens_input_lines]),
                            output_name=path.join(
                                sem_tokens_output_dir, f"{input.input_hash}_{next_sem_tokens_input_lines[0][0]}_{next_sem_tokens_input_lines[-1][0]}"
                            )
                        )

                    pipeline_states.save_processed_lines(
                        [pi for (pi, _) in next_sem_tokens_input_lines])

                    pipeline_states.save()

                    next_sem_tokens_input_lines = [(index, line)]
                else:
                    next_sem_tokens_input_lines = tmp_next_input_lines
        except Exception as e:
            logger.error(f"Unable to generate semantic tokens: {e}")
        finally:
            pipeline_states.save()
    except KeyboardInterrupt:
        if pipeline_states is not None:
            pipeline_states.save()
        sys.exit(130)


if __name__ == "__main__":
    main()
