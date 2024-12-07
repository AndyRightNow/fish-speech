from tools.llama.generate_class import llama_semantic_tokens_generate
from pathlib import Path
from loguru import logger
from os import path
import json
from hashlib import sha256
from .constants import base_dir
from .pipeline_states import PipelineStates
import sys

pipeline_states_json_path = "states.json"
base_output_dir = path.join(base_dir, "output")
max_sem_input_count = 1000
sem_tokens_output_dir = path.join(base_output_dir, "semantic_tokens")


if __name__ == "__main__":
    pipeline_states = None
    try:
        sem_tokens_generator = llama_semantic_tokens_generate(
            prompt_text=[
                "人间灯火倒映湖中，她的渴望让静水泛起涟漪。若代价只是孤独，那就让这份愿望肆意流淌。流入她所注视的世间，也流入她如湖水般澄澈的目光。"],
            prompt_tokens=[Path("4_output_trained.npy")],
        )

        input_lines = []
        input_hash = ''

        logger.info("Reading input file")
        with open(path.join(base_dir, "input.txt"), "r", encoding="utf-8") as input_file:
            file_content = input_file.read()
            m = sha256()
            input_lines = file_content.splitlines()
            m.update(str.encode(file_content))
            input_hash = m.hexdigest()

        pipeline_states = PipelineStates(input_hash=input_hash)

        next_sem_tokens_input_lines = []
        try:
            for index, line in enumerate(input_lines):
                if pipeline_states.is_line_processed(index):
                    continue

                tmp_next_input_lines = next_sem_tokens_input_lines + \
                    [(index, line)]

                if len("\n".join([line for (_, line) in tmp_next_input_lines])) > max_sem_input_count:
                    sem_tokens_generator.generate(
                        text='\n'.join(
                            [line for (_, line) in next_sem_tokens_input_lines]),
                        output_name=path.join(
                            sem_tokens_output_dir, f"{input_hash}_{next_sem_tokens_input_lines[0][0]}_{next_sem_tokens_input_lines[-1][0]}"
                        )
                    )

                    for (pi, _) in next_sem_tokens_input_lines:
                        pipeline_states.save_processed_line(pi)

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
