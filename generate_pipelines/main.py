from tools.llama.generate import main as generate_semantic_tokens
from pathlib import Path
from loguru import logger
from os import path
import json
from hashlib import sha256, sha1
import sys

base_dir = "generate_pipelines"
pipeline_states_json_path = "states.json"
base_output_dir = path.join(base_dir, "output")
max_sem_input_count = 1000

def wrapped_generate_semantic_tokens(text, file_hash):
    generate_semantic_tokens(
        text = text,
        prompt_text = ["人间灯火倒映湖中，她的渴望让静水泛起涟漪。若代价只是孤独，那就让这份愿望肆意流淌。流入她所注视的世间，也流入她如湖水般澄澈的目光。"],
        prompt_tokens = [Path("4_output_trained.npy")],
        output_name=f"generate_pipelines/output/semantic_tokens/{file_hash}",
        num_samples = 1,
        compile = True
    )


if __name__ == "__main__":
    pipeline_states = {}

    def save_pipeline_states():
        with open(path.join(base_dir, pipeline_states_json_path), 'w', encoding='utf-8') as pipeline_states_json_file:
            logger.info("Updating pipeline states")
            pipeline_states_json_file.write(json.dumps(pipeline_states))

    try:

        logger.info("Reading pipeline states")
        with open(path.join(base_dir, pipeline_states_json_path), 'r', encoding='utf-8') as pipeline_states_json_file:
            try:
                pipeline_states = json.loads(pipeline_states_json_file.read())
            except json.decoder.JSONDecodeError:
                pipeline_states = {}

        input_lines = []
        input_hash = ''

        logger.info("Reading input file")
        with open(path.join(base_dir, "input.txt"), "r", encoding="utf-8") as input_file:
            file_content = input_file.read()
            m = sha256()
            input_lines = file_content.splitlines()
            m.update(str.encode(file_content))
            input_hash = m.hexdigest()

        if input_hash not in pipeline_states:
            pipeline_states[input_hash] = {}

        if 'line_count' not in pipeline_states[input_hash]:
            pipeline_states[input_hash]['line_count'] = len(input_lines)
        
        next_sem_tokens_input = ''

        try:
            for line in input_lines:
                m = sha1()
                m.update(str.encode(line))
                line_hash = m.hexdigest()

                if 'processed_lines' in pipeline_states[input_hash] and line_hash in pipeline_states[input_hash]['processed_lines']:
                    continue
                
                tmp_next_input = f"{next_sem_tokens_input}\n{line}"

                if len(tmp_next_input) > max_sem_input_count:
                    wrapped_generate_semantic_tokens(next_sem_tokens_input, input_hash)
                    pipeline_states[input_hash]['processed_lines'] = list((set() if 'processed_lines' not in pipeline_states[input_hash] else set(pipeline_states[input_hash])).union({
                        line_hash
                    }))
                else:
                    next_sem_tokens_input = tmp_next_input
        except Exception as e:
            logger.error(f"Unable to generate semantic tokens: {e}")
        finally:
            save_pipeline_states()
    except KeyboardInterrupt:
        save_pipeline_states()
        sys.exit(130)
