from pathlib import Path
from loguru import logger
from tools.vqgan.inference_class import VQGanInference
from os import path
import constants
from pipeline_states import PipelineStates
import sys
from input import Input

sem_tokens_output_dir = path.join(constants.base_output_dir, "audio")


if __name__ == "__main__":
    pipeline_states = None
    try:
        input = Input("input.txt")
        pipeline_states = PipelineStates(input_hash=input.input_hash)
        audio_generator = VQGanInference()

        try:
            pass
        except Exception as e:
            logger.error(f"Unable to generate audio: {e}")
        finally:
            pipeline_states.save()
    except KeyboardInterrupt:
        if pipeline_states is not None:
            pipeline_states.save()
        sys.exit(130)
