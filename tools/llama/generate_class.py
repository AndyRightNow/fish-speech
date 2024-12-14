import time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo.config
import torch._inductor.config
from .generate import generate_long, load_model
import loguru


class LlamaSemanticTokensGenerator:
    __model = None
    __device = ''
    __prompt_text = []
    __prompt_tokens = []
    __decode_one_token = None
    __logger = loguru.logger

    def __init__(
        self,
        prompt_tokens: list[Path],
        prompt_text: list[str],
        checkpoint_path: Path = Path("checkpoints/fish-speech-1.5"),
        device: str = "cuda",
        compile: bool = True,
        seed: int = 42,
        half: bool = False,
        logger=loguru.logger
    ):
        self.__logger = logger

        precision = torch.half if half else torch.bfloat16

        if prompt_text is not None and len(prompt_text) != len(prompt_tokens):
            raise ValueError(
                f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same"
            )

        self.__prompt_text = prompt_text

        logger.debug("Loading model ...")
        t0 = time.time()
        model, decode_one_token = load_model(
            checkpoint_path, device, precision, compile=compile
        )

        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.debug(f"Time to load model: {time.time() - t0:.02f} seconds")

        if prompt_tokens is not None:
            prompt_tokens = [torch.from_numpy(np.load(p)).to(
                device) for p in prompt_tokens]

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.__model = model
        self.__decode_one_token = decode_one_token
        self.__prompt_tokens = prompt_tokens
        self.__device = device

    def generate(
        self,
        text: str,
        output_name: str,
        num_samples: int = 1,
        max_new_tokens: int = 0,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        temperature: float = 0.7,
        iterative_prompt: bool = True,
        chunk_length: int = 100
    ):
        generator = generate_long(
            model=self.__model,
            device=self.__device,
            decode_one_token=self.__decode_one_token,
            text=text,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            iterative_prompt=iterative_prompt,
            chunk_length=chunk_length,
            prompt_text=self.__prompt_text,
            prompt_tokens=self.__prompt_tokens,
        )

        idx = 0
        codes = []

        file_name = f"{output_name}_{idx}.npy"

        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
                self.__logger.debug(f"Sampled text: {response.text}")
            elif response.action == "next":
                if codes:
                    np.save(file_name, torch.cat(codes, dim=1).cpu().numpy())
                    self.__logger.debug(f"Saved codes to{file_name}")
                self.__logger.debug(f"Next sample")
                codes = []
                idx += 1
            else:
                self.__logger.error(f"Error: {response}")
