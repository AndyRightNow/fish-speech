
import numpy as np
import soundfile as sf
import torch
import torchaudio
from typing import Any
from loguru import logger

from .inference import load_model

from tools.file import AUDIO_EXTENSIONS


class VQGanInference:
    __model: Any = None
    __device = ''
    
    def __init__(
        self,
        config_name="firefly_gan_vq",
        checkpoint_path="checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        device="cuda"
    ):
        self.__model = load_model(config_name, checkpoint_path, device=device)
        self.__device = device

    def __restore(self, indices, output_path):
        # Restore
        feature_lengths = torch.tensor(
            [indices.shape[1]], device=self.__device)
        fake_audios, _ = self.__model.decode(
            indices=indices[None], feature_lengths=feature_lengths
        )
        audio_time = fake_audios.shape[-1] / \
            self.__model.spec_transform.sample_rate

        logger.info(
            f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}"
        )

        # Save audio
        fake_audio = fake_audios[0, 0].float().cpu().numpy()
        sf.write(output_path, fake_audio,
                 self.__model.spec_transform.sample_rate)
        logger.info(f"Saved audio to {output_path}")

    @torch.no_grad()
    def generate_from_npy(self, input_path, output_path):
        assert input_path.suffix == ".npy"

        logger.info(f"Processing precomputed indices from {input_path}")
        indices = np.load(input_path)
        indices = torch.from_numpy(indices).to(self.__device).long()
        assert indices.ndim == 2, f"Expected 2D indices, got {indices.ndim}"

        self.__restore(indices=indices, output_path=output_path)

    @torch.no_grad()
    def generate_from_audio(self, input_path, output_path):
        assert input_path.suffix in AUDIO_EXTENSIONS
        logger.info(f"Processing in-place reconstruction of {input_path}")

        # Load audio
        audio, sr = torchaudio.load(str(input_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(
            audio, sr, self.__model.spec_transform.sample_rate
        )

        audios = audio[None].to(self.__device)
        logger.info(
            f"Loaded audio with {audios.shape[2] / self.__model.spec_transform.sample_rate:.2f} seconds"
        )

        # VQ Encoder
        audio_lengths = torch.tensor(
            [audios.shape[2]], device=self.__device, dtype=torch.long)
        indices = self.__model.encode(audios, audio_lengths)[0][0]

        logger.info(f"Generated indices of shape {indices.shape}")

        # Save indices
        np.save(output_path.with_suffix(".npy"), indices.cpu().numpy())

        self.__restore(indices=indices, output_path=output_path)
