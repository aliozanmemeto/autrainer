# enrollment_aibo.py - Corrected implementation matching original paper
"""AIBO dataset with enrollment-based personalization for autrainer.

This implementation matches the original enrollment personalization paper,
where enrollment utterances are randomly cropped during training (not center-cropped).
"""

import os
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from autrainer.core.structs.abstract_data_struct import (
    AbstractDataBatch,
    AbstractDataItem,
)
from autrainer.datasets.abstract_dataset import AbstractDataset
from autrainer.datasets.utils import LabelEncoder
from autrainer.transforms import SmartCompose


@dataclass
class EnrollmentDataItem(AbstractDataItem):
    """Data item for enrollment-based personalization.

    Attributes:
        features: Main instance waveform tensor [L].
        enroll_neutral: Neutral enrollment waveform [L].
        enroll_emotional: Emotional enrollment waveforms [K, L].
        enroll_neutral_label: Encoded neutral label (int).
        enroll_emotional_labels: Encoded emotional labels [K].
        target: Target label for the instance.
        index: Sample index.
    """

    features: torch.Tensor
    enroll_neutral: torch.Tensor
    enroll_emotional: torch.Tensor
    enroll_neutral_label: int
    enroll_emotional_labels: torch.Tensor
    target: int
    index: int


@dataclass
class EnrollmentDataBatch(AbstractDataBatch[EnrollmentDataItem]):
    """Batch of enrollment data items.

    The `features` property returns only the main audio tensor for compatibility
    with autrainer's model input handling. Enrollment tensors are passed as
    separate batch attributes.
    """

    features: torch.Tensor  # [B, L] - main instance waveforms
    enroll_neutral: torch.Tensor  # [B, L]
    enroll_emotional: torch.Tensor  # [B, K, L]
    enroll_neutral_label: torch.Tensor  # [B]
    enroll_emotional_labels: torch.Tensor  # [B, K]
    target: torch.Tensor  # [B]
    index: torch.Tensor  # [B]

    def to(self, device: torch.device, **kwargs) -> "EnrollmentDataBatch":
        """Move all tensors to the specified device."""
        self.features = self.features.to(device, **kwargs)
        self.enroll_neutral = self.enroll_neutral.to(device, **kwargs)
        self.enroll_emotional = self.enroll_emotional.to(device, **kwargs)
        self.enroll_neutral_label = self.enroll_neutral_label.to(device, **kwargs)
        self.enroll_emotional_labels = self.enroll_emotional_labels.to(device, **kwargs)
        self.target = self.target.to(device, **kwargs)
        self.index = self.index.to(device, **kwargs)
        return self

    @classmethod
    def collate(cls, items: List[EnrollmentDataItem]) -> "EnrollmentDataBatch":
        """Collate a list of EnrollmentDataItem into a batch."""
        return cls(
            features=torch.stack([item.features for item in items], dim=0),
            enroll_neutral=torch.stack([item.enroll_neutral for item in items], dim=0),
            enroll_emotional=torch.stack([item.enroll_emotional for item in items], dim=0),
            enroll_neutral_label=torch.tensor(
                [item.enroll_neutral_label for item in items], dtype=torch.long
            ),
            enroll_emotional_labels=torch.stack(
                [item.enroll_emotional_labels for item in items], dim=0
            ),
            target=torch.tensor([item.target for item in items], dtype=torch.long),
            index=torch.tensor([item.index for item in items], dtype=torch.long),
        )


class EnrollmentAIBO(AbstractDataset):
    """FAU-AIBO dataset with enrollment-based personalization.

    This dataset implements the enrollment personalization approach where
    for each sample, we also provide:
    - A neutral enrollment utterance from the same speaker
    - Emotional enrollment utterances from the same speaker (one per class)

    Matches the original paper's data handling with random cropping during training.
    """

    # Neutral class for each task (matching original paper)
    NEUTRAL_CLASSES = {
        "2cl": "IDL",  # For 2-class: IDL is neutral, NEG is emotional
        "5cl": "N",  # For 5-class: N is neutral
    }

    def __init__(
        self,
        path: str,
        features_subdir: Optional[str] = None,
        seed: int = 42,
        metrics: Optional[List] = None,
        tracking_metric: Optional[str] = None,
        index_column: str = "file",
        target_column: str = "class",
        file_type: str = "wav",
        file_handler: Optional[Union[str, Dict]] = None,
        train_transform: Optional[SmartCompose] = None,
        dev_transform: Optional[SmartCompose] = None,
        test_transform: Optional[SmartCompose] = None,
        stratify: Optional[List[str]] = None,
        # AIBO-specific parameters
        aibo_task: str = "2cl",
        crop_size: int = 48000,
        sample_rate: int = 16000,
    ) -> None:
        """Initialize the EnrollmentAIBO dataset.

        Args:
            path: Root path to the AIBO dataset.
            features_subdir: Subdirectory containing audio files.
            seed: Random seed for reproducibility.
            metrics: List of metrics to calculate.
            tracking_metric: Metric to track for model selection.
            index_column: Column name for file index.
            target_column: Column name for target labels.
            file_type: Audio file extension.
            file_handler: File handler configuration.
            train_transform: Transforms for training set.
            dev_transform: Transforms for development set.
            test_transform: Transforms for test set.
            stratify: Columns to stratify on.
            aibo_task: Task variant in ["2cl", "5cl"].
            crop_size: Number of samples to crop audio to.
            sample_rate: Audio sample rate.
        """
        self.aibo_task = aibo_task
        self.crop_size = crop_size
        self.sample_rate = sample_rate

        if aibo_task not in ["2cl", "5cl"]:
            raise ValueError(f"aibo_task must be '2cl' or '5cl', got '{aibo_task}'")

        # Set defaults
        if metrics is None:
            metrics = ["autrainer.metrics.UAR"]
        if tracking_metric is None:
            tracking_metric = "autrainer.metrics.UAR"
        if file_handler is None:
            file_handler = {
                "autrainer.datasets.utils.AudioFileHandler": {
                    "target_sample_rate": sample_rate
                }
            }

        super().__init__(
            path=path,
            features_subdir=features_subdir,
            seed=seed,
            task="classification",
            metrics=metrics,
            tracking_metric=tracking_metric,
            index_column=index_column,
            target_column=target_column,
            file_type=file_type,
            file_handler=file_handler,
            train_transform=train_transform,
            dev_transform=dev_transform,
            test_transform=test_transform,
            stratify=stratify,
        )

        # Identify neutral and emotional classes
        self._neutral_class = self.NEUTRAL_CLASSES[self.aibo_task]
        all_labels = sorted(self._load_df["class"].unique().tolist())
        self._emotional_classes = [c for c in all_labels if c != self._neutral_class]
        self._all_labels = all_labels

        # Build enrollment map: (school, speaker, class) -> filename
        self._build_enrollment_map()

    @property
    def audio_subdir(self) -> str:
        """Subdirectory containing audio data."""
        return "wav"

    @cached_property
    def _load_df(self) -> pd.DataFrame:
        """Load and parse the AIBO labels file."""
        label_file = f"chunk_labels_{self.aibo_task}_corpus.txt"
        label_path = os.path.join(self.path, label_file)

        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f"Label file not found: {label_path}. "
                f"Expected file: {label_file}"
            )

        df = pd.read_csv(label_path, header=None, sep=r"\s+")
        df.columns = ["id", "class", "conf"]
        df["file"] = df["id"].apply(lambda x: x + ".wav")
        df["school"] = df["id"].apply(lambda x: x.split("_")[0])
        df["speaker"] = df["id"].apply(lambda x: x.split("_")[1])
        return df

    def _build_enrollment_map(self) -> None:
        """Build mapping from (school, speaker, class) to enrollment filename.

        For each speaker and class, we select the first file (alphabetically)
        as the enrollment utterance, matching the original paper.
        """
        self._enrollment_map: Dict[Tuple[str, str, str], str] = {}
        df_sorted = self._load_df.sort_values("file")

        for _, row in df_sorted.iterrows():
            key = (row["school"], row["speaker"], row["class"])
            if key not in self._enrollment_map:
                self._enrollment_map[key] = row["file"]

        # Track which files are used for enrollment (excluded from training)
        self._enrollment_files = set(self._enrollment_map.values())

    @cached_property
    def df_train(self) -> pd.DataFrame:
        """Training dataframe (Ohm school, excluding last 2 speakers for dev)."""
        df = self._load_df
        df_ohm = df[df["school"] == "Ohm"]
        speakers = sorted(df_ohm["speaker"].unique())
        train_speakers = speakers[:-2]
        return df_ohm[df_ohm["speaker"].isin(train_speakers)].reset_index(drop=True)

    @cached_property
    def df_dev(self) -> pd.DataFrame:
        """Development dataframe (last 2 speakers from Ohm school)."""
        df = self._load_df
        df_ohm = df[df["school"] == "Ohm"]
        speakers = sorted(df_ohm["speaker"].unique())
        dev_speakers = speakers[-2:]
        return df_ohm[df_ohm["speaker"].isin(dev_speakers)].reset_index(drop=True)

    @cached_property
    def df_test(self) -> pd.DataFrame:
        """Test dataframe (Mont school)."""
        df = self._load_df
        return df[df["school"] == "Mont"].reset_index(drop=True)

    @cached_property
    def target_transform(self) -> LabelEncoder:
        """Label encoder for target classes."""
        return LabelEncoder(self._all_labels)

    @property
    def default_collate_fn(self) -> Callable:
        """Return the custom collate function for enrollment batches."""
        return EnrollmentDataBatch.collate

    def _init_dataset(
        self,
        df: pd.DataFrame,
        transform: SmartCompose,
    ) -> "_EnrollmentAIBOWrapper":
        """Initialize a dataset wrapper for the given split."""
        is_train = df is self.df_train

        return _EnrollmentAIBOWrapper(
            df=df,
            path=self.path,
            audio_subdir=self.audio_subdir,
            enrollment_map=self._enrollment_map,
            enrollment_files=self._enrollment_files,
            neutral_class=self._neutral_class,
            emotional_classes=self._emotional_classes,
            target_transform=self.target_transform,
            transform=transform,
            crop_size=self.crop_size,
            random_crop=is_train,  # Random crop only during training
        )


class _EnrollmentAIBOWrapper(torch.utils.data.Dataset):
    """Internal dataset wrapper for AIBO with enrollment data."""

    def __init__(
        self,
        df: pd.DataFrame,
        path: str,
        audio_subdir: str,
        enrollment_map: Dict[Tuple[str, str, str], str],
        enrollment_files: set,
        neutral_class: str,
        emotional_classes: List[str],
        target_transform: LabelEncoder,
        transform: Optional[SmartCompose],
        crop_size: int,
        random_crop: bool = False,
    ):
        self.df = df
        self.path = path
        self.audio_subdir = audio_subdir
        self.enrollment_map = enrollment_map
        self.enrollment_files = enrollment_files
        self.neutral_class = neutral_class
        self.emotional_classes = emotional_classes
        self.target_transform = target_transform
        self.transform = transform
        self.crop_size = crop_size
        self.random_crop = random_crop

        # Exclude enrollment files from the dataset
        # (enrollment files are only used as references, not as training samples)
        self.valid_indices = self.df[
            ~self.df["file"].isin(self.enrollment_files)
        ].index.tolist()

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _load_audio(self, filepath: str) -> torch.Tensor:
        """Load audio file and return as tensor."""
        import audiofile

        audio, sr = audiofile.read(filepath)
        x = torch.from_numpy(audio.astype(np.float32))
        if x.ndim > 1:
            x = x.mean(dim=0)  # Convert stereo to mono
        return x

    def _crop_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Crop or pad audio to target size.

        Uses random crop for training, center crop for eval (matching original paper).
        """
        L = x.shape[-1]

        if L > self.crop_size:
            if self.random_crop:
                # Random crop (for training)
                start = torch.randint(0, L - self.crop_size + 1, (1,)).item()
            else:
                # Center crop (for eval)
                start = (L - self.crop_size) // 2
            return x[start : start + self.crop_size]
        elif L < self.crop_size:
            # Pad with zeros
            pad_total = self.crop_size - L
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return F.pad(x, (pad_left, pad_right), mode="constant", value=0)
        return x

    def _get_enrollment(
        self, school: str, speaker: str
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Get enrollment tensors for a speaker.

        Returns:
            enroll_neutral: [crop_size] neutral enrollment audio
            enroll_emotional: [K, crop_size] emotional enrollment audios
            neutral_label: encoded neutral label
            emotional_labels: [K] encoded emotional labels
        """
        # Neutral enrollment
        neutral_key = (school, speaker, self.neutral_class)
        if neutral_key in self.enrollment_map:
            filepath = os.path.join(
                self.path, self.audio_subdir, self.enrollment_map[neutral_key]
            )
            enroll_neutral = self._crop_audio(self._load_audio(filepath))
        else:
            # No neutral sample available - use zeros
            enroll_neutral = torch.zeros(self.crop_size, dtype=torch.float32)

        neutral_label = self.target_transform.encode(self.neutral_class)

        # Emotional enrollments
        K = len(self.emotional_classes)
        enroll_emotional = torch.zeros(K, self.crop_size, dtype=torch.float32)
        emotional_labels = torch.zeros(K, dtype=torch.long)

        for i, emo_class in enumerate(self.emotional_classes):
            emotional_labels[i] = self.target_transform.encode(emo_class)
            emo_key = (school, speaker, emo_class)
            if emo_key in self.enrollment_map:
                filepath = os.path.join(
                    self.path, self.audio_subdir, self.enrollment_map[emo_key]
                )
                enroll_emotional[i] = self._crop_audio(self._load_audio(filepath))
            # else: keep as zeros

        return enroll_neutral, enroll_emotional, neutral_label, emotional_labels

    def __getitem__(self, idx: int) -> EnrollmentDataItem:
        """Get a data item with its enrollment data."""
        df_idx = self.valid_indices[idx]
        row = self.df.loc[df_idx]

        # Load and crop main audio
        filepath = os.path.join(self.path, self.audio_subdir, row["file"])
        audio = self._crop_audio(self._load_audio(filepath))

        # Get target label
        target = self.target_transform.encode(row["class"])

        # Get enrollment data
        enroll_neutral, enroll_emotional, neutral_label, emotional_labels = (
            self._get_enrollment(row["school"], row["speaker"])
        )

        # Create data item
        item = EnrollmentDataItem(
            features=audio,
            enroll_neutral=enroll_neutral,
            enroll_emotional=enroll_emotional,
            enroll_neutral_label=neutral_label,
            enroll_emotional_labels=emotional_labels,
            target=target,
            index=idx,
        )

        # Apply transforms (only affects item.features)
        if self.transform is not None:
            item = self.transform(item)

        return item
