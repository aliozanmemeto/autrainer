# wav2vec2_enrollment.py - Corrected Wav2Vec2 model with enrollment personalization
"""Wav2Vec2 model with enrollment-based speaker personalization for emotion recognition.

This implementation follows the enrollment personalization approach where speaker-specific
enrollment utterances are used to adapt the model's predictions.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from autrainer.models.abstract_model import AbstractModel


class Wav2Vec2Enrollment(AbstractModel):
    """Wav2Vec2 model with enrollment-based personalization.

    The model uses attention mechanism to fuse speaker-specific enrollment
    embeddings with the main audio embedding for personalized predictions.

    Supports three enrollment modes:
    - "neutral": Only use neutral enrollment utterance
    - "emotional": Only use emotional enrollment utterances
    - "all": Use both neutral and emotional enrollments
    - None: Baseline without personalization
    """

    def __init__(
        self,
        output_dim: int,
        pretrained_name: str = "facebook/wav2vec2-base",
        dropout: float = 0.1,
        freeze_feature_extractor: bool = True,
        freeze_encoder: bool = False,
        unfreeze_last_n_layers: int = 0,
        num_heads: int = 4,
        enrollment: Optional[str] = None,
        transfer: Optional[Union[bool, str]] = None,
    ) -> None:
        """Initialize the Wav2Vec2 enrollment model.

        Args:
            output_dim: Number of output classes.
            pretrained_name: HuggingFace model name for pretrained weights.
            dropout: Dropout rate for classifier.
            freeze_feature_extractor: Whether to freeze the CNN feature extractor.
            freeze_encoder: Whether to freeze the transformer encoder.
            unfreeze_last_n_layers: Number of last encoder layers to unfreeze
                (only used if freeze_encoder=True).
            num_heads: Number of attention heads for enrollment fusion.
            enrollment: Enrollment mode - "neutral", "emotional", "all", or None.
            transfer: Transfer learning configuration (not used for HF models).
        """
        self.pretrained_name = pretrained_name
        self.dropout_rate = dropout
        self.freeze_feature_extractor = freeze_feature_extractor
        self.freeze_encoder = freeze_encoder
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.num_heads = num_heads
        self.enrollment = enrollment

        super().__init__(output_dim=output_dim, transfer=transfer)

        # Load Wav2Vec2 encoder
        self._load_encoder()

        # Validate attention heads
        if self.enrollment is not None and self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"num_heads={self.num_heads} must divide hidden_size={self.hidden_size}. "
                f"Model {self.pretrained_name} has hidden_size={self.hidden_size}."
            )

        # Apply freezing
        self._apply_freezing()

        # Enrollment attention (only if using personalization)
        if self.enrollment is not None:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=self.num_heads,
                batch_first=True,
            )
        else:
            self.attention = None

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.output_dim),
        )

    def _load_encoder(self) -> None:
        """Load the Wav2Vec2 encoder with error handling for older torch versions."""
        try:
            self.encoder = Wav2Vec2Model.from_pretrained(self.pretrained_name)
        except ValueError as e:
            if "vulnerability" in str(e).lower():
                # Fallback for torch < 2.6 security restrictions
                print(
                    f"[WARN] Loading config-only for {self.pretrained_name} "
                    "(torch security restriction). Model will be randomly initialized."
                )
                config = Wav2Vec2Config.from_pretrained(self.pretrained_name)
                self.encoder = Wav2Vec2Model(config)
            else:
                raise

        self.hidden_size = self.encoder.config.hidden_size

    def _apply_freezing(self) -> None:
        """Apply freezing configuration to encoder layers."""
        # Freeze feature extractor (CNN layers)
        if self.freeze_feature_extractor:
            if hasattr(self.encoder, "feature_extractor"):
                self.encoder.feature_extractor._freeze_parameters()

        # Freeze encoder (transformer layers)
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

            # Optionally unfreeze last N layers
            if self.unfreeze_last_n_layers > 0 and hasattr(self.encoder, "encoder"):
                layers_to_unfreeze = self.encoder.encoder.layers[
                    -self.unfreeze_last_n_layers :
                ]
                for layer in layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True

    def _encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Encode waveforms through Wav2Vec2 and return mean-pooled embeddings.

        Args:
            waveforms: Input tensor of shape [B, L] or [B*K, L].

        Returns:
            Mean-pooled embeddings of shape [B, H] or [B*K, H].
        """
        outputs = self.encoder(waveforms, attention_mask=None)
        hidden_states = outputs.last_hidden_state  # [B, T, H]
        pooled = hidden_states.mean(dim=1)  # [B, H]
        return pooled

    def embeddings(
        self,
        features: torch.Tensor,
        enroll_neutral: Optional[torch.Tensor] = None,
        enroll_emotional: Optional[torch.Tensor] = None,
        enroll_neutral_label: Optional[torch.Tensor] = None,
        enroll_emotional_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract personalized embeddings before the classifier.

        Args:
            features: Main audio waveforms [B, L].
            enroll_neutral: Neutral enrollment waveforms [B, L].
            enroll_emotional: Emotional enrollment waveforms [B, K, L].
            enroll_neutral_label: Neutral labels [B] (unused, for compatibility).
            enroll_emotional_labels: Emotional labels [B, K] (unused, for compatibility).

        Returns:
            Personalized embeddings [B, H].
        """
        # Encode main instance
        instance_emb = self._encode(features)  # [B, H]

        # If no enrollment mode, return instance embedding directly
        if self.enrollment is None:
            return instance_emb

        # Validate enrollment data is provided
        if enroll_neutral is None and enroll_emotional is None:
            raise ValueError(
                f"enrollment='{self.enrollment}' requires enrollment data, "
                "but none was provided."
            )

        # Build enrollment tensor based on mode
        if self.enrollment == "neutral":
            if enroll_neutral is None:
                raise ValueError("enrollment='neutral' requires enroll_neutral tensor")
            # [B, L] -> [B, 1, L]
            enroll = enroll_neutral.unsqueeze(1)

        elif self.enrollment == "emotional":
            if enroll_emotional is None:
                raise ValueError("enrollment='emotional' requires enroll_emotional tensor")
            # [B, K, L]
            enroll = enroll_emotional

        elif self.enrollment == "all":
            if enroll_neutral is None or enroll_emotional is None:
                raise ValueError(
                    "enrollment='all' requires both enroll_neutral and enroll_emotional"
                )
            # Concatenate neutral and emotional: [B, 1+K, L]
            enroll = torch.cat(
                [enroll_neutral.unsqueeze(1), enroll_emotional],
                dim=1,
            )

        else:
            raise ValueError(
                f"Unknown enrollment mode: '{self.enrollment}'. "
                "Must be 'neutral', 'emotional', 'all', or None."
            )

        # Encode enrollment utterances
        B, C, L = enroll.shape
        enroll_flat = enroll.reshape(B * C, L)  # [B*C, L]
        enroll_emb_flat = self._encode(enroll_flat)  # [B*C, H]
        enroll_emb = enroll_emb_flat.view(B, C, -1)  # [B, C, H]

        # Apply attention: instance attends to enrollment
        query = instance_emb.unsqueeze(1)  # [B, 1, H]
        attended, _ = self.attention(query, enroll_emb, enroll_emb)  # [B, 1, H]

        # Residual connection
        personalized = (query + attended).squeeze(1)  # [B, H]

        return personalized

    def forward(
        self,
        features: torch.Tensor,
        enroll_neutral: Optional[torch.Tensor] = None,
        enroll_emotional: Optional[torch.Tensor] = None,
        enroll_neutral_label: Optional[torch.Tensor] = None,
        enroll_emotional_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: extract personalized embeddings and classify.

        Args:
            features: Main audio waveforms [B, L].
            enroll_neutral: Neutral enrollment waveforms [B, L].
            enroll_emotional: Emotional enrollment waveforms [B, K, L].
            enroll_neutral_label: Neutral labels [B] (unused).
            enroll_emotional_labels: Emotional labels [B, K] (unused).

        Returns:
            Classification logits [B, output_dim].
        """
        personalized = self.embeddings(
            features=features,
            enroll_neutral=enroll_neutral,
            enroll_emotional=enroll_emotional,
            enroll_neutral_label=enroll_neutral_label,
            enroll_emotional_labels=enroll_emotional_labels,
        )

        logits = self.classifier(personalized)
        return logits
