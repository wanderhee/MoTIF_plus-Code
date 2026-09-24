import torch
import torch.nn as nn

from .qformer import QFormer, QFormerConfig


class MoTIFPlusForConditionalGeneration(nn.Module):
    def __init__(
        self,
        visual_encoder: nn.Module,
        llm: nn.Module,
        qformer_config: QFormerConfig,
        max_frames: int = 8,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.llm = llm
        self.qformer = QFormer(qformer_config)
        self.qformer_config = qformer_config

        d_llm = llm.config.hidden_size
        self.llm_proj = nn.Linear(qformer_config.hidden_size, d_llm)
        self.max_frames = max_frames
        self.ignore_index = ignore_index

        self.frame_embeddings = nn.Parameter(
            torch.zeros(max_frames, qformer_config.d_visual)
        )
        nn.init.normal_(self.frame_embeddings, std=0.02)

        for p in self.visual_encoder.parameters():
            p.requires_grad = False

    def _extract_patch_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        is_video = pixel_values.dim() == 5
        if is_video:
            B, T, C, H, W = pixel_values.shape
            flat = pixel_values.reshape(B * T, C, H, W)
        else:
            B = pixel_values.shape[0]
            T = 1
            flat = pixel_values

        visual = self.visual_encoder

        hidden = visual.patch_embed(flat)
        for block in visual.blocks:
            hidden = block(hidden)

        hidden = hidden.reshape(B, T, -1, hidden.shape[-1])

        hidden = hidden + self.frame_embeddings[:T][None, :, None, :]

        return hidden.reshape(B, -1, hidden.shape[-1])

    def forward_qformer(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patch_features = self._extract_patch_features(pixel_values)
        return self.qformer(patch_features)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        patch_features = self._extract_patch_features(pixel_values)
        query = self.qformer(patch_features)
        visual_embeds = self.llm_proj(query)

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        B, nq = visual_embeds.shape[0], visual_embeds.shape[1]

        if attention_mask is not None:
            visual_mask = torch.ones(
                B, nq, dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        if labels is not None:
            visual_labels = torch.full(
                (B, nq), self.ignore_index, dtype=labels.dtype, device=labels.device
            )
            labels = torch.cat([visual_labels, labels], dim=1)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **generate_kwargs,
    ):
        patch_features = self._extract_patch_features(pixel_values)
        query = self.qformer(patch_features)
        visual_embeds = self.llm_proj(query)

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        B, nq = visual_embeds.shape[0], visual_embeds.shape[1]
        if attention_mask is not None:
            visual_mask = torch.ones(
                B, nq, dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
