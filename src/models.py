from enum import Enum


class LLMs(Enum):
    PHI_3_5_MINI_INSTRUCT_4BIT = "Phi-3.5-mini-instruct-4bit"
    SMOLLM_1_7B_FP16 = "SmolLM-1.7B-fp16"
    JOSEIFIED_QWEN2_5_14B_INSTRUCT_ABLITERATED_V4 = (
        "Josiefied-Qwen2.5-14B-Instruct-abliterated-v4"
    )
    QWQ_32B_PREVIEW_BF16 = "QwQ-32B-Preview-bf16"
    YI_1_5_34B_CHAT_8BIT = "Yi-1.5-34B-Chat-8bit"
    QWQ_32B_BF16 = "QwQ-32B-bf16"
    JINAAI_READERLM_V2 = "jinaai-ReaderLM-v2"
    MISTRAL_NEMO_INSTRUCT_2407_BF16 = "Mistral-Nemo-Instruct-2407-bf16"
    QWEN2_5_14B_INSTRUCT_1M_BF16 = "Qwen2.5-14B-Instruct-1M-bf16"
    MISTRAL_SMALL_24B_INSTRUCT_2501_BF16 = "Mistral-Small-24B-Instruct-2501-bf16"
    QWEN2_5_32B_INSTRUCT_BF16 = "Qwen2.5-32B-Instruct-bf16"
    MIXTRAL_8X22B_INSTRUCT_V0_1_8BIT = "Mixtral-8x22B-Instruct-v0.1-8bit"
    QWEN2_5_CODER_32B_INSTRUCT_BF16 = "Qwen2.5-Coder-32B-Instruct-bf16"
    OLMOE_1B_7B_0125_INSTRUCT = "OLMoE-1B-7B-0125-Instruct"
    MAMBA_CODESTRAL_7B_V0_1 = "Mamba-Codestral-7B-v0.1"


class VLMs(Enum):
    FLORENCE_2_LARGE_FT_BF16 = "Florence-2-large-ft-bf16"
    QWEN2_5_VL_32B_INSTRUCT_BF16 = "Qwen2.5-VL-32B-Instruct-bf16"
    OLMOCR_7B_0225_PREVIWE_BF16 = "olmOCR-7B-0225-preview-bf16"
    MOLMO_7B_D_0924_BF16 = "Molmo-7B-D-0924-bf16"
    SMOLVLM_INSTRUCT_BF16 = "SmolVLM-Instruct-bf16"


class ALMs(Enum):
    KOKORO_82M_BF16 = "Kokoro-82M-bf16"
    WHISPER_LARGE_V3_MLX = "whisper-large-v3-mlx"
