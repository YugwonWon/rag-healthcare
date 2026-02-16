"""
치매노인 맞춤형 헬스케어 RAG 챗봇 애플리케이션
"""

import os
import warnings

# tqdm/safetensors/transformers 로그 억제 (모든 모듈보다 먼저 설정)
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LangChain deprecation 경고 억제 (가장 먼저 설정)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", message=".*LangChain.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

import logging
for _name in ["transformers", "safetensors", "tqdm", "tqdm.auto",
              "transformers.modeling_utils", "transformers.configuration_utils",
              "transformers.tokenization_utils_base",
              "huggingface_hub", "huggingface_hub.utils",
              "sentence_transformers"]:
    logging.getLogger(_name).setLevel(logging.ERROR)

__version__ = "0.1.0"
