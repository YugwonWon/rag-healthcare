#!/bin/bash
# MeloTTS 한국어 TTS 사이드카용 격리 venv 구성 (재현 가능)
#
# 왜 별도 venv인가:
# - MeloTTS는 transformers==4.27.4를 핀해서 메인 앱 venv(transformers 5.x)와 공존 불가.
# - 한국어 G2P용 python-mecab-ko(모듈 `mecab`)와 일본어용 mecab-python3(모듈 `MeCab`)가
#   macOS 대소문자 비구분 FS에서 같은 폴더로 충돌 → mecab-python3 제거하고 fugashi만 사용.
# - melo는 한국어인데도 import 시 일본어 모듈을 끌어오므로, 일본어 MeCab 의존을
#   비치명적으로 패치한다(아래 7단계).
#
# 사전 요구: Homebrew, python3.11, `brew install mecab`(fugashi 빌드용).
# 사용: ./scripts/setup_melo_tts.sh [venv경로]   (기본: <repo>/.venv-tts)

set -e

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${1:-$PROJECT/.venv-tts}"
PY311="$(command -v python3.11 || echo /opt/homebrew/bin/python3.11)"
export PATH="/opt/homebrew/bin:$PATH"   # mecab-config (fugashi 빌드)

echo "▶ Python 3.11: $PY311"
"$PY311" --version
echo "▶ venv 생성: $VENV"
"$PY311" -m venv "$VENV"
PIP="$VENV/bin/pip"
PYV="$VENV/bin/python"

"$PIP" install -q -U pip wheel setuptools

echo "▶ 1) MeloTTS(git) 설치 (torch, transformers==4.27.4 등 동반)"
"$PIP" install "git+https://github.com/myshell-ai/MeloTTS.git"

echo "▶ 2) 전체 unidic 제거 (빈 dicdir의 mecabrc 오류 회피, unidic-lite 사용)"
"$PIP" uninstall -y unidic || true

echo "▶ 3) mecab-python3 제거 (python-mecab-ko의 mecab/ 와 대소문자 충돌)"
"$PIP" uninstall -y mecab-python3 || true

echo "▶ 4) 한국어 G2P 바인딩(python-mecab-ko) 클린 설치"
"$PIP" install --force-reinstall --no-deps python-mecab-ko python-mecab-ko-dic

echo "▶ 5) 일본어 BERT 토크나이저용 fugashi + unidic-lite (mecab과 네임스페이스 안 겹침)"
"$PIP" install fugashi unidic-lite

echo "▶ 6) 사이드카 서버 의존성(fastapi/uvicorn)"
"$PIP" install fastapi "uvicorn[standard]"

echo "▶ 7) melo 소스 패치 (한국어 전용: 일본어 MeCab 의존 비치명화)"
SP="$("$PYV" -c 'import melo, os; print(os.path.dirname(melo.__file__))')"

# 7a) japanese.py: import MeCab 실패를 비치명적으로, _TAGGER 가드
"$PYV" - "$SP/text/japanese.py" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
s = s.replace(
    'except ImportError as e:\n    raise ImportError("Japanese requires mecab-python3 and unidic-lite.") from e',
    'except ImportError:\n    MeCab = None  # KR 전용: 일본어 미사용, 비치명적',
)
s = s.replace(
    "_TAGGER = MeCab.Tagger()",
    "_TAGGER = MeCab.Tagger() if MeCab is not None else None",
)
open(p, "w", encoding="utf-8").write(s)
print("  - japanese.py patched")
PYEOF

# 7b) cleaner.py: 언어 모듈을 지연 로딩(한국어만 로드 → 중국어 BERT 등 불필요 로드 회피)
cat > "$SP/text/cleaner.py" <<'PYEOF'
import copy
import importlib

_LANG_MODULE_NAMES = {"ZH": "chinese", "JP": "japanese", "EN": "english",
                      "ZH_MIX_EN": "chinese_mix", "KR": "korean",
                      "FR": "french", "SP": "spanish", "ES": "spanish"}
_module_cache = {}


def _get_language_module(language):
    if language not in _module_cache:
        _module_cache[language] = importlib.import_module(
            f"melo.text.{_LANG_MODULE_NAMES[language]}")
    return _module_cache[language]


from . import cleaned_text_to_sequence


def clean_text(text, language):
    m = _get_language_module(language)
    norm_text = m.text_normalize(text)
    phones, tones, word2ph = m.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language, device=None):
    m = _get_language_module(language)
    norm_text = m.text_normalize(text)
    phones, tones, word2ph = m.g2p(norm_text)
    word2ph_bak = copy.deepcopy(word2ph)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    bert = m.get_bert_feature(norm_text, word2ph, device=device)
    return norm_text, phones, tones, word2ph_bak, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)
PYEOF
echo "  - cleaner.py patched(지연 로딩)"

echo "✅ 완료: $VENV"
echo "   사이드카 실행: ./scripts/run_melo_sidecar.sh"
