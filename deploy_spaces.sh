#!/bin/bash
# HuggingFace Spaces 배포 스크립트

set -e

# 설정
HF_SPACE="${HF_SPACE:-Yugwon/rag-healthcare}"
SPACES_DIR="$(dirname "$0")/spaces"

# 색상 출력
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 HuggingFace Spaces 배포 시작${NC}"
echo "  Space: ${HF_SPACE}"
echo "  디렉토리: ${SPACES_DIR}"

# spaces 디렉토리 확인
if [ ! -d "${SPACES_DIR}" ]; then
    echo "❌ spaces 디렉토리를 찾을 수 없습니다: ${SPACES_DIR}"
    exit 1
fi

# HuggingFace CLI 확인
if ! command -v hf &> /dev/null; then
    echo -e "${YELLOW}⚠️ hf CLI가 설치되어 있지 않습니다. 설치 중...${NC}"
    pipx install huggingface_hub
fi

# 로그인 확인
echo -e "\n${YELLOW}🔐 HuggingFace 로그인 확인 중...${NC}"
if ! hf auth whoami &> /dev/null; then
    echo "로그인이 필요합니다."
    hf auth login
fi

# 업로드
echo -e "\n${YELLOW}📤 Spaces에 업로드 중...${NC}"
cd "${SPACES_DIR}"
hf upload ${HF_SPACE} . --repo-type space

echo -e "\n${GREEN}✅ 배포 완료!${NC}"
echo -e "🔗 Space URL: https://huggingface.co/spaces/${HF_SPACE}"
echo ""
echo -e "${YELLOW}⚠️ 주의: 백엔드 URL 설정이 필요합니다${NC}"
echo "  1. https://huggingface.co/spaces/${HF_SPACE}/settings 접속"
echo "  2. Repository secrets에 BACKEND_URL 추가"
echo "     예: https://your-cloudrun-url.run.app"
