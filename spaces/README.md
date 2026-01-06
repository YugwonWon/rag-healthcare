---
title: 치매노인 맞춤형 헬스케어 챗봇
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.0
app_file: app.py
pinned: false
license: mit
---

# 🏥 치매노인 맞춤형 헬스케어 RAG 챗봇

따뜻하고 친절한 AI 도우미와 대화해보세요.

## 주요 기능

- **개인화된 대화**: 닉네임 기반으로 이전 대화를 기억하고 맞춤형 응답 제공
- **복약 알림**: 복약 시간을 부드럽게 상기
- **일상 루틴 관리**: 식사, 산책, 취침 등 일과 관리
- **증상 모니터링**: 대화 중 건강 이상 징후 감지

## 사용 방법

1. 닉네임을 입력하고 "시작하기" 버튼 클릭
2. 채팅창에서 자유롭게 대화
3. 프로필 탭에서 추가 정보 입력 가능

## 기술 스택

- **백엔드**: FastAPI + ChromaDB + Qwen 2.5 3B
- **프론트엔드**: Gradio
- **임베딩**: sentence-transformers/all-MiniLM-L6-v2

---

Made with ❤️ for dementia care
