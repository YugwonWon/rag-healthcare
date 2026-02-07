# 모델 배포 디렉토리

## 구조

```
models/
├── README.md
├── Modelfile.k-exaone-counseling   # EXAONE 4.0 1.2B 템플릿
├── Modelfile.kanana-counseling     # Kanana 2.1B 템플릿
├── k-exaone-counseling.gguf        # ← 배포 시 여기에 복사 (git에 포함 안 됨)
└── kanana-counseling.gguf          # ← 배포 시 여기에 복사 (git에 포함 안 됨)
```

## 사용법

### 1. GGUF 파일 복사
```bash
# EXAONE 모델 (현재 사용 중)
cp /Volumes/SAMSUNG-USB/models/exaone-counseling-q8.gguf models/k-exaone-counseling.gguf

# 또는 Kanana 모델
cp finetuning/output/kanana-counseling-merged/kanana-counseling-q4_k_m.gguf models/kanana-counseling.gguf
```

### 2. .env에서 모델 지정
```bash
OLLAMA_MODEL=k-exaone-counseling   # 또는 kanana-counseling
```

### 3. 배포
```bash
./deploy_cloudrun.sh
```

## 새 모델 추가하기

1. `models/Modelfile.{모델명}` 파일 생성
2. `models/{모델명}.gguf` 파일 배치
3. `.env`에 `OLLAMA_MODEL={모델명}` 설정
4. 배포 실행

Docker 빌드 시 `models/` 디렉토리 전체가 컨테이너에 복사되고,
`start_with_ollama.sh`가 `OLLAMA_MODEL` 환경변수에 해당하는 Modelfile을 자동으로 찾아 등록합니다.
