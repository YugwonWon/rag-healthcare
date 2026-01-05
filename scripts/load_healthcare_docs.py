"""
healthcare_docs 폴더의 텍스트 파일들을 ChromaDB에 로드하는 스크립트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.vector_store import get_chroma_handler
from app.logger import get_logger

logger = get_logger(__name__)


def load_text_file(file_path: Path) -> list[dict]:
    """
    텍스트 파일을 읽어서 청크로 분할
    
    Args:
        file_path: 텍스트 파일 경로
    
    Returns:
        청크 리스트 (각 청크는 text, metadata를 포함)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 구분선(---) 또는 빈 줄로 섹션 분할
    sections = []
    current_section = []
    
    for line in content.split("\n"):
        if line.strip().startswith("---") or (not line.strip() and current_section and len("\n".join(current_section)) > 500):
            if current_section:
                section_text = "\n".join(current_section).strip()
                if section_text:
                    sections.append(section_text)
                current_section = []
        else:
            current_section.append(line)
    
    # 마지막 섹션 추가
    if current_section:
        section_text = "\n".join(current_section).strip()
        if section_text:
            sections.append(section_text)
    
    # 섹션이 너무 크면 추가 분할 (1000자 기준)
    chunks = []
    for section in sections:
        if len(section) > 1000:
            # 문단 단위로 분할
            paragraphs = section.split("\n\n")
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) > 1000:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks.append(section)
    
    # 메타데이터 추가
    file_name = file_path.stem  # 확장자 제외한 파일명
    result = []
    for i, chunk in enumerate(chunks):
        if chunk.strip():
            result.append({
                "text": chunk,
                "metadata": {
                    "source": file_name,
                    "file_path": str(file_path),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "category": "healthcare_docs"
                }
            })
    
    return result


def load_all_documents(docs_dir: Path) -> None:
    """
    폴더 내 모든 텍스트 파일을 ChromaDB에 로드
    
    Args:
        docs_dir: 문서 폴더 경로
    """
    chroma = get_chroma_handler()
    
    # 기존 문서 수 확인
    existing_count = chroma._docs_collection.count()
    logger.info(f"기존 문서 수: {existing_count}")
    
    # 텍스트 파일 수집
    txt_files = list(docs_dir.glob("*.txt"))
    logger.info(f"발견된 텍스트 파일 수: {len(txt_files)}")
    
    total_chunks = 0
    
    for txt_file in txt_files:
        logger.info(f"처리 중: {txt_file.name}")
        
        try:
            chunks = load_text_file(txt_file)
            
            if not chunks:
                logger.warning(f"청크 없음: {txt_file.name}")
                continue
            
            # ChromaDB에 추가
            documents = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            ids = [f"{txt_file.stem}_{i}" for i in range(len(chunks))]
            
            chroma.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            total_chunks += len(chunks)
            logger.info(f"  -> {len(chunks)}개 청크 추가 완료")
            
        except Exception as e:
            logger.error(f"파일 처리 실패: {txt_file.name} - {e}")
    
    # 최종 문서 수 확인
    final_count = chroma._docs_collection.count()
    logger.info(f"로드 완료! 총 {total_chunks}개 청크 추가됨")
    logger.info(f"최종 문서 수: {final_count}")


def main():
    """메인 실행 함수"""
    # healthcare_docs 폴더 경로
    docs_dir = project_root / "data" / "healthcare_docs"
    
    if not docs_dir.exists():
        logger.error(f"폴더를 찾을 수 없습니다: {docs_dir}")
        return
    
    logger.info(f"문서 로드 시작: {docs_dir}")
    load_all_documents(docs_dir)
    
    # 테스트 검색
    print("\n" + "="*50)
    print("테스트 검색: '노안'")
    print("="*50)
    
    chroma = get_chroma_handler()
    results = chroma.search_documents("노안 증상과 치료", n_results=3)
    
    if results and results.get("documents"):
        for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            print(f"\n[결과 {i+1}] (출처: {metadata.get('source', 'unknown')})")
            print(f"{doc[:300]}...")
    else:
        print("검색 결과 없음")


if __name__ == "__main__":
    main()
