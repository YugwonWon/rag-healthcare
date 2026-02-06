"""
ChromaDB 데이터 전체 재구축 스크립트
- healthcare_docs/*.txt 및 대화예제(conversations/*.txt)를 모두 로드
- 기존 docs 컬렉션을 초기화한 뒤 최신 파일로 재구축
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


def load_text_file(file_path: Path, category: str) -> list[dict]:
    """
    텍스트 파일을 읽어서 청크로 분할
    
    Args:
        file_path: 텍스트 파일 경로
        category: 카테고리 (healthcare_docs / conversations)
    
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
    file_name = file_path.stem
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
                    "category": category
                }
            })
    
    return result


def rebuild_database(clear_existing: bool = True) -> dict:
    """
    ChromaDB docs 컬렉션을 재구축
    
    Args:
        clear_existing: True면 기존 데이터를 삭제 후 재구축
    
    Returns:
        결과 통계
    """
    chroma = get_chroma_handler()
    
    # 기존 문서 수 확인
    existing_count = chroma._docs_collection.count()
    logger.info(f"기존 문서 수: {existing_count}")
    
    if clear_existing and existing_count > 0:
        logger.info("기존 docs 컬렉션 데이터를 삭제합니다...")
        # 기존 문서 모두 가져와서 삭제
        existing = chroma._docs_collection.get()
        if existing and existing.get("ids"):
            chroma._docs_collection.delete(ids=existing["ids"])
            logger.info(f"  -> {len(existing['ids'])}개 기존 문서 삭제 완료")
    
    stats = {"healthcare_docs": 0, "conversations": 0, "total_chunks": 0, "files_processed": 0}
    
    # 1) healthcare_docs 로드
    docs_dir = project_root / "data" / "healthcare_docs"
    if docs_dir.exists():
        txt_files = sorted(docs_dir.glob("*.txt"))
        logger.info(f"\n📂 healthcare_docs: {len(txt_files)}개 파일 발견")
        
        for txt_file in txt_files:
            try:
                chunks = load_text_file(txt_file, category="healthcare_docs")
                if not chunks:
                    logger.warning(f"  ⚠️ 청크 없음: {txt_file.name}")
                    continue
                
                documents = [c["text"] for c in chunks]
                metadatas = [c["metadata"] for c in chunks]
                ids = [f"hd_{txt_file.stem}_{i}" for i in range(len(chunks))]
                
                chroma.add_documents(documents=documents, metadatas=metadatas, ids=ids)
                
                stats["healthcare_docs"] += len(chunks)
                stats["files_processed"] += 1
                logger.info(f"  ✅ {txt_file.name} -> {len(chunks)}개 청크")
            except Exception as e:
                logger.error(f"  ❌ 실패: {txt_file.name} - {e}")
    else:
        logger.warning(f"healthcare_docs 폴더 없음: {docs_dir}")
    
    # 2) 대화예제(conversations) 로드
    conv_dir = project_root / "data" / "conversations"
    if conv_dir.exists():
        conv_files = sorted(conv_dir.glob("*.txt"))
        logger.info(f"\n📂 conversations: {len(conv_files)}개 파일 발견")
        
        for txt_file in conv_files:
            try:
                chunks = load_text_file(txt_file, category="conversations")
                if not chunks:
                    logger.warning(f"  ⚠️ 청크 없음: {txt_file.name}")
                    continue
                
                documents = [c["text"] for c in chunks]
                metadatas = [c["metadata"] for c in chunks]
                ids = [f"cv_{txt_file.stem}_{i}" for i in range(len(chunks))]
                
                chroma.add_documents(documents=documents, metadatas=metadatas, ids=ids)
                
                stats["conversations"] += len(chunks)
                stats["files_processed"] += 1
                logger.info(f"  ✅ {txt_file.name} -> {len(chunks)}개 청크")
            except Exception as e:
                logger.error(f"  ❌ 실패: {txt_file.name} - {e}")
    else:
        logger.warning(f"conversations 폴더 없음: {conv_dir}")
    
    stats["total_chunks"] = stats["healthcare_docs"] + stats["conversations"]
    
    # 최종 확인
    final_count = chroma._docs_collection.count()
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 재구축 결과:")
    logger.info(f"   healthcare_docs 청크: {stats['healthcare_docs']}개")
    logger.info(f"   conversations 청크:   {stats['conversations']}개")
    logger.info(f"   처리된 파일 수:       {stats['files_processed']}개")
    logger.info(f"   총 청크 수:           {stats['total_chunks']}개")
    logger.info(f"   DB 최종 문서 수:      {final_count}개")
    logger.info(f"{'='*60}")
    
    return stats


def test_search():
    """재구축 후 테스트 검색"""
    chroma = get_chroma_handler()
    
    test_queries = ["폐 건강", "수면 장애", "갱년기 증상", "구강 관리"]
    
    print("\n" + "=" * 60)
    print("🔍 테스트 검색 결과")
    print("=" * 60)
    
    for query in test_queries:
        results = chroma.search_documents(query, n_results=2)
        print(f"\n🔎 '{query}':")
        
        if results and results.get("documents") and results["documents"][0]:
            for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                source = metadata.get("source", "unknown")
                category = metadata.get("category", "unknown")
                print(f"  [{i+1}] ({category}/{source}) {doc[:150]}...")
        else:
            print("  결과 없음")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ChromaDB 데이터 재구축")
    parser.add_argument("--no-clear", action="store_true", help="기존 데이터를 삭제하지 않고 추가만 함")
    parser.add_argument("--test", action="store_true", help="재구축 후 테스트 검색 실행")
    args = parser.parse_args()
    
    logger.info("🚀 ChromaDB 데이터 재구축 시작")
    stats = rebuild_database(clear_existing=not args.no_clear)
    
    if args.test:
        test_search()
    
    logger.info("✅ 완료!")


if __name__ == "__main__":
    main()
