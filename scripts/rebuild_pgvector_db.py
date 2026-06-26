"""
Cloud SQL (pgvector) 문서 데이터 재구축 스크립트
- healthcare_docs/*.txt 및 대화예제(conversations/*.txt)를 
  LangChain + pgvector (Cloud SQL)에 로드
- 기존 문서 삭제 후 최신 파일로 재구축
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


def rebuild_pgvector_db():
    """pgvector DB의 문서 컬렉션을 재구축"""
    
    # 1) 설정 확인
    db_url = settings.database_url
    if not db_url:
        logger.error("DATABASE_URL이 설정되지 않았습니다. .env 파일을 확인하세요.")
        return
    
    logger.info(f"DB 연결: {db_url[:40]}...")
    
    # 2) LangChain 스토어 초기화
    from app.langchain_store import LangChainDataStore
    store = LangChainDataStore(db_url)
    
    if not store.is_postgres_enabled:
        logger.error("PostgreSQL이 활성화되지 않았습니다.")
        return
    
    # 3) 기존 문서 삭제
    logger.info("\n🗑️  기존 문서 삭제 중...")
    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # 현재 문서 수 확인
        cur.execute("SELECT COUNT(*) FROM langchain_pg_embedding")
        before_count = cur.fetchone()[0]
        logger.info(f"   기존 문서 수: {before_count}")
        
        # 기존 문서 데이터 삭제 (컬렉션 메타데이터는 유지)
        cur.execute("DELETE FROM langchain_pg_embedding")
        conn.commit()
        logger.info(f"   -> {before_count}개 문서 삭제 완료")
        
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"   기존 문서 삭제 실패: {e}")
        logger.info("   기존 데이터 삭제를 건너뛰고 추가 모드로 진행합니다.")
    
    # 4) healthcare_docs 로드
    docs_dir = project_root / "data" / "healthcare_docs"
    conv_dir = project_root / "data" / "conversations"
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    from langchain_core.documents import Document
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n---\n", "\n\n", "\n", " "]
    )
    
    all_docs = []
    stats = {"healthcare_docs": 0, "conversations": 0}
    
    # healthcare_docs
    if docs_dir.exists():
        txt_files = sorted(docs_dir.glob("*.txt"))
        logger.info(f"\n📂 healthcare_docs: {len(txt_files)}개 파일")
        
        for txt_file in txt_files:
            try:
                loader = TextLoader(str(txt_file), encoding="utf-8")
                docs = loader.load()
                # 메타데이터에 카테고리 추가
                for doc in docs:
                    doc.metadata["category"] = "healthcare_docs"
                    doc.metadata["source_name"] = txt_file.stem
                
                splits = splitter.split_documents(docs)
                all_docs.extend(splits)
                stats["healthcare_docs"] += len(splits)
                logger.info(f"  ✅ {txt_file.name} -> {len(splits)}개 청크")
            except Exception as e:
                logger.error(f"  ❌ {txt_file.name}: {e}")
    
    # conversations
    if conv_dir.exists():
        conv_files = sorted(conv_dir.glob("*.txt"))
        logger.info(f"\n📂 conversations: {len(conv_files)}개 파일")
        
        for txt_file in conv_files:
            try:
                loader = TextLoader(str(txt_file), encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["category"] = "conversations"
                    doc.metadata["source_name"] = txt_file.stem
                
                splits = splitter.split_documents(docs)
                all_docs.extend(splits)
                stats["conversations"] += len(splits)
                logger.info(f"  ✅ {txt_file.name} -> {len(splits)}개 청크")
            except Exception as e:
                logger.error(f"  ❌ {txt_file.name}: {e}")
    
    # 5) pgvector에 일괄 로드
    total = len(all_docs)
    logger.info(f"\n📤 pgvector에 {total}개 청크 로딩 중...")
    
    # 배치 처리 (한 번에 너무 많이 보내면 타임아웃 가능)
    batch_size = 50
    loaded = 0
    
    for i in range(0, total, batch_size):
        batch = all_docs[i:i + batch_size]
        try:
            store.vectorstore.add_documents(batch)
            loaded += len(batch)
            logger.info(f"  [{loaded}/{total}] 로드 완료")
        except Exception as e:
            logger.error(f"  배치 {i//batch_size + 1} 로드 실패: {e}")
    
    # 6) 최종 확인
    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM langchain_pg_embedding")
        final_count = cur.fetchone()[0]
        cur.close()
        conn.close()
    except Exception:
        final_count = "확인 불가"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 pgvector DB 재구축 결과:")
    logger.info(f"   healthcare_docs 청크: {stats['healthcare_docs']}개")
    logger.info(f"   conversations 청크:   {stats['conversations']}개")
    logger.info(f"   총 로드 청크:         {loaded}개")
    logger.info(f"   DB 최종 문서 수:      {final_count}개")
    logger.info(f"{'='*60}")


def test_search():
    """pgvector 검색 테스트"""
    from app.langchain_store import get_langchain_store
    store = get_langchain_store()
    
    test_queries = ["폐 건강 관리", "수면 장애 개선", "갱년기 증상", "구강 관리 방법"]
    
    print("\n" + "=" * 60)
    print("🔍 pgvector 검색 테스트")
    print("=" * 60)
    
    for query in test_queries:
        results = store.search_documents(query, k=3)
        print(f"\n🔎 '{query}':")
        if results:
            for i, r in enumerate(results):
                source = r["metadata"].get("source_name", "unknown")
                category = r["metadata"].get("category", "unknown")
                score = r.get("score", "N/A")
                content_preview = r["content"][:120].replace("\n", " ")
                print(f"  [{i+1}] ({category}/{source}, score:{score:.3f}) {content_preview}...")
        else:
            print("  결과 없음")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cloud SQL pgvector 문서 재구축")
    parser.add_argument("--test", action="store_true", help="재구축 후 검색 테스트")
    args = parser.parse_args()
    
    logger.info("🚀 Cloud SQL pgvector DB 재구축 시작")
    rebuild_pgvector_db()
    
    if args.test:
        test_search()
    
    logger.info("✅ 완료!")
