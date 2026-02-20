"""
Neo4j 지식그래프 + GraphRAG 통합 테스트

전제 조건:
  - Neo4j 서버 실행 중 (docker compose up -d neo4j)
  - build_neo4j_kg.py 실행 완료 또는 최소 seed 데이터 존재

테스트 항목:
  1. Neo4j 연결 테스트
  2. 노드/엣지 MERGE 테스트
  3. 검색 API 테스트 (get_condition_info, get_symptom_conditions, find_related_nodes)
  4. GraphRAG 검색기 키워드 추출 테스트
  5. GraphRAG end-to-end 검색 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_neo4j_connection():
    """1. Neo4j 연결 테스트"""
    print("\n" + "=" * 50)
    print("테스트 1: Neo4j 연결")
    print("=" * 50)

    from app.knowledge_graph.health_kg import get_neo4j_kg

    try:
        kg = get_neo4j_kg()
        stats = kg.get_stats()
        print(f"  ✅ 연결 성공 | 노드={stats['node_count']}, 엣지={stats['edge_count']}")
        return True
    except Exception as e:
        print(f"  ❌ 연결 실패: {e}")
        return False


def test_merge_operations():
    """2. 노드/엣지 MERGE 테스트"""
    print("\n" + "=" * 50)
    print("테스트 2: MERGE 연산 (쓰기)")
    print("=" * 50)

    from app.knowledge_graph.health_kg import get_neo4j_kg, NodeLabel, RelType

    kg = get_neo4j_kg()

    # 테스트 노드 생성
    kg.merge_node("테스트_질환", NodeLabel.CONDITION, "테스트용 질환")
    kg.merge_node("테스트_증상", NodeLabel.SYMPTOM, "테스트용 증상")

    # 관계 생성
    kg.merge_relationship("테스트_증상", "테스트_질환", RelType.SYMPTOM_OF)

    # 검증
    info = kg.get_condition_info("테스트_질환")
    assert info.get("condition") == "테스트_질환", "질환 정보 조회 실패"
    assert len(info.get("symptoms", [])) > 0, "증상 관계 조회 실패"

    print(f"  ✅ MERGE 성공 | 질환={info['condition']}, 증상={[s['name'] for s in info['symptoms']]}")

    # 클린업 (테스트 데이터 삭제)
    with kg.driver.session(database=kg._database) as session:
        session.run("MATCH (n) WHERE n.name STARTS WITH '테스트_' DETACH DELETE n")
    print("  ✅ 테스트 데이터 정리 완료")

    return True


def test_seed_and_query():
    """3. 시드 데이터 + 검색 API 테스트"""
    print("\n" + "=" * 50)
    print("테스트 3: 시드 데이터 + 검색 API")
    print("=" * 50)

    from app.knowledge_graph.health_kg import get_neo4j_kg, NodeLabel, RelType

    kg = get_neo4j_kg()

    # 시드: 당뇨 관련 기본 데이터
    kg.merge_node("당뇨", NodeLabel.CONDITION, "혈당 조절 이상")
    kg.merge_node("고혈당", NodeLabel.SYMPTOM, "혈당이 높은 상태")
    kg.merge_node("저혈당", NodeLabel.SYMPTOM, "혈당이 낮아 어지러움")
    kg.merge_node("혈당 모니터링", NodeLabel.TREATMENT, "정기적 혈당 측정")
    kg.merge_node("식이 조절", NodeLabel.TREATMENT, "당분 제한, 균형 잡힌 식단")

    kg.merge_relationship("고혈당", "당뇨", RelType.SYMPTOM_OF)
    kg.merge_relationship("저혈당", "당뇨", RelType.SYMPTOM_OF)
    kg.merge_relationship("당뇨", "혈당 모니터링", RelType.MANAGED_BY)
    kg.merge_relationship("당뇨", "식이 조절", RelType.MANAGED_BY)

    # 검색 테스트
    info = kg.get_condition_info("당뇨")
    print(f"  질환: {info.get('condition')}")
    print(f"  증상: {[s['name'] for s in info.get('symptoms', [])]}")
    print(f"  치료: {[t['name'] for t in info.get('treatments', [])]}")

    conditions = kg.get_symptom_conditions("고혈당")
    print(f"  '고혈당' → 가능 질환: {conditions}")

    related = kg.find_related_nodes("당뇨", depth=2)
    print(f"  '당뇨' 관련 노드 ({len(related)}개): {[n['name'] for n in related[:5]]}")

    all_nodes = kg.get_all_node_names()
    print(f"  전체 노드 수: {len(all_nodes)}개")

    assert "당뇨" in info.get("condition", ""), "질환 조회 실패"
    assert len(info.get("symptoms", [])) >= 2, "증상 조회 부족"
    assert "당뇨" in conditions, "증상→질환 매핑 실패"

    print("  ✅ 검색 API 테스트 통과")
    return True


def test_graph_rag_keywords():
    """4. GraphRAG 키워드 추출 테스트"""
    print("\n" + "=" * 50)
    print("테스트 4: GraphRAG 키워드 추출")
    print("=" * 50)

    from app.knowledge_graph.graph_rag import GraphRAGRetriever
    from app.knowledge_graph.health_kg import get_neo4j_kg

    kg = get_neo4j_kg()
    retriever = GraphRAGRetriever(kg=kg)

    test_queries = [
        ("혈당이 높아서 걱정이에요", ["고혈당"]),
        ("잠을 못 자서 힘들어요", ["불면증"]),
        ("발톱이 안쪽으로 파고들어요", ["내향성 발톱"]),
        ("눈이 침침해요", ["시력 저하"]),
    ]

    passed = 0
    for query, expected in test_queries:
        keywords = retriever._extract_keywords_hybrid(query)
        hit = any(exp in keywords for exp in expected)
        status = "✅" if hit else "⚠️"
        print(f"  {status} '{query}' → {keywords} (기대: {expected})")
        if hit:
            passed += 1

    print(f"\n  결과: {passed}/{len(test_queries)} 통과")
    return passed > 0


def test_graph_rag_search():
    """5. GraphRAG end-to-end 검색 테스트"""
    print("\n" + "=" * 50)
    print("테스트 5: GraphRAG E2E 검색")
    print("=" * 50)

    from app.knowledge_graph.graph_rag import GraphRAGRetriever
    from app.knowledge_graph.health_kg import get_neo4j_kg

    kg = get_neo4j_kg()
    retriever = GraphRAGRetriever(kg=kg)

    queries = [
        "혈당이 높아서 걱정이에요",
        "잠을 못 자요",
        "발톱이 아파요",
    ]

    for query in queries:
        result = retriever.search(query)
        has_result = bool(result.strip())
        status = "✅" if has_result else "⚠️ (빈 결과)"
        print(f"  {status} '{query}'")
        if has_result:
            for line in result.split("\n")[:4]:
                print(f"      {line}")
        print()

    return True


def test_node_index_stats():
    """6. 전체 그래프 통계"""
    print("\n" + "=" * 50)
    print("테스트 6: 그래프 통계")
    print("=" * 50)

    from app.knowledge_graph.health_kg import get_neo4j_kg

    kg = get_neo4j_kg()
    stats = kg.get_stats()
    all_names = kg.get_all_node_names()

    print(f"  총 노드: {stats['node_count']}")
    print(f"  총 관계: {stats['edge_count']}")
    print(f"  노드명 목록 (상위 20): {all_names[:20]}")

    labels = ["Condition", "Symptom", "Treatment", "BodyPart", "Lifestyle"]
    for label in labels:
        names = kg.get_node_names_by_label(label)
        if names:
            print(f"  {label}: {len(names)}개 — {names[:5]}")

    return True


def main():
    print("🧪 Neo4j GraphRAG 통합 테스트")
    print("=" * 60)

    results = {}

    # 1. 연결 테스트 (이게 실패하면 나머지 불필요)
    try:
        results["연결"] = test_neo4j_connection()
    except Exception as e:
        print(f"  ❌ Neo4j 연결 실패: {e}")
        print("\n💡 Neo4j가 실행 중인지 확인하세요:")
        print("   docker compose up -d neo4j")
        return

    if not results["연결"]:
        print("\n❌ Neo4j 연결 실패. 이후 테스트를 건너뜁니다.")
        return

    # 2~6 테스트
    tests = [
        ("MERGE 연산", test_merge_operations),
        ("시드+검색 API", test_seed_and_query),
        ("키워드 추출", test_graph_rag_keywords),
        ("E2E 검색", test_graph_rag_search),
        ("그래프 통계", test_node_index_stats),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  ❌ {name} 테스트 실패: {e}")
            results[name] = False

    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 테스트 결과 요약")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} — {name}")

    total_pass = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  총 결과: {total_pass}/{total} 통과")


if __name__ == "__main__":
    main()
