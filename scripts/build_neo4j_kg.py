"""
Neo4j 지식그래프 자동 구축 파이프라인

healthcare_docs/*.txt 문서를 읽어서
LLM(LLMGraphTransformer)으로 엔티티/관계를 자동 추출하고
Neo4j에 적재하는 원커맨드 스크립트.

사용법:
  # 기본 (Ollama 로컬 LLM 사용)
  python scripts/build_neo4j_kg.py

  # OpenAI GPT 사용 (더 높은 추출 품질)
  python scripts/build_neo4j_kg.py --use-openai --openai-key sk-xxx

  # 기존 그래프 초기화 후 재구축
  python scripts/build_neo4j_kg.py --reset

  # 특정 문서만 처리
  python scripts/build_neo4j_kg.py --file data/healthcare_docs/당뇨.txt
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

# ── 스키마 정의 (추출 시 LLM에게 제공) ──
ALLOWED_NODES = [
    "Condition",    # 질환/질병
    "Symptom",      # 증상
    "Treatment",    # 치료/관리법
    "BodyPart",     # 신체 부위
    "RiskFactor",   # 위험 요인
    "Medication",   # 약물
    "Lifestyle",    # 생활습관
]

ALLOWED_RELATIONSHIPS = [
    "SYMPTOM_OF",   # 증상 → 질환
    "MANAGED_BY",   # 질환 → 치료
    "CAUSES",       # 위험요인 → 질환
    "PREVENTS",     # 생활습관 → 질환
    "AFFECTS",      # 질환 → 신체부위
    "TREATS",       # 약물 → 질환
    "RELATED_TO",   # 일반 관계
]

# ── 노드 속성 ──
NODE_PROPERTIES = ["description"]


def load_documents(docs_dir: Path, target_file: str = None) -> list:
    """
    healthcare_docs 폴더의 텍스트 파일을 LangChain Document로 로드.
    """
    from langchain_core.documents import Document

    documents = []

    if target_file:
        files = [Path(target_file)]
    else:
        files = sorted(docs_dir.glob("*.txt"))

    for file_path in files:
        if not file_path.exists():
            logger.warning(f"파일 없음: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            continue

        # 문서를 섹션 단위로 분할 (너무 긴 문서는 LLM 추출 품질 저하)
        sections = _split_into_sections(content)
        for i, section in enumerate(sections):
            if len(section.strip()) < 50:  # 너무 짧은 섹션 스킵
                continue
            doc = Document(
                page_content=section,
                metadata={
                    "source": file_path.stem,
                    "file_path": str(file_path),
                    "section_index": i,
                },
            )
            documents.append(doc)

    logger.info(f"📄 문서 로드 완료: {len(files)}개 파일 → {len(documents)}개 섹션")
    return documents


def _split_into_sections(content: str, max_length: int = 8000) -> list[str]:
    """
    문서를 섹션 단위로 분할 후 작은 섹션들을 병합.
    GPT-4o-mini 128K 컨텍스트를 활용하여 큰 청크로 만들어 API 호출 수를 줄인다.
    """
    import re

    # 번호 제목(1. 2. 3. 등) 기준 분할
    raw_sections = re.split(r'\n(?=\d+[\.\)]\s)', content)

    # 각 섹션이 너무 길면 추가 분할
    small_sections = []
    for section in raw_sections:
        if len(section) > max_length:
            # 빈 줄 기준 분할
            paragraphs = section.split("\n\n")
            current = ""
            for para in paragraphs:
                if len(current) + len(para) > max_length:
                    if current.strip():
                        small_sections.append(current.strip())
                    current = para
                else:
                    current += "\n\n" + para if current else para
            if current.strip():
                small_sections.append(current.strip())
        else:
            if section.strip():
                small_sections.append(section.strip())

    # 작은 섹션들을 max_length까지 병합하여 API 호출 수 최소화
    result = []
    current = ""
    for section in small_sections:
        if len(current) + len(section) + 2 > max_length:
            if current.strip():
                result.append(current.strip())
            current = section
        else:
            current += "\n\n" + section if current else section
    if current.strip():
        result.append(current.strip())

    return result


def create_llm(use_openai: bool = False, openai_key: str = None):
    """LLM 인스턴스 생성"""
    if use_openai and openai_key:
        from langchain_openai import ChatOpenAI
        logger.info("🤖 OpenAI GPT-4o-mini 사용 (고품질 추출)")
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_key,
            temperature=0,
        )
    else:
        from langchain_ollama import ChatOllama
        logger.info(f"🤖 Ollama {settings.OLLAMA_MODEL} 사용 (로컬 추출)")
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
        )


def create_graph_transformer(llm):
    """LLMGraphTransformer 인스턴스 생성 (스키마 제한 적용)"""
    from langchain_experimental.graph_transformers import LLMGraphTransformer

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS,
        node_properties=NODE_PROPERTIES,
        strict_mode=True,  # 스키마 외 노드/관계 제거
    )
    return transformer


def extract_and_load(
    documents: list,
    transformer,
    neo4j_kg,
    batch_size: int = 5,
    max_workers: int = 8,
) -> dict:
    """
    문서를 LLM으로 병렬 처리하여 그래프 엔티티/관계를 추출하고 Neo4j에 적재한다.
    ThreadPoolExecutor로 max_workers개 문서를 동시에 처리한다.

    Returns:
        통계 dict {total_nodes, total_edges, processed_docs, failed_docs}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total_graph_docs = []
    failed = 0
    completed = 0
    total = len(documents)

    def process_one(doc, idx):
        """단일 문서 처리 (스레드에서 실행)"""
        try:
            start = time.time()
            graph_docs = transformer.convert_to_graph_documents([doc])
            elapsed = time.time() - start
            nodes_count = sum(len(gd.nodes) for gd in graph_docs)
            edges_count = sum(len(gd.relationships) for gd in graph_docs)
            return {
                "idx": idx,
                "graph_docs": graph_docs,
                "nodes": nodes_count,
                "edges": edges_count,
                "elapsed": elapsed,
                "source": doc.metadata.get("source", "?"),
                "error": None,
            }
        except Exception as e:
            return {"idx": idx, "graph_docs": [], "nodes": 0, "edges": 0,
                    "elapsed": 0, "source": doc.metadata.get("source", "?"),
                    "error": str(e)}

    print(f"\n🚀 병렬 추출 시작: {total}개 문서, {max_workers}개 워커", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one, doc, i): i
            for i, doc in enumerate(documents)
        }

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if result["error"]:
                failed += 1
                print(f"  ❌ [{completed}/{total}] {result['source']} - 실패: {result['error']}", flush=True)
            else:
                total_graph_docs.extend(result["graph_docs"])
                print(
                    f"  ✅ [{completed}/{total}] {result['source']} "
                    f"노드={result['nodes']}, 관계={result['edges']}, "
                    f"{result['elapsed']:.1f}초",
                    flush=True,
                )

    # Neo4j에 벌크 적재
    if total_graph_docs:
        print(f"\n📥 Neo4j 적재 시작: {len(total_graph_docs)}개 GraphDocument...", flush=True)
        try:
            neo4j_kg.bulk_import_graph_documents(total_graph_docs)
        except Exception as e:
            logger.error(f"Neo4j 벌크 적재 실패: {e}")
            # 개별 적재 폴백
            print("개별 적재 시도...", flush=True)
            for gd in total_graph_docs:
                try:
                    neo4j_kg.bulk_import_graph_documents([gd])
                except Exception as e2:
                    logger.warning(f"개별 적재 실패: {e2}")

    # 최종 통계
    stats = neo4j_kg.get_stats()
    return {
        "total_nodes": stats["node_count"],
        "total_edges": stats["edge_count"],
        "processed_docs": len(documents) - failed,
        "failed_docs": failed,
        "graph_documents": len(total_graph_docs),
    }


def seed_base_knowledge(neo4j_kg) -> dict:
    """
    LLM 추출 전에 핵심 스키마 노드와 관계를 시드(seed)로 심는다.
    기존 healthcare_docs 기반 도메인 지식의 골격을 구성한다.
    LLM 추출 결과와 MERGE되므로 중복되지 않는다.
    
    Returns:
        {nodes: int, edges: int}
    """
    from app.knowledge_graph.health_kg import NodeLabel, RelType

    # ── 노드 시드 ──
    nodes = [
        # 질환
        ("수면장애", NodeLabel.CONDITION, "고령자 수면 장애"),
        ("불면증", NodeLabel.SYMPTOM, "잠들기 어렵거나 자주 깨는 증상"),
        ("수면 패턴 변화", NodeLabel.SYMPTOM, "수면 주기 변화"),
        ("주간 졸림", NodeLabel.SYMPTOM, "낮에 졸리는 증상"),
        ("규칙적 수면 습관", NodeLabel.TREATMENT, "일정한 시간 취침/기상"),
        ("수면 환경 개선", NodeLabel.TREATMENT, "어둡고 조용한 환경, 적정 온도"),
        ("낮 활동량 증가", NodeLabel.TREATMENT, "낮에 적절한 운동과 활동"),
        ("카페인 제한", NodeLabel.LIFESTYLE, "오후 카페인 섭취 제한"),

        ("내향성 발톱", NodeLabel.CONDITION, "발톱이 살 안쪽으로 파고드는 질환"),
        ("발톱 변형", NodeLabel.SYMPTOM, "발톱이 두꺼워지거나 휘는 증상"),
        ("발톱 통증", NodeLabel.SYMPTOM, "발톱 주변 통증, 염증"),
        ("발톱 관리", NodeLabel.TREATMENT, "올바른 발톱 깎기 (일자로)"),
        ("편한 신발", NodeLabel.TREATMENT, "발에 맞는 편안한 신발 착용"),
        ("발", NodeLabel.BODY_PART, ""),
        ("발톱 무좀", NodeLabel.CONDITION, "발톱 곰팡이 감염"),

        ("당뇨", NodeLabel.CONDITION, "혈당 조절 이상"),
        ("고혈당", NodeLabel.SYMPTOM, "혈당이 높은 상태"),
        ("저혈당", NodeLabel.SYMPTOM, "혈당이 낮아 어지러움"),
        ("갈증", NodeLabel.SYMPTOM, "심한 갈증"),
        ("빈뇨", NodeLabel.SYMPTOM, "소변을 자주 보는 증상"),
        ("혈당 모니터링", NodeLabel.TREATMENT, "정기적 혈당 측정"),
        ("식이 조절", NodeLabel.TREATMENT, "당분 제한, 균형 잡힌 식단"),
        ("규칙적 운동", NodeLabel.LIFESTYLE, "가벼운 유산소 운동"),

        ("갱년기", NodeLabel.CONDITION, "호르몬 변화에 의한 증상"),
        ("안면홍조", NodeLabel.SYMPTOM, "얼굴이 갑자기 달아오르는 증상"),
        ("감정 기복", NodeLabel.SYMPTOM, "감정 변화가 심한 상태"),
        ("호르몬 치료", NodeLabel.TREATMENT, "의사 처방 호르몬 요법"),
        ("스트레스 관리", NodeLabel.LIFESTYLE, "명상, 이완 등"),

        ("치주 질환", NodeLabel.CONDITION, "잇몸 질환"),
        ("잇몸 출혈", NodeLabel.SYMPTOM, "잇몸에서 피가 나는 증상"),
        ("구취", NodeLabel.SYMPTOM, "입 냄새"),
        ("치아 흔들림", NodeLabel.SYMPTOM, "치아가 흔들리는 증상"),
        ("구강 위생", NodeLabel.TREATMENT, "올바른 칫솔질, 치실 사용"),
        ("정기 치과 검진", NodeLabel.TREATMENT, "6개월마다 치과 방문"),
        ("구강", NodeLabel.BODY_PART, ""),

        ("난청", NodeLabel.CONDITION, "청력 저하"),
        ("소리 안 들림", NodeLabel.SYMPTOM, "소리가 잘 안 들리는 증상"),
        ("이명", NodeLabel.SYMPTOM, "귀에서 소리가 나는 증상"),
        ("보청기", NodeLabel.TREATMENT, "보청기 착용"),
        ("청력 검사", NodeLabel.TREATMENT, "정기 청력 검사"),
        ("귀", NodeLabel.BODY_PART, ""),

        ("말초신경병증", NodeLabel.CONDITION, "말초 신경 손상"),
        ("손발 저림", NodeLabel.SYMPTOM, "손이나 발이 저린 증상"),
        ("감각 이상", NodeLabel.SYMPTOM, "감각이 둔해지는 증상"),
        ("혈액 순환 개선", NodeLabel.TREATMENT, "가벼운 운동, 마사지"),
        ("비타민B 보충", NodeLabel.TREATMENT, "비타민B12 등 보충"),

        ("요실금", NodeLabel.CONDITION, "소변 조절 어려움"),
        ("소변 실수", NodeLabel.SYMPTOM, "의도치 않은 소변 배출"),
        ("빈뇨감", NodeLabel.SYMPTOM, "소변을 자주 보고 싶은 느낌"),
        ("골반저 운동", NodeLabel.TREATMENT, "케겔 운동 등"),
        ("배뇨 훈련", NodeLabel.TREATMENT, "정해진 시간 배뇨 습관"),
        ("방광", NodeLabel.BODY_PART, ""),

        ("탈모", NodeLabel.CONDITION, "머리카락이 빠지는 상태"),
        ("머리카락 빠짐", NodeLabel.SYMPTOM, "머리카락이 많이 빠지는 증상"),
        ("두피 관리", NodeLabel.TREATMENT, "두피 청결, 마사지"),
        ("영양 섭취", NodeLabel.TREATMENT, "단백질, 철분 등 영양소 보충"),
        ("두피", NodeLabel.BODY_PART, ""),

        ("폐질환", NodeLabel.CONDITION, "만성 폐쇄성 폐질환 등"),
        ("만성기침", NodeLabel.SYMPTOM, "오래 지속되는 기침"),
        ("호흡곤란", NodeLabel.SYMPTOM, "숨이 차는 증상"),
        ("가래", NodeLabel.SYMPTOM, "가래가 많은 증상"),
        ("호흡 운동", NodeLabel.TREATMENT, "복식 호흡, 입술 오므리기 호흡"),
        ("금연", NodeLabel.LIFESTYLE, "흡연 중단"),
        ("폐", NodeLabel.BODY_PART, ""),

        ("노인성 피부", NodeLabel.CONDITION, "고령자 피부 건조, 가려움"),
        ("피부 가려움", NodeLabel.SYMPTOM, "피부가 가려운 증상"),
        ("피부 건조", NodeLabel.SYMPTOM, "피부가 건조한 상태"),
        ("보습제 사용", NodeLabel.TREATMENT, "보습크림 정기 도포"),
        ("미지근한 물 세안", NodeLabel.TREATMENT, "뜨거운 물 대신 미지근한 물"),
        ("피부", NodeLabel.BODY_PART, ""),

        ("욕창", NodeLabel.CONDITION, "오래 누워있어 피부가 손상"),
        ("피부 발적", NodeLabel.SYMPTOM, "피부가 빨갛게 되는 증상"),
        ("체위 변경", NodeLabel.TREATMENT, "2시간마다 자세 바꾸기"),
        ("피부 보호", NodeLabel.TREATMENT, "쿠션, 매트리스 사용"),

        ("변비", NodeLabel.CONDITION, "배변 어려움"),
        ("배변 곤란", NodeLabel.SYMPTOM, "변을 보기 어려운 증상"),
        ("복부 팽만", NodeLabel.SYMPTOM, "배가 더부룩한 증상"),
        ("수분 섭취", NodeLabel.TREATMENT, "충분한 물 마시기"),
        ("섬유질 섭취", NodeLabel.TREATMENT, "과일, 채소, 잡곡 섭취"),
        ("장", NodeLabel.BODY_PART, ""),

        ("노안", NodeLabel.CONDITION, "가까운 것이 잘 안 보이는 상태"),
        ("시력 저하", NodeLabel.SYMPTOM, "눈이 침침해지는 증상"),
        ("근거리 흐림", NodeLabel.SYMPTOM, "가까운 글씨가 잘 안 보임"),
        ("돋보기 사용", NodeLabel.TREATMENT, "적절한 도수의 돋보기"),
        ("안과 정기 검진", NodeLabel.TREATMENT, "정기 안과 검진"),
        ("눈", NodeLabel.BODY_PART, ""),

        ("식욕부진", NodeLabel.CONDITION, "먹고 싶은 욕구 저하"),
        ("체중 감소", NodeLabel.SYMPTOM, "의도치 않은 체중 감소"),
        ("기력 저하", NodeLabel.SYMPTOM, "기운이 없는 상태"),
        ("소량 다회 식사", NodeLabel.TREATMENT, "조금씩 자주 먹기"),
        ("영양 보충제", NodeLabel.TREATMENT, "경구 영양 보충"),

        ("골다공증", NodeLabel.CONDITION, "뼈가 약해지는 상태"),
        ("뼈 통증", NodeLabel.SYMPTOM, "뼈 부위 통증"),
        ("낙상 위험", NodeLabel.RISK_FACTOR, "넘어지기 쉬운 상태"),
        ("칼슘 섭취", NodeLabel.TREATMENT, "칼슘 + 비타민D 보충"),
        ("낙상 예방", NodeLabel.TREATMENT, "환경 정리, 미끄럼 방지"),
        ("뼈", NodeLabel.BODY_PART, ""),
    ]

    for name, label, desc in nodes:
        neo4j_kg.merge_node(name, label, desc)

    # ── 관계 시드 ──
    edges = [
        # 수면장애
        ("불면증", "수면장애", RelType.SYMPTOM_OF),
        ("수면 패턴 변화", "수면장애", RelType.SYMPTOM_OF),
        ("주간 졸림", "수면장애", RelType.SYMPTOM_OF),
        ("수면장애", "규칙적 수면 습관", RelType.MANAGED_BY),
        ("수면장애", "수면 환경 개선", RelType.MANAGED_BY),
        ("수면장애", "낮 활동량 증가", RelType.MANAGED_BY),
        ("카페인 제한", "수면장애", RelType.PREVENTS),

        # 발톱
        ("발톱 변형", "내향성 발톱", RelType.SYMPTOM_OF),
        ("발톱 통증", "내향성 발톱", RelType.SYMPTOM_OF),
        ("내향성 발톱", "발톱 관리", RelType.MANAGED_BY),
        ("내향성 발톱", "편한 신발", RelType.MANAGED_BY),
        ("내향성 발톱", "발", RelType.AFFECTS),
        ("발톱 변형", "발톱 무좀", RelType.SYMPTOM_OF),
        ("발톱 무좀", "발톱 관리", RelType.MANAGED_BY),

        # 당뇨
        ("고혈당", "당뇨", RelType.SYMPTOM_OF),
        ("저혈당", "당뇨", RelType.SYMPTOM_OF),
        ("갈증", "당뇨", RelType.SYMPTOM_OF),
        ("빈뇨", "당뇨", RelType.SYMPTOM_OF),
        ("당뇨", "혈당 모니터링", RelType.MANAGED_BY),
        ("당뇨", "식이 조절", RelType.MANAGED_BY),
        ("규칙적 운동", "당뇨", RelType.PREVENTS),

        # 갱년기
        ("안면홍조", "갱년기", RelType.SYMPTOM_OF),
        ("감정 기복", "갱년기", RelType.SYMPTOM_OF),
        ("갱년기", "호르몬 치료", RelType.MANAGED_BY),
        ("스트레스 관리", "갱년기", RelType.PREVENTS),

        # 구강
        ("잇몸 출혈", "치주 질환", RelType.SYMPTOM_OF),
        ("구취", "치주 질환", RelType.SYMPTOM_OF),
        ("치아 흔들림", "치주 질환", RelType.SYMPTOM_OF),
        ("치주 질환", "구강 위생", RelType.MANAGED_BY),
        ("치주 질환", "정기 치과 검진", RelType.MANAGED_BY),
        ("치주 질환", "구강", RelType.AFFECTS),

        # 난청
        ("소리 안 들림", "난청", RelType.SYMPTOM_OF),
        ("이명", "난청", RelType.SYMPTOM_OF),
        ("난청", "보청기", RelType.MANAGED_BY),
        ("난청", "청력 검사", RelType.MANAGED_BY),
        ("난청", "귀", RelType.AFFECTS),

        # 손발 저림
        ("손발 저림", "말초신경병증", RelType.SYMPTOM_OF),
        ("감각 이상", "말초신경병증", RelType.SYMPTOM_OF),
        ("당뇨", "말초신경병증", RelType.CAUSES),
        ("말초신경병증", "혈액 순환 개선", RelType.MANAGED_BY),
        ("말초신경병증", "비타민B 보충", RelType.MANAGED_BY),

        # 요실금
        ("소변 실수", "요실금", RelType.SYMPTOM_OF),
        ("빈뇨감", "요실금", RelType.SYMPTOM_OF),
        ("요실금", "골반저 운동", RelType.MANAGED_BY),
        ("요실금", "배뇨 훈련", RelType.MANAGED_BY),
        ("요실금", "방광", RelType.AFFECTS),

        # 탈모
        ("머리카락 빠짐", "탈모", RelType.SYMPTOM_OF),
        ("탈모", "두피 관리", RelType.MANAGED_BY),
        ("탈모", "영양 섭취", RelType.MANAGED_BY),
        ("탈모", "두피", RelType.AFFECTS),
        ("스트레스 관리", "탈모", RelType.PREVENTS),

        # 폐질환
        ("만성기침", "폐질환", RelType.SYMPTOM_OF),
        ("호흡곤란", "폐질환", RelType.SYMPTOM_OF),
        ("가래", "폐질환", RelType.SYMPTOM_OF),
        ("폐질환", "호흡 운동", RelType.MANAGED_BY),
        ("금연", "폐질환", RelType.PREVENTS),
        ("폐질환", "폐", RelType.AFFECTS),

        # 피부
        ("피부 가려움", "노인성 피부", RelType.SYMPTOM_OF),
        ("피부 건조", "노인성 피부", RelType.SYMPTOM_OF),
        ("노인성 피부", "보습제 사용", RelType.MANAGED_BY),
        ("노인성 피부", "미지근한 물 세안", RelType.MANAGED_BY),
        ("노인성 피부", "피부", RelType.AFFECTS),

        # 욕창
        ("피부 발적", "욕창", RelType.SYMPTOM_OF),
        ("욕창", "체위 변경", RelType.MANAGED_BY),
        ("욕창", "피부 보호", RelType.MANAGED_BY),
        ("욕창", "피부", RelType.AFFECTS),

        # 변비
        ("배변 곤란", "변비", RelType.SYMPTOM_OF),
        ("복부 팽만", "변비", RelType.SYMPTOM_OF),
        ("변비", "수분 섭취", RelType.MANAGED_BY),
        ("변비", "섬유질 섭취", RelType.MANAGED_BY),
        ("규칙적 운동", "변비", RelType.PREVENTS),
        ("변비", "장", RelType.AFFECTS),

        # 노안
        ("시력 저하", "노안", RelType.SYMPTOM_OF),
        ("근거리 흐림", "노안", RelType.SYMPTOM_OF),
        ("노안", "돋보기 사용", RelType.MANAGED_BY),
        ("노안", "안과 정기 검진", RelType.MANAGED_BY),
        ("노안", "눈", RelType.AFFECTS),

        # 식욕부진
        ("체중 감소", "식욕부진", RelType.SYMPTOM_OF),
        ("기력 저하", "식욕부진", RelType.SYMPTOM_OF),
        ("식욕부진", "소량 다회 식사", RelType.MANAGED_BY),
        ("식욕부진", "영양 보충제", RelType.MANAGED_BY),

        # 골다공증
        ("뼈 통증", "골다공증", RelType.SYMPTOM_OF),
        ("골다공증", "낙상 위험", RelType.CAUSES),
        ("골다공증", "칼슘 섭취", RelType.MANAGED_BY),
        ("골다공증", "낙상 예방", RelType.MANAGED_BY),
        ("규칙적 운동", "골다공증", RelType.PREVENTS),
        ("골다공증", "뼈", RelType.AFFECTS),
    ]

    for src, dst, rel_type in edges:
        neo4j_kg.merge_relationship(src, dst, rel_type)

    logger.info(f"🌱 기본 스키마 시드 완료: {len(nodes)}개 노드, {len(edges)}개 관계")
    return {"nodes": len(nodes), "edges": len(edges)}


def main():
    parser = argparse.ArgumentParser(
        description="Neo4j 지식그래프 자동 구축 파이프라인"
    )
    parser.add_argument(
        "--docs-dir",
        default="data/healthcare_docs",
        help="문서 폴더 경로 (기본: data/healthcare_docs)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="특정 파일만 처리",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 그래프 초기화 후 재구축",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="OpenAI GPT 사용 (기본: Ollama 로컬). .env의 OPENAI_API_KEY 자동 사용",
    )
    parser.add_argument(
        "--openai-key",
        default=None,
        help="OpenAI API 키 (미지정 시 .env의 OPENAI_API_KEY 사용)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="배치 크기 (기본: 3)",
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="기본 스키마 시드 건너뛰기",
    )
    args = parser.parse_args()

    docs_dir = project_root / args.docs_dir

    # .env 파일 로드 (OPENAI_API_KEY, Neo4j 설정 등)
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")

    # OpenAI 키: CLI 인자 > .env > 환경변수 순서
    openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if args.use_openai and not openai_key:
        print("❌ --use-openai 지정했으나 OPENAI_API_KEY가 없습니다.")
        print("   .env에 OPENAI_API_KEY=sk-xxx 를 설정하거나 --openai-key 를 지정하세요.")
        return

    print("=" * 60)
    print("🧠 Neo4j 지식그래프 자동 구축 파이프라인")
    print("=" * 60)
    print(f"  Neo4j URI : {settings.NEO4J_URI}")
    print(f"  문서 폴더  : {docs_dir}")
    print(f"  LLM      : {'OpenAI GPT-4o-mini' if args.use_openai else f'Ollama {settings.OLLAMA_MODEL}'}")
    print(f"  배치 크기  : {args.batch_size}")
    print(f"  초기화     : {'예' if args.reset else '아니오'}")
    print("=" * 60)

    # 1. Neo4j 연결
    from app.knowledge_graph.health_kg import get_neo4j_kg
    neo4j_kg = get_neo4j_kg()

    # 2. 초기화 (선택)
    if args.reset:
        neo4j_kg.clear_all()
        print("✅ 기존 그래프 초기화 완료")

    # 3. 기본 스키마 시드
    if not args.skip_seed:
        seed_stats = seed_base_knowledge(neo4j_kg)
        print(f"✅ 기본 스키마 시드 완료: {seed_stats['nodes']}개 노드, {seed_stats['edges']}개 관계")

    # 4. 문서 로드
    documents = load_documents(docs_dir, target_file=args.file)
    if not documents:
        print("❌ 로드할 문서가 없습니다.")
        return

    print(f"✅ 문서 로드 완료: {len(documents)}개 섹션")

    # 5. LLM + Transformer 초기화
    llm = create_llm(
        use_openai=args.use_openai,
        openai_key=openai_key,
    )
    transformer = create_graph_transformer(llm)

    # 6. 추출 + 적재
    start_time = time.time()
    stats = extract_and_load(
        documents=documents,
        transformer=transformer,
        neo4j_kg=neo4j_kg,
        batch_size=args.batch_size,
    )
    total_time = time.time() - start_time

    # 7. 결과 출력
    print("\n" + "=" * 60)
    print("📊 구축 결과")
    print("=" * 60)
    print(f"  처리 문서        : {stats['processed_docs']}개 섹션")
    print(f"  추출 실패        : {stats['failed_docs']}개")
    print(f"  GraphDocument   : {stats['graph_documents']}개")
    print(f"  Neo4j 총 노드    : {stats['total_nodes']}개")
    print(f"  Neo4j 총 관계    : {stats['total_edges']}개")
    print(f"  총 소요 시간      : {total_time:.1f}초")
    print("=" * 60)
    print(f"\n✅ 완료! Neo4j 브라우저에서 확인: http://localhost:7474")
    print(f"   Cypher 예시: MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")


if __name__ == "__main__":
    main()
