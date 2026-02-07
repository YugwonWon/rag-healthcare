"""
건강 도메인 지식그래프 (Health Knowledge Graph)

NetworkX 기반 방향성 그래프로
증상 → 질환 → 치료/관리 → 주의사항 관계를 모델링한다.

노드 타입: SYMPTOM, CONDITION, TREATMENT, BODY_PART, RISK_FACTOR
엣지 타입: INDICATES, TREATS, AFFECTS, CAUSES, PREVENTS
"""

from typing import Optional

import networkx as nx

from app.logger import get_logger

logger = get_logger(__name__)


class NodeType:
    SYMPTOM = "symptom"            # 증상
    CONDITION = "condition"        # 질환/상태
    TREATMENT = "treatment"        # 치료/관리법
    BODY_PART = "body_part"        # 신체 부위
    RISK_FACTOR = "risk_factor"    # 위험 요인
    MEDICATION = "medication"      # 약물
    LIFESTYLE = "lifestyle"        # 생활습관


class EdgeType:
    INDICATES = "indicates"        # 증상 → 질환 (이 증상은 이 질환을 시사한다)
    TREATS = "treats"              # 치료 → 질환 (이 치료가 이 질환에 효과적)
    AFFECTS = "affects"            # 질환 → 신체부위
    CAUSES = "causes"              # 위험요인 → 질환
    PREVENTS = "prevents"          # 생활습관 → 질환 (예방)
    SYMPTOM_OF = "symptom_of"      # 증상 → 질환
    MANAGED_BY = "managed_by"      # 질환 → 치료/관리
    RELATED_TO = "related_to"      # 일반 관계


class HealthKnowledgeGraph:
    """건강 도메인 지식그래프"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_knowledge_graph()
        logger.info(
            f"🧠 지식그래프 초기화: "
            f"노드={self.graph.number_of_nodes()}, "
            f"엣지={self.graph.number_of_edges()}"
        )

    def _add_node(self, name: str, node_type: str, **attrs):
        self.graph.add_node(name, type=node_type, **attrs)

    def _add_edge(self, src: str, dst: str, edge_type: str, **attrs):
        self.graph.add_edge(src, dst, type=edge_type, **attrs)

    def _build_knowledge_graph(self):
        """data/healthcare_docs/ 기반 지식그래프 구축"""

        # ═══════════════════════════════════════════
        # 수면 장애
        # ═══════════════════════════════════════════
        self._add_node("수면장애", NodeType.CONDITION, desc="고령자 수면 장애")
        self._add_node("불면증", NodeType.SYMPTOM, desc="잠들기 어렵거나 자주 깨는 증상")
        self._add_node("수면 패턴 변화", NodeType.SYMPTOM, desc="수면 주기 변화")
        self._add_node("주간 졸림", NodeType.SYMPTOM, desc="낮에 졸리는 증상")
        self._add_node("규칙적 수면 습관", NodeType.TREATMENT, desc="일정한 시간 취침/기상")
        self._add_node("수면 환경 개선", NodeType.TREATMENT, desc="어둡고 조용한 환경, 적정 온도")
        self._add_node("낮 활동량 증가", NodeType.TREATMENT, desc="낮에 적절한 운동과 활동")
        self._add_node("카페인 제한", NodeType.LIFESTYLE, desc="오후 카페인 섭취 제한")

        self._add_edge("불면증", "수면장애", EdgeType.SYMPTOM_OF)
        self._add_edge("수면 패턴 변화", "수면장애", EdgeType.SYMPTOM_OF)
        self._add_edge("주간 졸림", "수면장애", EdgeType.SYMPTOM_OF)
        self._add_edge("수면장애", "규칙적 수면 습관", EdgeType.MANAGED_BY)
        self._add_edge("수면장애", "수면 환경 개선", EdgeType.MANAGED_BY)
        self._add_edge("수면장애", "낮 활동량 증가", EdgeType.MANAGED_BY)
        self._add_edge("카페인 제한", "수면장애", EdgeType.PREVENTS)

        # ═══════════════════════════════════════════
        # 발톱 질환
        # ═══════════════════════════════════════════
        self._add_node("내향성 발톱", NodeType.CONDITION, desc="발톱이 살 안쪽으로 파고드는 질환 (족간증)")
        self._add_node("발톱 변형", NodeType.SYMPTOM, desc="발톱이 두꺼워지거나 휘는 증상")
        self._add_node("발톱 통증", NodeType.SYMPTOM, desc="발톱 주변 통증, 염증")
        self._add_node("발톱 관리", NodeType.TREATMENT, desc="올바른 발톱 깎기 (일자로)")
        self._add_node("편한 신발", NodeType.TREATMENT, desc="발에 맞는 편안한 신발 착용")
        self._add_node("발", NodeType.BODY_PART)
        self._add_node("발톱 무좀", NodeType.CONDITION, desc="발톱 곰팡이 감염")

        self._add_edge("발톱 변형", "내향성 발톱", EdgeType.SYMPTOM_OF)
        self._add_edge("발톱 통증", "내향성 발톱", EdgeType.SYMPTOM_OF)
        self._add_edge("내향성 발톱", "발톱 관리", EdgeType.MANAGED_BY)
        self._add_edge("내향성 발톱", "편한 신발", EdgeType.MANAGED_BY)
        self._add_edge("내향성 발톱", "발", EdgeType.AFFECTS)
        self._add_edge("발톱 변형", "발톱 무좀", EdgeType.SYMPTOM_OF)
        self._add_edge("발톱 무좀", "발톱 관리", EdgeType.MANAGED_BY)

        # ═══════════════════════════════════════════
        # 당뇨
        # ═══════════════════════════════════════════
        self._add_node("당뇨", NodeType.CONDITION, desc="혈당 조절 이상")
        self._add_node("고혈당", NodeType.SYMPTOM, desc="혈당이 높은 상태")
        self._add_node("저혈당", NodeType.SYMPTOM, desc="혈당이 낮아 어지러움")
        self._add_node("갈증", NodeType.SYMPTOM, desc="심한 갈증")
        self._add_node("빈뇨", NodeType.SYMPTOM, desc="소변을 자주 보는 증상")
        self._add_node("혈당 모니터링", NodeType.TREATMENT, desc="정기적 혈당 측정")
        self._add_node("식이 조절", NodeType.TREATMENT, desc="당분 제한, 균형 잡힌 식단")
        self._add_node("규칙적 운동", NodeType.LIFESTYLE, desc="가벼운 유산소 운동")

        self._add_edge("고혈당", "당뇨", EdgeType.SYMPTOM_OF)
        self._add_edge("저혈당", "당뇨", EdgeType.SYMPTOM_OF)
        self._add_edge("갈증", "당뇨", EdgeType.SYMPTOM_OF)
        self._add_edge("빈뇨", "당뇨", EdgeType.SYMPTOM_OF)
        self._add_edge("당뇨", "혈당 모니터링", EdgeType.MANAGED_BY)
        self._add_edge("당뇨", "식이 조절", EdgeType.MANAGED_BY)
        self._add_edge("규칙적 운동", "당뇨", EdgeType.PREVENTS)

        # ═══════════════════════════════════════════
        # 갱년기
        # ═══════════════════════════════════════════
        self._add_node("갱년기", NodeType.CONDITION, desc="호르몬 변화에 의한 증상")
        self._add_node("안면홍조", NodeType.SYMPTOM, desc="얼굴이 갑자기 달아오르는 증상")
        self._add_node("감정 기복", NodeType.SYMPTOM, desc="감정 변화가 심한 상태")
        self._add_node("호르몬 치료", NodeType.TREATMENT, desc="의사 처방 호르몬 요법")
        self._add_node("스트레스 관리", NodeType.LIFESTYLE, desc="명상, 이완 등")

        self._add_edge("안면홍조", "갱년기", EdgeType.SYMPTOM_OF)
        self._add_edge("감정 기복", "갱년기", EdgeType.SYMPTOM_OF)
        self._add_edge("갱년기", "호르몬 치료", EdgeType.MANAGED_BY)
        self._add_edge("스트레스 관리", "갱년기", EdgeType.PREVENTS)

        # ═══════════════════════════════════════════
        # 구강 관리
        # ═══════════════════════════════════════════
        self._add_node("치주 질환", NodeType.CONDITION, desc="잇몸 질환")
        self._add_node("잇몸 출혈", NodeType.SYMPTOM, desc="잇몸에서 피가 나는 증상")
        self._add_node("구취", NodeType.SYMPTOM, desc="입 냄새")
        self._add_node("치아 흔들림", NodeType.SYMPTOM, desc="치아가 흔들리는 증상")
        self._add_node("구강 위생", NodeType.TREATMENT, desc="올바른 칫솔질, 치실 사용")
        self._add_node("정기 치과 검진", NodeType.TREATMENT, desc="6개월마다 치과 방문")
        self._add_node("구강", NodeType.BODY_PART)

        self._add_edge("잇몸 출혈", "치주 질환", EdgeType.SYMPTOM_OF)
        self._add_edge("구취", "치주 질환", EdgeType.SYMPTOM_OF)
        self._add_edge("치아 흔들림", "치주 질환", EdgeType.SYMPTOM_OF)
        self._add_edge("치주 질환", "구강 위생", EdgeType.MANAGED_BY)
        self._add_edge("치주 질환", "정기 치과 검진", EdgeType.MANAGED_BY)
        self._add_edge("치주 질환", "구강", EdgeType.AFFECTS)

        # ═══════════════════════════════════════════
        # 난청
        # ═══════════════════════════════════════════
        self._add_node("난청", NodeType.CONDITION, desc="청력 저하")
        self._add_node("소리 안 들림", NodeType.SYMPTOM, desc="소리가 잘 안 들리는 증상")
        self._add_node("이명", NodeType.SYMPTOM, desc="귀에서 소리가 나는 증상")
        self._add_node("보청기", NodeType.TREATMENT, desc="보청기 착용")
        self._add_node("청력 검사", NodeType.TREATMENT, desc="정기 청력 검사")
        self._add_node("귀", NodeType.BODY_PART)

        self._add_edge("소리 안 들림", "난청", EdgeType.SYMPTOM_OF)
        self._add_edge("이명", "난청", EdgeType.SYMPTOM_OF)
        self._add_edge("난청", "보청기", EdgeType.MANAGED_BY)
        self._add_edge("난청", "청력 검사", EdgeType.MANAGED_BY)
        self._add_edge("난청", "귀", EdgeType.AFFECTS)

        # ═══════════════════════════════════════════
        # 손발 저림
        # ═══════════════════════════════════════════
        self._add_node("말초신경병증", NodeType.CONDITION, desc="말초 신경 손상")
        self._add_node("손발 저림", NodeType.SYMPTOM, desc="손이나 발이 저린 증상")
        self._add_node("감각 이상", NodeType.SYMPTOM, desc="감각이 둔해지는 증상")
        self._add_node("혈액 순환 개선", NodeType.TREATMENT, desc="가벼운 운동, 마사지")
        self._add_node("비타민B 보충", NodeType.TREATMENT, desc="비타민B12 등 보충")

        self._add_edge("손발 저림", "말초신경병증", EdgeType.SYMPTOM_OF)
        self._add_edge("감각 이상", "말초신경병증", EdgeType.SYMPTOM_OF)
        self._add_edge("당뇨", "말초신경병증", EdgeType.CAUSES)
        self._add_edge("말초신경병증", "혈액 순환 개선", EdgeType.MANAGED_BY)
        self._add_edge("말초신경병증", "비타민B 보충", EdgeType.MANAGED_BY)

        # ═══════════════════════════════════════════
        # 요실금
        # ═══════════════════════════════════════════
        self._add_node("요실금", NodeType.CONDITION, desc="소변 조절 어려움")
        self._add_node("소변 실수", NodeType.SYMPTOM, desc="의도치 않은 소변 배출")
        self._add_node("빈뇨감", NodeType.SYMPTOM, desc="소변을 자주 보고 싶은 느낌")
        self._add_node("골반저 운동", NodeType.TREATMENT, desc="케겔 운동 등")
        self._add_node("배뇨 훈련", NodeType.TREATMENT, desc="정해진 시간 배뇨 습관")
        self._add_node("방광", NodeType.BODY_PART)

        self._add_edge("소변 실수", "요실금", EdgeType.SYMPTOM_OF)
        self._add_edge("빈뇨감", "요실금", EdgeType.SYMPTOM_OF)
        self._add_edge("요실금", "골반저 운동", EdgeType.MANAGED_BY)
        self._add_edge("요실금", "배뇨 훈련", EdgeType.MANAGED_BY)
        self._add_edge("요실금", "방광", EdgeType.AFFECTS)

        # ═══════════════════════════════════════════
        # 탈모
        # ═══════════════════════════════════════════
        self._add_node("탈모", NodeType.CONDITION, desc="머리카락이 빠지는 상태")
        self._add_node("머리카락 빠짐", NodeType.SYMPTOM, desc="머리카락이 많이 빠지는 증상")
        self._add_node("두피 관리", NodeType.TREATMENT, desc="두피 청결, 마사지")
        self._add_node("영양 섭취", NodeType.TREATMENT, desc="단백질, 철분 등 영양소 보충")
        self._add_node("두피", NodeType.BODY_PART)

        self._add_edge("머리카락 빠짐", "탈모", EdgeType.SYMPTOM_OF)
        self._add_edge("탈모", "두피 관리", EdgeType.MANAGED_BY)
        self._add_edge("탈모", "영양 섭취", EdgeType.MANAGED_BY)
        self._add_edge("탈모", "두피", EdgeType.AFFECTS)
        self._add_edge("스트레스 관리", "탈모", EdgeType.PREVENTS)

        # ═══════════════════════════════════════════
        # 폐 질환
        # ═══════════════════════════════════════════
        self._add_node("폐질환", NodeType.CONDITION, desc="만성 폐쇄성 폐질환 등")
        self._add_node("만성기침", NodeType.SYMPTOM, desc="오래 지속되는 기침")
        self._add_node("호흡곤란", NodeType.SYMPTOM, desc="숨이 차는 증상")
        self._add_node("가래", NodeType.SYMPTOM, desc="가래가 많은 증상")
        self._add_node("호흡 운동", NodeType.TREATMENT, desc="복식 호흡, 입술 오므리기 호흡")
        self._add_node("금연", NodeType.LIFESTYLE, desc="흡연 중단")
        self._add_node("폐", NodeType.BODY_PART)

        self._add_edge("만성기침", "폐질환", EdgeType.SYMPTOM_OF)
        self._add_edge("호흡곤란", "폐질환", EdgeType.SYMPTOM_OF)
        self._add_edge("가래", "폐질환", EdgeType.SYMPTOM_OF)
        self._add_edge("폐질환", "호흡 운동", EdgeType.MANAGED_BY)
        self._add_edge("금연", "폐질환", EdgeType.PREVENTS)
        self._add_edge("폐질환", "폐", EdgeType.AFFECTS)

        # ═══════════════════════════════════════════
        # 피부 관리
        # ═══════════════════════════════════════════
        self._add_node("노인성 피부", NodeType.CONDITION, desc="고령자 피부 건조, 가려움")
        self._add_node("피부 가려움", NodeType.SYMPTOM, desc="피부가 가려운 증상")
        self._add_node("피부 건조", NodeType.SYMPTOM, desc="피부가 건조한 상태")
        self._add_node("보습제 사용", NodeType.TREATMENT, desc="보습크림 정기 도포")
        self._add_node("미지근한 물 세안", NodeType.TREATMENT, desc="뜨거운 물 대신 미지근한 물")
        self._add_node("피부", NodeType.BODY_PART)

        self._add_edge("피부 가려움", "노인성 피부", EdgeType.SYMPTOM_OF)
        self._add_edge("피부 건조", "노인성 피부", EdgeType.SYMPTOM_OF)
        self._add_edge("노인성 피부", "보습제 사용", EdgeType.MANAGED_BY)
        self._add_edge("노인성 피부", "미지근한 물 세안", EdgeType.MANAGED_BY)
        self._add_edge("노인성 피부", "피부", EdgeType.AFFECTS)

        # ═══════════════════════════════════════════
        # 욕창
        # ═══════════════════════════════════════════
        self._add_node("욕창", NodeType.CONDITION, desc="오래 누워있어 피부가 손상")
        self._add_node("피부 발적", NodeType.SYMPTOM, desc="피부가 빨갛게 되는 증상")
        self._add_node("체위 변경", NodeType.TREATMENT, desc="2시간마다 자세 바꾸기")
        self._add_node("피부 보호", NodeType.TREATMENT, desc="쿠션, 매트리스 사용")

        self._add_edge("피부 발적", "욕창", EdgeType.SYMPTOM_OF)
        self._add_edge("욕창", "체위 변경", EdgeType.MANAGED_BY)
        self._add_edge("욕창", "피부 보호", EdgeType.MANAGED_BY)
        self._add_edge("욕창", "피부", EdgeType.AFFECTS)

        # ═══════════════════════════════════════════
        # 변비
        # ═══════════════════════════════════════════
        self._add_node("변비", NodeType.CONDITION, desc="배변 어려움")
        self._add_node("배변 곤란", NodeType.SYMPTOM, desc="변을 보기 어려운 증상")
        self._add_node("복부 팽만", NodeType.SYMPTOM, desc="배가 더부룩한 증상")
        self._add_node("수분 섭취", NodeType.TREATMENT, desc="충분한 물 마시기")
        self._add_node("섬유질 섭취", NodeType.TREATMENT, desc="과일, 채소, 잡곡 섭취")
        self._add_node("장", NodeType.BODY_PART)

        self._add_edge("배변 곤란", "변비", EdgeType.SYMPTOM_OF)
        self._add_edge("복부 팽만", "변비", EdgeType.SYMPTOM_OF)
        self._add_edge("변비", "수분 섭취", EdgeType.MANAGED_BY)
        self._add_edge("변비", "섬유질 섭취", EdgeType.MANAGED_BY)
        self._add_edge("규칙적 운동", "변비", EdgeType.PREVENTS)
        self._add_edge("변비", "장", EdgeType.AFFECTS)

        # ═══════════════════════════════════════════
        # 노안
        # ═══════════════════════════════════════════
        self._add_node("노안", NodeType.CONDITION, desc="가까운 것이 잘 안 보이는 상태")
        self._add_node("시력 저하", NodeType.SYMPTOM, desc="눈이 침침해지는 증상")
        self._add_node("근거리 흐림", NodeType.SYMPTOM, desc="가까운 글씨가 잘 안 보임")
        self._add_node("돋보기 사용", NodeType.TREATMENT, desc="적절한 도수의 돋보기")
        self._add_node("안과 정기 검진", NodeType.TREATMENT, desc="정기 안과 검진")
        self._add_node("눈", NodeType.BODY_PART)

        self._add_edge("시력 저하", "노안", EdgeType.SYMPTOM_OF)
        self._add_edge("근거리 흐림", "노안", EdgeType.SYMPTOM_OF)
        self._add_edge("노안", "돋보기 사용", EdgeType.MANAGED_BY)
        self._add_edge("노안", "안과 정기 검진", EdgeType.MANAGED_BY)
        self._add_edge("노안", "눈", EdgeType.AFFECTS)

        # ═══════════════════════════════════════════
        # 식욕부진 / 저영양
        # ═══════════════════════════════════════════
        self._add_node("식욕부진", NodeType.CONDITION, desc="먹고 싶은 욕구 저하")
        self._add_node("체중 감소", NodeType.SYMPTOM, desc="의도치 않은 체중 감소")
        self._add_node("기력 저하", NodeType.SYMPTOM, desc="기운이 없는 상태")
        self._add_node("소량 다회 식사", NodeType.TREATMENT, desc="조금씩 자주 먹기")
        self._add_node("영양 보충제", NodeType.TREATMENT, desc="경구 영양 보충")

        self._add_edge("체중 감소", "식욕부진", EdgeType.SYMPTOM_OF)
        self._add_edge("기력 저하", "식욕부진", EdgeType.SYMPTOM_OF)
        self._add_edge("식욕부진", "소량 다회 식사", EdgeType.MANAGED_BY)
        self._add_edge("식욕부진", "영양 보충제", EdgeType.MANAGED_BY)

        # ═══════════════════════════════════════════
        # 골다공증
        # ═══════════════════════════════════════════
        self._add_node("골다공증", NodeType.CONDITION, desc="뼈가 약해지는 상태")
        self._add_node("뼈 통증", NodeType.SYMPTOM, desc="뼈 부위 통증")
        self._add_node("낙상 위험", NodeType.RISK_FACTOR, desc="넘어지기 쉬운 상태")
        self._add_node("칼슘 섭취", NodeType.TREATMENT, desc="칼슘 + 비타민D 보충")
        self._add_node("낙상 예방", NodeType.TREATMENT, desc="환경 정리, 미끄럼 방지")
        self._add_node("뼈", NodeType.BODY_PART)

        self._add_edge("뼈 통증", "골다공증", EdgeType.SYMPTOM_OF)
        self._add_edge("골다공증", "낙상 위험", EdgeType.CAUSES)
        self._add_edge("골다공증", "칼슘 섭취", EdgeType.MANAGED_BY)
        self._add_edge("골다공증", "낙상 예방", EdgeType.MANAGED_BY)
        self._add_edge("규칙적 운동", "골다공증", EdgeType.PREVENTS)
        self._add_edge("골다공증", "뼈", EdgeType.AFFECTS)

    # ═══════════════════════════════════════════
    # 검색 API
    # ═══════════════════════════════════════════

    def find_related_nodes(self, keyword: str, depth: int = 2) -> list[dict]:
        """
        키워드와 관련된 노드들을 BFS로 탐색한다.

        Args:
            keyword: 검색 키워드
            depth: 탐색 깊이 (기본 2홉)

        Returns:
            [{name, type, desc, relation, distance}, ...]
        """
        # 키워드와 매칭되는 노드 찾기
        matched_nodes = []
        for node, data in self.graph.nodes(data=True):
            if keyword in node or keyword in data.get("desc", ""):
                matched_nodes.append(node)

        if not matched_nodes:
            return []

        # BFS 탐색
        results = []
        visited = set()

        for start_node in matched_nodes:
            queue = [(start_node, 0)]
            visited.add(start_node)

            while queue:
                current, dist = queue.pop(0)
                node_data = self.graph.nodes[current]

                results.append({
                    "name": current,
                    "type": node_data.get("type", "unknown"),
                    "desc": node_data.get("desc", ""),
                    "distance": dist,
                })

                if dist < depth:
                    # 나가는 엣지 (successors)
                    for neighbor in self.graph.successors(current):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

                    # 들어오는 엣지 (predecessors)
                    for neighbor in self.graph.predecessors(current):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

        return results

    def get_condition_info(self, condition_name: str) -> dict:
        """
        질환명으로 증상, 치료법, 주의사항 등 종합 정보를 가져온다.
        """
        if condition_name not in self.graph:
            return {}

        info = {
            "condition": condition_name,
            "description": self.graph.nodes[condition_name].get("desc", ""),
            "symptoms": [],
            "treatments": [],
            "risk_factors": [],
            "prevention": [],
            "body_parts": [],
        }

        # 증상 (이 질환을 가리키는 SYMPTOM_OF 엣지)
        for pred in self.graph.predecessors(condition_name):
            edge_data = self.graph.edges[pred, condition_name]
            node_data = self.graph.nodes[pred]
            if edge_data.get("type") == EdgeType.SYMPTOM_OF:
                info["symptoms"].append({
                    "name": pred,
                    "desc": node_data.get("desc", ""),
                })
            elif edge_data.get("type") == EdgeType.CAUSES:
                info["risk_factors"].append({
                    "name": pred,
                    "desc": node_data.get("desc", ""),
                })
            elif edge_data.get("type") == EdgeType.PREVENTS:
                info["prevention"].append({
                    "name": pred,
                    "desc": node_data.get("desc", ""),
                })

        # 치료/관리 (이 질환에서 나가는 MANAGED_BY 엣지)
        for succ in self.graph.successors(condition_name):
            edge_data = self.graph.edges[condition_name, succ]
            node_data = self.graph.nodes[succ]
            if edge_data.get("type") == EdgeType.MANAGED_BY:
                info["treatments"].append({
                    "name": succ,
                    "desc": node_data.get("desc", ""),
                })
            elif edge_data.get("type") == EdgeType.AFFECTS:
                info["body_parts"].append(succ)

        return info

    def get_symptom_conditions(self, symptom: str) -> list[str]:
        """증상으로부터 가능한 질환 목록을 반환한다."""
        conditions = []
        if symptom not in self.graph:
            # 부분 매칭
            for node in self.graph.nodes:
                if symptom in node and self.graph.nodes[node].get("type") == NodeType.SYMPTOM:
                    for succ in self.graph.successors(node):
                        edge_data = self.graph.edges[node, succ]
                        if edge_data.get("type") == EdgeType.SYMPTOM_OF:
                            conditions.append(succ)
            return list(set(conditions))

        for succ in self.graph.successors(symptom):
            edge_data = self.graph.edges[symptom, succ]
            if edge_data.get("type") == EdgeType.SYMPTOM_OF:
                conditions.append(succ)
        return conditions


# 싱글톤
_health_kg: Optional[HealthKnowledgeGraph] = None


def get_health_kg() -> HealthKnowledgeGraph:
    global _health_kg
    if _health_kg is None:
        _health_kg = HealthKnowledgeGraph()
    return _health_kg
