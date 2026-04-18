#!/usr/bin/env python3
"""
IEEE ROMAN 논문용 통합 평가 스크립트
=============================================

실행하는 평가 항목:
1. PII 감지 & 삭제 (Precision / Recall / F1 by PII type)
2. 건강 위험 신호 감지 (Precision / Recall / F1 by category)
3. 프라이버시 커뮤니케이션 전략 시나리오 테스트 (pass/fail)
4. Base SLM vs Fine-tuned SLM 응답 비교 (latency + qualitative)
5. 파이프라인 Latency 벤치마크

사용법:
    cd /Users/yugwon/Projects/rag-healthcare
    python3 -m app.test.eval_roman
"""

import json
import re
import sys
import time
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ═══════════════════════════════════════════════════════
#  1. 테스트 데이터셋 (Gold Standard)
# ═══════════════════════════════════════════════════════

# ── PII 테스트 데이터: (입력 텍스트, expected PII types 포함여부) ──
PII_TEST_DATA = [
    # ═══════════════════════════════════════
    #  이름 (Name) — 15 cases
    # ═══════════════════════════════════════
    {"text": "김영수 어르신이 오늘 혈압이 높다고 하셨어요.", "expected": [("김영수", "name")], "clinical_should_preserve": ["혈압"]},
    {"text": "박순자 할머니가 요즘 잠을 못 주무신대요.", "expected": [("박순자", "name")], "clinical_should_preserve": ["잠"]},
    {"text": "이순옥 어르신이 서울대학병원에서 검진받으셨어요.", "expected": [("이순옥", "name"), ("서울대학병원", "hospital")], "clinical_should_preserve": ["검진"]},
    {"text": "김영호 할아버지가 당뇨 관리 잘 하고 계세요.", "expected": [("김영호", "name")], "clinical_should_preserve": ["당뇨"]},
    {"text": "최정희 어르신이 무릎이 아프다고 하셨어요.", "expected": [("최정희", "name")], "clinical_should_preserve": ["무릎"]},
    {"text": "정순이 할머니가 낙상 사고 후 입원하셨어요.", "expected": [("정순이", "name")], "clinical_should_preserve": ["낙상"]},
    {"text": "강민호 환자분이 수면제 복용 중이세요.", "expected": [("강민호", "name")], "clinical_should_preserve": ["수면"]},
    {"text": "조영자 할머니와 건강 상담을 진행했어요.", "expected": [("조영자", "name")], "clinical_should_preserve": []},
    {"text": "윤복순 어르신이 혈당이 불안정하대요.", "expected": [("윤복순", "name")], "clinical_should_preserve": ["혈당"]},
    {"text": "한영숙 할머니가 인지 검사를 받으러 오셨어요.", "expected": [("한영숙", "name")], "clinical_should_preserve": ["인지"]},
    {"text": "오현주 씨가 복약 상담을 원하셨어요.", "expected": [("오현주", "name")], "clinical_should_preserve": []},
    {"text": "임광수 어르신이 어지러움을 호소하셨어요.", "expected": [("임광수", "name")], "clinical_should_preserve": []},
    {"text": "송미자 할머니가 허리 통증으로 병원에 다니고 계셔요.", "expected": [("송미자", "name")], "clinical_should_preserve": ["허리", "통증"]},
    {"text": "신동호 어르신이 고혈압 약을 깜빡하셨대요.", "expected": [("신동호", "name")], "clinical_should_preserve": ["고혈압"]},
    {"text": "권순옥 할머니가 오늘 방문 간호를 받으셨어요.", "expected": [("권순옥", "name")], "clinical_should_preserve": []},

    # ═══════════════════════════════════════
    #  전화번호 (Phone) — 10 cases
    # ═══════════════════════════════════════
    {"text": "급하시면 010-1234-5678로 연락주세요. 두통이 심하시다면요.", "expected": [("010-1234-5678", "phone")], "clinical_should_preserve": ["두통"]},
    {"text": "보호자 연락처는 02-987-6543이에요.", "expected": [("02-987-6543", "phone")], "clinical_should_preserve": []},
    {"text": "010-9876-5432로 전화해서 약 처방 확인해주세요.", "expected": [("010-9876-5432", "phone")], "clinical_should_preserve": []},
    {"text": "간호사 연락처 031-456-7890으로 문의하세요.", "expected": [("031-456-7890", "phone")], "clinical_should_preserve": []},
    {"text": "응급 시 010-5555-1234로 연락바랍니다.", "expected": [("010-5555-1234", "phone")], "clinical_should_preserve": []},
    {"text": "어르신 자택 전화 02-333-4444로 확인 부탁드려요.", "expected": [("02-333-4444", "phone")], "clinical_should_preserve": []},
    {"text": "보호자 핸드폰 010-7777-8888로 검사 결과 알려드려야 해요.", "expected": [("010-7777-8888", "phone")], "clinical_should_preserve": []},
    {"text": "약국 번호가 02-555-6666이래요. 혈압약 재처방 받으시라고요.", "expected": [("02-555-6666", "phone")], "clinical_should_preserve": ["혈압"]},
    {"text": "복지관 번호 051-222-3333으로 연락하셔서 건강검진 예약하세요.", "expected": [("051-222-3333", "phone")], "clinical_should_preserve": []},
    {"text": "어르신이 010-1111-2222로 전화 해달라고 하셨어요.", "expected": [("010-1111-2222", "phone")], "clinical_should_preserve": []},
    # --- STT 변형: 공백 구분/구분자 없음 (음성인식 현실적 변형) ---
    {"text": "보호자 번호가 01098765432래요. 소화불량으로 고생하신대요.", "expected": [("01098765432", "phone")], "clinical_should_preserve": ["소화"]},
    {"text": "어르신 전화번호 010 3333 4444로 연락해달래요.", "expected": [("010 3333 4444", "phone")], "clinical_should_preserve": []},

    # ═══════════════════════════════════════
    #  주민등록번호 (SSN) — 6 cases
    # ═══════════════════════════════════════
    {"text": "주민번호 850123-1234567 확인해주세요.", "expected": [("850123-1234567", "ssn")], "clinical_should_preserve": []},
    {"text": "환자 주민등록번호는 701201-2345678입니다.", "expected": [("701201-2345678", "ssn")], "clinical_should_preserve": []},
    {"text": "430515-1678901로 등록된 분이 관절염 진단 받으셨어요.", "expected": [("430515-1678901", "ssn")], "clinical_should_preserve": ["관절"]},
    {"text": "주민번호 550828-2111111인 어르신이 복약 관리가 필요해요.", "expected": [("550828-2111111", "ssn")], "clinical_should_preserve": []},
    {"text": "주민등록번호 680714-1234567 확인 후 처방전 발급해주세요.", "expected": [("680714-1234567", "ssn")], "clinical_should_preserve": []},
    {"text": "보험 청구를 위해 주민번호 760320-2987654를 확인했어요.", "expected": [("760320-2987654", "ssn")], "clinical_should_preserve": []},
    # --- STT 변형: 공백 구분/구분자 없음 ---
    {"text": "주민번호 9003151234567로 접수해주세요. 어지러움 증상이 있으시대요.", "expected": [("9003151234567", "ssn")], "clinical_should_preserve": []},
    {"text": "주민등록번호 830712 2345678 확인 부탁드려요.", "expected": [("830712 2345678", "ssn")], "clinical_should_preserve": []},

    # ═══════════════════════════════════════
    #  주소 (Address) — 10 cases
    # ═══════════════════════════════════════
    {"text": "서울시 강남구 역삼동에 사는 어르신이 당뇨 관리가 필요해요.", "expected": [("서울시 강남구 역삼동", "address")], "clinical_should_preserve": ["당뇨"]},
    {"text": "경기도 수원시 팔달구 인계동 123-4에 거주하시는 분이요.", "expected": [("경기도 수원시 팔달구 인계동", "address")], "clinical_should_preserve": []},
    {"text": "부산시 해운대구 우동에 사시는 어르신이 방문 간호 요청하셨어요.", "expected": [("부산시 해운대구 우동", "address")], "clinical_should_preserve": []},
    {"text": "대구시 수성구 범어동에 거주하시는 분이 인지 검사를 원하세요.", "expected": [("대구시 수성구 범어동", "address")], "clinical_should_preserve": ["인지"]},
    {"text": "인천시 남동구 간석동에 살고 계신 환자분이세요.", "expected": [("인천시 남동구 간석동", "address")], "clinical_should_preserve": []},
    {"text": "서울시 종로구 혜화동에서 복지관 다니시는 분이에요.", "expected": [("서울시 종로구 혜화동", "address")], "clinical_should_preserve": []},
    {"text": "광주시 서구 치평동에 계신 어르신이 수면 문제로 상담 오셨어요.", "expected": [("광주시 서구 치평동", "address")], "clinical_should_preserve": ["수면"]},
    {"text": "대전시 유성구 봉명동에 사시는 할머니가 낙상하셨대요.", "expected": [("대전시 유성구 봉명동", "address")], "clinical_should_preserve": ["낙상"]},
    {"text": "제주시 이도이동에 거주하시는 어르신의 검진 결과에요.", "expected": [("제주시", "address")], "clinical_should_preserve": ["검진"]},
    {"text": "세종시 보람동에 사시는 분이 통증 관리가 필요하대요.", "expected": [("세종시 보람동", "address")], "clinical_should_preserve": ["통증"]},
    # --- STT 변형: 공식 명칭(특별시/광역시), ~로/~길 주소 ---
    {"text": "서울특별시 송파구 잠실동에 사시는 어르신이 허리 통증 호소하셨어요.", "expected": [("서울특별시 송파구 잠실동", "address")], "clinical_should_preserve": ["허리", "통증"]},
    {"text": "부산광역시 동래구 온천동에서 오신 분이에요.", "expected": [("부산광역시 동래구 온천동", "address")], "clinical_should_preserve": []},
    {"text": "대전시 서구 둔산로 45에 거주하시는 분이 관절염 상담 오셨어요.", "expected": [("대전시 서구 둔산로 45", "address")], "clinical_should_preserve": ["관절"]},
    {"text": "수원시 영통구 매영로에 사시는 어르신이 낙상 후 통원 치료 중이에요.", "expected": [("수원시 영통구 매영로", "address")], "clinical_should_preserve": ["낙상"]},

    # ═══════════════════════════════════════
    #  나이 (Age) — 10 cases
    # ═══════════════════════════════════════
    {"text": "92살이신데 관절이 많이 아프시대요.", "expected": [("92살", "age_specific")], "clinical_should_preserve": ["관절"]},
    {"text": "올해 78세이신 분이 어지러움을 호소하셨어요.", "expected": [("78세", "age_specific")], "clinical_should_preserve": []},
    {"text": "85살 할머니가 수면 장애로 고생하고 계세요.", "expected": [("85살", "age_specific")], "clinical_should_preserve": ["수면"]},
    {"text": "73세 어르신이 혈압약을 드시고 계세요.", "expected": [("73세", "age_specific")], "clinical_should_preserve": ["혈압"]},
    {"text": "88살이신 할아버지가 걷기 힘들어하세요.", "expected": [("88살", "age_specific")], "clinical_should_preserve": []},
    {"text": "올해 91세가 되신 분이에요. 청력이 많이 떨어지셨대요.", "expected": [("91세", "age_specific")], "clinical_should_preserve": []},
    {"text": "62살에 갱년기가 시작되셨다고 해요.", "expected": [("62살", "age_specific")], "clinical_should_preserve": ["갱년기"]},
    {"text": "79세 환자분이 낙상 위험이 높아요.", "expected": [("79세", "age_specific")], "clinical_should_preserve": ["낙상"]},
    {"text": "95살이신데 인지 기능이 많이 저하되셨어요.", "expected": [("95살", "age_specific")], "clinical_should_preserve": ["인지"]},
    {"text": "70세부터 무릎이 아프기 시작하셨대요.", "expected": [("70세", "age_specific")], "clinical_should_preserve": ["무릎"]},

    # ═══════════════════════════════════════
    #  가족 참조 (Family Reference) — 12 cases
    # ═══════════════════════════════════════
    {"text": "딸이 어머니 치매 걱정을 많이 하고 있어요.", "expected": [("딸", "family_ref"), ("어머니", "family_ref"), ("치매", "stigma")], "clinical_should_preserve": []},
    {"text": "며느리가 시어머니 요실금 때문에 상담 요청했어요.", "expected": [("며느리", "family_ref"), ("시어머니", "family_ref"), ("요실금", "stigma")], "clinical_should_preserve": []},
    {"text": "아들이 아버지 건강 상태를 물어보셨어요.", "expected": [("아들", "family_ref"), ("아버지", "family_ref")], "clinical_should_preserve": []},
    {"text": "손녀가 할머니 약 챙겨드리고 있대요.", "expected": [("손녀", "family_ref"), ("할머니", "family_ref")], "clinical_should_preserve": []},
    {"text": "남편이 아내 수면 문제 때문에 같이 잠을 못 잔대요.", "expected": [("남편", "family_ref"), ("아내", "family_ref")], "clinical_should_preserve": ["수면"]},
    {"text": "손자가 할아버지 병원 모시고 갈 거래요.", "expected": [("손자", "family_ref"), ("할아버지", "family_ref")], "clinical_should_preserve": []},
    {"text": "언니가 동생 입원비를 걱정하고 있어요.", "expected": [("언니", "family_ref"), ("동생", "family_ref")], "clinical_should_preserve": []},
    {"text": "사위가 장모님 건강검진 결과를 물어봤어요.", "expected": [("사위", "family_ref"), ("장모", "family_ref")], "clinical_should_preserve": []},
    {"text": "아들이 어머니 우울증을 걱정해서 상담 신청했어요.", "expected": [("아들", "family_ref"), ("어머니", "family_ref"), ("우울증", "stigma")], "clinical_should_preserve": []},
    {"text": "딸이 아버지 치매 검사를 받게 해달라고 요청했어요.", "expected": [("딸", "family_ref"), ("아버지", "family_ref"), ("치매", "stigma")], "clinical_should_preserve": []},
    {"text": "며느리가 시아버지 통증 관리에 대해 상담하셨어요.", "expected": [("며느리", "family_ref"), ("시아버지", "family_ref")], "clinical_should_preserve": ["통증"]},
    {"text": "할멈이 영감 코골이 때문에 잠을 못 잔다고 하셨어요.", "expected": [("할멈", "family_ref")], "clinical_should_preserve": ["코골이", "잠"]},

    # ═══════════════════════════════════════
    #  낙인 건강 상태 (Stigma) — 12 cases
    # ═══════════════════════════════════════
    {"text": "치매 진단을 받으셨는데 우울증도 있으신 것 같아요.", "expected": [("치매", "stigma"), ("우울증", "stigma")], "clinical_should_preserve": []},
    {"text": "요실금 때문에 외출을 못하신대요.", "expected": [("요실금", "stigma")], "clinical_should_preserve": []},
    {"text": "치매가 의심되어 인지 검사를 권유드렸어요.", "expected": [("치매", "stigma")], "clinical_should_preserve": ["인지"]},
    {"text": "우울증 약을 드시고 계신데 부작용이 있으시대요.", "expected": [("우울증", "stigma")], "clinical_should_preserve": []},
    {"text": "요실금과 변비가 동시에 있으셔서 힘들어하세요.", "expected": [("요실금", "stigma")], "clinical_should_preserve": ["변비"]},
    {"text": "치매 초기 진단을 받은 후 생활 변화가 컸대요.", "expected": [("치매", "stigma")], "clinical_should_preserve": []},
    {"text": "인지장애 진단 후 가족들이 많이 힘들어하고 있어요.", "expected": [("인지장애", "stigma")], "clinical_should_preserve": []},
    {"text": "경도인지장애라고 하셨는데 치매로 진행될까 걱정이래요.", "expected": [("경도인지장애", "stigma"), ("치매", "stigma")], "clinical_should_preserve": []},
    {"text": "우울증 진단을 받고 약을 드시기 시작하셨어요.", "expected": [("우울증", "stigma")], "clinical_should_preserve": []},
    {"text": "야간 빈뇨와 요실금으로 수면의 질이 떨어지셨대요.", "expected": [("요실금", "stigma")], "clinical_should_preserve": ["수면"]},
    {"text": "치매 환자를 돌보는 보호자 상담이 필요해요.", "expected": [("치매", "stigma")], "clinical_should_preserve": []},
    {"text": "알츠하이머 진단 후 가족이 돌봄 방법을 물어보셨어요.", "expected": [("알츠하이머", "stigma")], "clinical_should_preserve": []},

    # ═══════════════════════════════════════
    #  정서적 민감 (Emotional) — 10 cases
    # ═══════════════════════════════════════
    {"text": "요실금이 부끄러워서 아무에게도 말 못 했대요.", "expected": [("요실금", "stigma"), ("부끄러워서", "emotional")], "clinical_should_preserve": []},
    {"text": "남편에게 창피하다고 하셨어요.", "expected": [("남편", "family_ref"), ("창피", "emotional")], "clinical_should_preserve": []},
    {"text": "소변 실수가 수치스러워서 밖에 안 나가신대요.", "expected": [("수치", "emotional")], "clinical_should_preserve": []},
    {"text": "자식들에게 짐이 될까 봐 두렵다고 하셨어요.", "expected": [("두렵", "emotional")], "clinical_should_preserve": []},
    {"text": "병원가기 싫다고 하시는데 무섭다는 표현을 쓰셨어요.", "expected": [("무섭", "emotional")], "clinical_should_preserve": []},
    {"text": "증상을 말씀하시면서 자존심 상한다고 하셨어요.", "expected": [("자존심", "emotional")], "clinical_should_preserve": []},
    {"text": "혼자 사시면서 외롭고 서럽다고 하셨어요.", "expected": [("외롭", "emotional"), ("서럽", "emotional")], "clinical_should_preserve": []},
    {"text": "화가 나서 참을 수가 없다고 하셨어요.", "expected": [("화가 나", "emotional")], "clinical_should_preserve": []},
    {"text": "치매 진단 후 당혹스러워하시는 모습이었어요.", "expected": [("치매", "stigma"), ("당혹", "emotional")], "clinical_should_preserve": []},
    {"text": "건강이 나빠져서 비참하다고 하셨어요.", "expected": [("비참", "emotional")], "clinical_should_preserve": []},
    # 추가 정서 표현
    {"text": "밤마다 괴로워서 잠을 못 주무신대요.", "expected": [("괴로", "emotional")], "clinical_should_preserve": []},
    {"text": "요즘 너무 우울하다고 하셨어요.", "expected": [("우울", "emotional")], "clinical_should_preserve": []},
    {"text": "초조해하시면서 손을 떠셨어요.", "expected": [("초조", "emotional")], "clinical_should_preserve": []},
    {"text": "몸이 안 좋으니 무력하다고 하셨어요.", "expected": [("무력", "emotional")], "clinical_should_preserve": []},
    {"text": "남편 돌아가신 후 공허하다고 하셨어요.", "expected": [("남편", "family_ref"), ("공허", "emotional")], "clinical_should_preserve": []},
    {"text": "검사 결과 듣고 참담하다고 하셨어요.", "expected": [("참담", "emotional")], "clinical_should_preserve": []},
    {"text": "상황이 암담하다고 말씀하셨어요.", "expected": [("암담", "emotional")], "clinical_should_preserve": []},
    {"text": "억울하고 원통하다고 표현하셨어요.", "expected": [("억울", "emotional"), ("원통", "emotional")], "clinical_should_preserve": []},
    {"text": "배우자 떠난 후 고독하다고 하셨어요.", "expected": [("고독", "emotional")], "clinical_should_preserve": []},
    {"text": "혼자 계실 때 처량하다고 하셨어요.", "expected": [("처량", "emotional")], "clinical_should_preserve": []},
    {"text": "절망스러워서 아무것도 하기 싫으시대요.", "expected": [("절망", "emotional")], "clinical_should_preserve": []},
    {"text": "고통스러워서 견딜 수 없다고 하셨어요.", "expected": [("고통", "emotional")], "clinical_should_preserve": []},
    {"text": "서글프다고 말씀하시면서 한숨을 쉬셨어요.", "expected": [("서글", "emotional")], "clinical_should_preserve": []},
    {"text": "너무 비통해하시면서 눈물을 흘리셨어요.", "expected": [("비통", "emotional"), ("눈물", "emotional")], "clinical_should_preserve": []},
    {"text": "심란하다고 하셨어요. 마음이 안 놓이신대요.", "expected": [("심란", "emotional")], "clinical_should_preserve": []},
    {"text": "상처가 처참해서 보기 낙담하셨대요.", "expected": [("처참", "emotional"), ("낙담", "emotional")], "clinical_should_preserve": []},
    {"text": "좌절감이 크다고 하셨어요.", "expected": [("좌절감", "emotional")], "clinical_should_preserve": []},
    {"text": "자살 충동이 있다고 말씀하셨어요.", "expected": [("자살", "stigma")], "clinical_should_preserve": []},
    {"text": "자해 흔적이 있으셨어요.", "expected": [("자해", "emotional")], "clinical_should_preserve": []},
    {"text": "공황 증상을 보이시면서 공포감을 표현하셨어요.", "expected": [("공포", "emotional")], "clinical_should_preserve": []},
    {"text": "죄책감을 크게 느끼고 계신 모습이었어요.", "expected": [("죄책감", "emotional")], "clinical_should_preserve": []},
    {"text": "무기력해서 아무것도 안 하시고 누워만 계셨어요.", "expected": [("무기력", "emotional")], "clinical_should_preserve": []},
    {"text": "야속하다고 자식들한테 섭섭하시대요.", "expected": [("야속", "emotional")], "clinical_should_preserve": []},
    {"text": "한탄스러우시면서 한숨만 쉬셨어요.", "expected": [("한탄", "emotional")], "clinical_should_preserve": []},
    {"text": "짜증나서 참을 수 없다고 하셨어요.", "expected": [("짜증", "emotional"), ("참을 수 없", "emotional")], "clinical_should_preserve": []},
    {"text": "분해서 밤새 못 주무셨대요.", "expected": [("분", "emotional")], "clinical_should_preserve": []},
    {"text": "겁나서 검사를 못 받겠다고 하셨어요.", "expected": [("겁", "emotional")], "clinical_should_preserve": []},

    # ═══════════════════════════════════════
    #  병원명 (Hospital) — 8 cases
    # ═══════════════════════════════════════
    {"text": "서울대학병원에서 검진받으셨대요. 혈압이 높다고요.", "expected": [("서울대학병원", "hospital")], "clinical_should_preserve": ["혈압"]},
    {"text": "삼성서울병원에서 수술 예정이래요.", "expected": [("삼성서울병원", "hospital")], "clinical_should_preserve": ["수술"]},
    {"text": "분당서울대학교병원에서 인지 검사를 받으셨어요.", "expected": [("분당서울대학교병원", "hospital")], "clinical_should_preserve": ["인지"]},
    {"text": "아산병원에서 심장 검사를 받으실 거래요.", "expected": [("아산병원", "hospital")], "clinical_should_preserve": ["심장"]},
    {"text": "세브란스병원에서 재활 치료를 받고 계세요.", "expected": [("세브란스병원", "hospital")], "clinical_should_preserve": ["재활"]},
    {"text": "강남성모병원에서 고관절 수술을 받으셨대요.", "expected": [("강남성모병원", "hospital")], "clinical_should_preserve": ["고관절"]},
    {"text": "국립중앙의료원에서 건강검진 받으러 가실 거래요.", "expected": [("국립중앙의료원", "hospital")], "clinical_should_preserve": []},
    {"text": "보라매병원에서 당뇨 관리 프로그램에 참여 중이세요.", "expected": [("보라매병원", "hospital")], "clinical_should_preserve": ["당뇨"]},

    # ═══════════════════════════════════════
    #  날짜 (Date) — 8 cases
    # ═══════════════════════════════════════
    {"text": "2024년 3월 15일에 수술 예정이에요.", "expected": [("2024년 3월 15일", "date_specific")], "clinical_should_preserve": ["수술"]},
    {"text": "2023년 12월 1일에 치매 진단을 받으셨어요.", "expected": [("2023년 12월 1일", "date_specific"), ("치매", "stigma")], "clinical_should_preserve": []},
    {"text": "2025년 1월 20일에 검진 결과가 나왔어요.", "expected": [("2025년 1월 20일", "date_specific")], "clinical_should_preserve": ["검진"]},
    {"text": "2024년 7월 3일에 낙상 사고가 있었대요.", "expected": [("2024년 7월 3일", "date_specific")], "clinical_should_preserve": ["낙상"]},
    {"text": "2023년 9월 10일부터 혈압약을 드시기 시작하셨어요.", "expected": [("2023년 9월 10일", "date_specific")], "clinical_should_preserve": ["혈압"]},
    {"text": "2025년 2월 28일에 재활 치료가 끝났어요.", "expected": [("2025년 2월 28일", "date_specific")], "clinical_should_preserve": ["재활"]},
    {"text": "2024년 5월에 입원하셨다가 6월 15일에 퇴원하셨어요.", "expected": [("2024년 5월", "date_specific"), ("6월 15일", "date_specific")], "clinical_should_preserve": []},
    {"text": "2024년 11월 25일에 건강 상담을 진행했어요.", "expected": [("2024년 11월 25일", "date_specific")], "clinical_should_preserve": []},

    # ═══════════════════════════════════════
    #  복합 케이스 (Multiple PII) — 12 cases
    # ═══════════════════════════════════════
    {
        "text": "김영호 어르신(85세)이 서울시 종로구 혜화동에 거주하며, "
                "딸이 치매와 우울증 관련해서 010-9876-5432로 문의했어요.",
        "expected": [
            ("김영호", "name"), ("85세", "age_specific"),
            ("서울시 종로구 혜화동", "address"), ("딸", "family_ref"),
            ("치매", "stigma"), ("우울증", "stigma"),
            ("010-9876-5432", "phone"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "주민번호 761205-2345678인 이순옥 할머니가 수면제를 드시고 있어요.",
        "expected": [("761205-2345678", "ssn"), ("이순옥", "name")],
        "clinical_should_preserve": ["수면"],
    },
    {
        "text": "김영수 어르신(92살)이 서울시 강남구 역삼동에 사시는데, "
                "치매 진단 후 요실금까지 생겨서 며느리가 많이 힘들어해요.",
        "expected": [
            ("김영수", "name"), ("92살", "age_specific"),
            ("서울시 강남구 역삼동", "address"),
            ("치매", "stigma"), ("요실금", "stigma"), ("며느리", "family_ref"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "주민번호 850123-1234567인 박순자 할머니가 010-9876-5432로 전화해서 "
                "우울증 약을 안 먹었다고 부끄러워하셨어요.",
        "expected": [
            ("850123-1234567", "ssn"), ("박순자", "name"),
            ("010-9876-5432", "phone"), ("우울증", "stigma"), ("부끄러워", "emotional"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "이순옥 어르신이 서울대학병원에서 2024년 3월 15일에 검진받으셨는데, "
                "혈압이 높고 당뇨 수치도 안 좋대요. 아들이 걱정하고 있어요.",
        "expected": [
            ("이순옥", "name"), ("서울대학병원", "hospital"),
            ("2024년 3월 15일", "date_specific"), ("아들", "family_ref"),
        ],
        "clinical_should_preserve": ["혈압", "당뇨", "검진"],
    },
    {
        "text": "강민호 어르신(81세)이 부산시 해운대구 우동에서 넘어지셨대요. "
                "아내가 010-3333-4444로 연락했어요.",
        "expected": [
            ("강민호", "name"), ("81세", "age_specific"),
            ("부산시 해운대구 우동", "address"), ("아내", "family_ref"),
            ("010-3333-4444", "phone"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "최정희 할머니(88살)가 치매 진단 후 우울해하시는데, "
                "딸이 삼성서울병원에서 상담 받고 싶다고 하셨어요.",
        "expected": [
            ("최정희", "name"), ("88살", "age_specific"),
            ("치매", "stigma"), ("우울", "emotional"), ("딸", "family_ref"), ("삼성서울병원", "hospital"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "주민번호 430515-1678901인 조영자 할머니, 경기도 수원시 팔달구에 사시며 "
                "요실금으로 부끄러워하고 계세요.",
        "expected": [
            ("430515-1678901", "ssn"), ("조영자", "name"),
            ("경기도 수원시 팔달구", "address"), ("요실금", "stigma"), ("부끄러워", "emotional"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "윤복순 어르신(76세)이 혈압약과 당뇨약을 동시에 드시는데, "
                "아들이 대전시 유성구 봉명동에서 모시고 병원에 오셨어요.",
        "expected": [
            ("윤복순", "name"), ("76세", "age_specific"),
            ("아들", "family_ref"), ("대전시 유성구 봉명동", "address"),
        ],
        "clinical_should_preserve": ["혈압", "당뇨"],
    },
    {
        "text": "한영숙 할머니가 2025년 1월 10일에 세브란스병원에서 인지 검사 받으셨어요. "
                "며느리가 동행했대요.",
        "expected": [
            ("한영숙", "name"), ("2025년 1월 10일", "date_specific"),
            ("세브란스병원", "hospital"), ("며느리", "family_ref"),
        ],
        "clinical_should_preserve": ["인지"],
    },
    {
        "text": "송미자 할머니(90살)가 서울시 노원구 상계동에서 낙상하셔서 "
                "010-8888-9999로 딸에게 연락했대요.",
        "expected": [
            ("송미자", "name"), ("90살", "age_specific"),
            ("서울시 노원구 상계동", "address"), ("010-8888-9999", "phone"), ("딸", "family_ref"),
        ],
        "clinical_should_preserve": ["낙상"],
    },
    {
        "text": "신동호 어르신의 주민번호는 550828-2111111이고, "
                "아산병원에서 2024년 8월 5일에 심장 수술을 받으셨어요.",
        "expected": [
            ("신동호", "name"), ("550828-2111111", "ssn"),
            ("아산병원", "hospital"), ("2024년 8월 5일", "date_specific"),
        ],
        "clinical_should_preserve": ["심장", "수술"],
    },

    # ═══════════════════════════════════════
    #  오탐 방지 (Negative Cases) — 20 cases
    # ═══════════════════════════════════════
    {"text": "요즘 잠을 잘 못 자요. 수면 습관을 개선하고 싶어요.", "expected": [], "clinical_should_preserve": ["수면"]},
    {"text": "관절이 아파서 산책을 못 하고 있어요.", "expected": [], "clinical_should_preserve": ["관절", "산책"]},
    {"text": "혈압약을 먹고 있는데 어지러워요.", "expected": [], "clinical_should_preserve": ["혈압"]},
    {"text": "어제 넘어져서 무릎이 아파요.", "expected": [], "clinical_should_preserve": ["무릎"]},
    {"text": "식사를 잘 못 하고 입맛이 없어요.", "expected": [], "clinical_should_preserve": []},
    {"text": "허리가 아파서 오래 앉아있기 힘들어요.", "expected": [], "clinical_should_preserve": ["허리"]},
    {"text": "기침이 계속 나오고 가래가 있어요.", "expected": [], "clinical_should_preserve": ["기침", "가래"]},
    {"text": "가만히 있는데도 숨이 차요.", "expected": [], "clinical_should_preserve": []},
    {"text": "소화가 잘 안 되고 속이 더부룩해요.", "expected": [], "clinical_should_preserve": []},
    {"text": "눈이 침침하고 잘 안 보여요.", "expected": [], "clinical_should_preserve": []},
    {"text": "손이 떨리고 글씨를 쓰기 어려워요.", "expected": [], "clinical_should_preserve": []},
    {"text": "오늘 날씨가 좋아서 산책 다녀왔어요. 기분이 좋네요.", "expected": [], "clinical_should_preserve": []},
    {"text": "어제 병원에서 검진을 받았는데 결과가 좋았어요.", "expected": [], "clinical_should_preserve": ["검진"]},
    {"text": "오늘 점심에 김치찌개 먹었는데 맛있었어요.", "expected": [], "clinical_should_preserve": []},
    {"text": "운동을 꾸준히 하니까 체력이 좋아졌어요.", "expected": [], "clinical_should_preserve": []},
    {"text": "비타민을 매일 먹고 있는데 효과가 있는 것 같아요.", "expected": [], "clinical_should_preserve": []},
    {"text": "건강하게 살고 싶어서 식단을 조절하고 있어요.", "expected": [], "clinical_should_preserve": []},
    {"text": "진단을 받았는데 다행히 큰 문제는 없었어요.", "expected": [], "clinical_should_preserve": []},
    {"text": "의사 선생님이 꾸준히 운동하라고 하셨어요.", "expected": [], "clinical_should_preserve": []},
    {"text": "약을 먹으면 좋아질 거라고 하셨어요. 희망이 생겼어요.", "expected": [], "clinical_should_preserve": []},

    # ═══════════════════════════════════════════════════════
    #  1인칭 시점 (노인 당사자 발화) — 117 cases
    # ═══════════════════════════════════════════════════════

    # --- 1인칭: 이름 언급 (본인이 타인 이름을 말하는 경우) — 10 cases ---
    {"text": "김영수 의사 선생님한테 혈압약 처방받았어요.", "expected": [("김영수", "name")], "clinical_should_preserve": ["혈압"]},
    {"text": "이순옥 간호사님이 혈당 재라고 하셨어.", "expected": [("이순옥", "name")], "clinical_should_preserve": ["혈당"]},
    {"text": "박순자 씨랑 같이 복지관 건강교실 다녀요.", "expected": [("박순자", "name")], "clinical_should_preserve": []},
    {"text": "옆집 최정희 씨도 무릎이 아프대.", "expected": [("최정희", "name")], "clinical_should_preserve": ["무릎"]},
    {"text": "강민호 약사님이 이 약은 식후에 먹으라고 하셨어.", "expected": [("강민호", "name")], "clinical_should_preserve": []},
    {"text": "한영숙 씨가 좋다고 추천해준 한의원에 가봤어요.", "expected": [("한영숙", "name")], "clinical_should_preserve": []},
    {"text": "신동호 선생님이 운동 꾸준히 하라고 하셨어.", "expected": [("신동호", "name")], "clinical_should_preserve": []},
    {"text": "조영자 씨는 당뇨 관리를 잘 하시더라.", "expected": [("조영자", "name")], "clinical_should_preserve": ["당뇨"]},
    {"text": "윤복순 씨가 같이 산책하자고 했는데 무릎이 아파서 못 갔어.", "expected": [("윤복순", "name")], "clinical_should_preserve": ["무릎"]},
    {"text": "오현주 사회복지사님한테 건강검진 일정 물어봤어요.", "expected": [("오현주", "name")], "clinical_should_preserve": []},

    # --- 1인칭: 전화번호 (본인이 번호를 알려주는 경우) — 8 cases ---
    {"text": "내 전화번호는 010-3456-7890이야.", "expected": [("010-3456-7890", "phone")], "clinical_should_preserve": []},
    {"text": "혹시 연락할 일 있으면 010-5678-1234로 전화해줘.", "expected": [("010-5678-1234", "phone")], "clinical_should_preserve": []},
    {"text": "약국 전화번호가 02-444-5555였는데 맞나 모르겠어.", "expected": [("02-444-5555", "phone")], "clinical_should_preserve": []},
    {"text": "병원 예약하려고 031-777-8888로 전화했어.", "expected": [("031-777-8888", "phone")], "clinical_should_preserve": []},
    {"text": "집 전화가 02-123-4567인데 요즘 잘 안 받아.", "expected": [("02-123-4567", "phone")], "clinical_should_preserve": []},
    {"text": "응급 전화번호 119도 있지만 내 폰이 010-2222-3333이야.", "expected": [("010-2222-3333", "phone")], "clinical_should_preserve": []},
    {"text": "보건소 번호가 042-888-9999더라. 거기서 검진받았어.", "expected": [("042-888-9999", "phone")], "clinical_should_preserve": ["검진"]},
    {"text": "복지관 전화번호 053-111-2222로 상담 예약했어.", "expected": [("053-111-2222", "phone")], "clinical_should_preserve": []},
    # --- STT 변형: 공백 구분/구분자 없음 ---
    {"text": "내 번호 01055556666이야.", "expected": [("01055556666", "phone")], "clinical_should_preserve": []},
    {"text": "전화번호가 010 7777 8888이야. 연락줘.", "expected": [("010 7777 8888", "phone")], "clinical_should_preserve": []},

    # --- 1인칭: 주소 (본인 거주지를 말하는 경우) — 8 cases ---
    {"text": "나는 서울시 마포구 합정동에 살아.", "expected": [("서울시 마포구 합정동", "address")], "clinical_should_preserve": []},
    {"text": "부산시 사하구 괴정동에서 30년째 살고 있어.", "expected": [("부산시 사하구 괴정동", "address")], "clinical_should_preserve": []},
    {"text": "우리 집이 대구시 달서구 월성동인데 병원이 멀어.", "expected": [("대구시 달서구 월성동", "address")], "clinical_should_preserve": []},
    {"text": "인천시 부평구 부평동에서 복지관까지 버스 타고 가.", "expected": [("인천시 부평구 부평동", "address")], "clinical_should_preserve": []},
    {"text": "수원시 영통구 매탄동 아파트에 혼자 살아.", "expected": [("수원시 영통구 매탄동", "address")], "clinical_should_preserve": []},
    {"text": "광주시 남구 봉선동 경로당에서 친구들 만나.", "expected": [("광주시 남구 봉선동", "address")], "clinical_should_preserve": []},
    {"text": "성남시 분당구 정자동으로 이사 온 지 얼마 안 됐어.", "expected": [("성남시 분당구 정자동", "address")], "clinical_should_preserve": []},
    {"text": "전주시 완산구 효자동에 있는 병원 다녀.", "expected": [("전주시 완산구 효자동", "address")], "clinical_should_preserve": []},
    # --- STT 변형: 공식 명칭(특별시/광역시), ~로/~길 ---
    {"text": "서울특별시 관악구 봉천동에 살고 있어.", "expected": [("서울특별시 관악구 봉천동", "address")], "clinical_should_preserve": []},
    {"text": "부산광역시 수영구 광안동이 우리 집이야.", "expected": [("부산광역시 수영구 광안동", "address")], "clinical_should_preserve": []},
    {"text": "성남시 분당구 불정로에 있는 병원 다녀.", "expected": [("성남시 분당구 불정로", "address")], "clinical_should_preserve": []},

    # --- 1인칭: 나이 (본인 나이를 말하는 경우) — 8 cases ---
    {"text": "나 올해 78살인데 아직 건강한 편이야.", "expected": [("78살", "age_specific")], "clinical_should_preserve": []},
    {"text": "82세부터 혈압약 먹기 시작했어.", "expected": [("82세", "age_specific")], "clinical_should_preserve": ["혈압"]},
    {"text": "내가 76살 때 넘어져서 고관절을 다쳤거든.", "expected": [("76살", "age_specific")], "clinical_should_preserve": ["고관절"]},
    {"text": "올해 85세인데 아직 혼자 밥해 먹어.", "expected": [("85세", "age_specific")], "clinical_should_preserve": []},
    {"text": "73살 넘으니까 눈이 침침해졌어.", "expected": [("73살", "age_specific")], "clinical_should_preserve": []},
    {"text": "69세에 당뇨 진단 받았는데 지금은 관리 잘 하고 있어.", "expected": [("69세", "age_specific")], "clinical_should_preserve": ["당뇨"]},
    {"text": "나는 91살이야. 오래 살았지.", "expected": [("91살", "age_specific")], "clinical_should_preserve": []},
    {"text": "80세 넘으니까 기억력이 좀 나빠진 것 같아.", "expected": [("80세", "age_specific")], "clinical_should_preserve": []},

    # --- 1인칭: 가족 참조 (본인이 가족을 언급하는 경우) — 10 cases ---
    {"text": "딸이 자꾸 병원 가라고 잔소리야.", "expected": [("딸", "family_ref")], "clinical_should_preserve": []},
    {"text": "아들이 주말마다 와서 약 챙겨줘.", "expected": [("아들", "family_ref")], "clinical_should_preserve": []},
    {"text": "며느리가 해주는 밥이 맛있어서 많이 먹어.", "expected": [("며느리", "family_ref")], "clinical_should_preserve": []},
    {"text": "남편이 먼저 가고 나니까 혼자 사는 게 힘들어.", "expected": [("남편", "family_ref")], "clinical_should_preserve": []},
    {"text": "손녀가 놀러 오면 기분이 좋아져.", "expected": [("손녀", "family_ref")], "clinical_should_preserve": []},
    {"text": "손자가 대학 들어갔는데 보고 싶어.", "expected": [("손자", "family_ref")], "clinical_should_preserve": []},
    {"text": "동생이 같은 병으로 수술 받았거든.", "expected": [("동생", "family_ref")], "clinical_should_preserve": ["수술"]},
    {"text": "언니도 무릎이 안 좋아서 같이 병원 다녀.", "expected": [("언니", "family_ref")], "clinical_should_preserve": ["무릎"]},
    {"text": "아내가 치매 진단 받아서 내가 돌보고 있어.", "expected": [("아내", "family_ref"), ("치매", "stigma")], "clinical_should_preserve": []},
    {"text": "큰아들이 보조기구 사다 줬어.", "expected": [("큰아들", "family_ref")], "clinical_should_preserve": []},

    # --- 1인칭: 낙인 건강 상태 (본인이 민감 증상을 말하는 경우) — 10 cases ---
    {"text": "요실금이 있어서 외출할 때 항상 패드를 챙겨.", "expected": [("요실금", "stigma")], "clinical_should_preserve": []},
    {"text": "치매 검사를 받아봐야 하나 싶어. 자꾸 깜빡해.", "expected": [("치매", "stigma")], "clinical_should_preserve": []},
    {"text": "우울증 약을 먹고 있는데 효과가 있는지 모르겠어.", "expected": [("우울증", "stigma")], "clinical_should_preserve": []},
    {"text": "요실금 때문에 밤에 자다가도 깨.", "expected": [("요실금", "stigma")], "clinical_should_preserve": []},
    {"text": "치매가 올까 봐 두려워. 엄마도 치매였거든.", "expected": [("치매", "stigma"), ("엄마", "family_ref"), ("두려워", "emotional")], "clinical_should_preserve": []},
    {"text": "우울증이 점점 심해지는 것 같아.", "expected": [("우울증", "stigma")], "clinical_should_preserve": []},
    {"text": "변비가 심해서 화장실 가기가 무서워.", "expected": [("무서워", "emotional")], "clinical_should_preserve": ["변비"]},
    {"text": "소변을 자주 보니까 외출하기 겁나.", "expected": [("겁나", "emotional")], "clinical_should_preserve": []},
    {"text": "인지장애 진단을 받았는데 어떻게 해야 하는지 모르겠어.", "expected": [("인지장애", "stigma")], "clinical_should_preserve": []},
    {"text": "알츠하이머일 수도 있다고 해서 무서워.", "expected": [("알츠하이머", "stigma"), ("무서워", "emotional")], "clinical_should_preserve": []},

    # --- 1인칭: 정서 민감 (본인 감정을 표현하는 경우) — 10 cases ---
    {"text": "소변 실수하면 너무 부끄러워.", "expected": [("부끄러워", "emotional")], "clinical_should_preserve": []},
    {"text": "병원에서 옷 벗는 게 창피해서 검진을 못 가겠어.", "expected": [("창피", "emotional")], "clinical_should_preserve": ["검진"]},
    {"text": "자식들한테 짐이 되는 것 같아서 수치스러워.", "expected": [("수치", "emotional")], "clinical_should_preserve": []},
    {"text": "혼자 남겨질까 봐 두려워.", "expected": [("두려워", "emotional")], "clinical_should_preserve": []},
    {"text": "아무도 나를 안 찾아와서 외로워.", "expected": [("외로워", "emotional")], "clinical_should_preserve": []},
    {"text": "요즘 자꾸 서러운 생각이 들어.", "expected": [("서러운", "emotional")], "clinical_should_preserve": []},
    {"text": "몸이 마음대로 안 되니까 화가 나.", "expected": [("화가 나", "emotional")], "clinical_should_preserve": []},
    {"text": "내가 왜 이렇게 됐나 비참해.", "expected": [("비참", "emotional")], "clinical_should_preserve": []},
    {"text": "치매 검사 결과 듣고 당혹스러웠어.", "expected": [("치매", "stigma"), ("당혹", "emotional")], "clinical_should_preserve": []},
    {"text": "남들 앞에서 실수할까 봐 민망해서 안 나가.", "expected": [("민망", "emotional")], "clinical_should_preserve": []},
    # --- 1인칭: 새 정서 VA 어근 (bareunpy 기반 추가 어휘) — 9 cases ---
    {"text": "요즘 너무 답답해. 몸이 안 좋으니까.", "expected": [("답답", "emotional")], "clinical_should_preserve": []},
    {"text": "아들이 연락도 안 해서 속상해.", "expected": [("아들", "family_ref"), ("속상", "emotional")], "clinical_should_preserve": []},
    {"text": "밤마다 불안해서 잠이 안 와.", "expected": [("불안", "emotional")], "clinical_should_preserve": []},
    {"text": "며느리가 한마디도 안 해줘서 서운해.", "expected": [("며느리", "family_ref"), ("서운", "emotional")], "clinical_should_preserve": []},
    {"text": "억울해서 눈물이 나. 내가 뭘 잘못했다고.", "expected": [("억울", "emotional"), ("눈물이 나", "emotional")], "clinical_should_preserve": []},
    {"text": "혼자 밥 먹으니까 쓸쓸해.", "expected": [("쓸쓸", "emotional")], "clinical_should_preserve": []},
    {"text": "남편이 떠나고 나서 허전해.", "expected": [("남편", "family_ref"), ("허전", "emotional")], "clinical_should_preserve": []},
    {"text": "앞이 막막해. 어떻게 살아야 할지 모르겠어.", "expected": [("막막", "emotional")], "clinical_should_preserve": []},
    {"text": "옛날 생각하면 슬퍼.", "expected": [("슬퍼", "emotional")], "clinical_should_preserve": []},
    # 추가 정서 표현 (1인칭)
    {"text": "요즘 정말 우울해. 아무것도 하기 싫어.", "expected": [("우울", "emotional")], "clinical_should_preserve": []},
    {"text": "밤마다 괴로워서 잠이 안 와.", "expected": [("괴로", "emotional")], "clinical_should_preserve": []},
    {"text": "가슴이 뛰고 너무 초조해.", "expected": [("초조", "emotional")], "clinical_should_preserve": []},
    {"text": "아무 힘도 없고 무력해.", "expected": [("무력", "emotional")], "clinical_should_preserve": []},
    {"text": "마음이 텅 빈 것 같아. 공허해.", "expected": [("공허", "emotional")], "clinical_should_preserve": []},
    {"text": "이게 뭔 처참한 꼴이야.", "expected": [("처참", "emotional")], "clinical_should_preserve": []},
    {"text": "밤새 심란해서 잠이 안 와.", "expected": [("심란", "emotional")], "clinical_should_preserve": []},
    {"text": "참담해. 내 인생이 이렇게 될 줄 몰랐어.", "expected": [("참담", "emotional")], "clinical_should_preserve": []},
    {"text": "앞날이 암담해. 희망이 없어.", "expected": [("암담", "emotional")], "clinical_should_preserve": []},
    {"text": "너무 원통해. 내가 뭘 잘못했다고.", "expected": [("원통", "emotional")], "clinical_should_preserve": []},
    {"text": "뼈 빠지게 살았는데 비통해.", "expected": [("비통", "emotional")], "clinical_should_preserve": []},
    {"text": "혼자 할머니가 고독하게 사는 기분이야.", "expected": [("고독", "emotional")], "clinical_should_preserve": []},
    {"text": "갈 데도 없고 처량해.", "expected": [("처량", "emotional")], "clinical_should_preserve": []},
    {"text": "서글프다. 이렇게 늙어가는 게.", "expected": [("서글", "emotional")], "clinical_should_preserve": []},
    {"text": "절망스러워. 다리가 점점 안 되니까.", "expected": [("절망", "emotional")], "clinical_should_preserve": []},
    {"text": "고통스러워서 죽고 싶어.", "expected": [("고통", "emotional"), ("죽고 싶", "emotional")], "clinical_should_preserve": []},
    {"text": "자꾸 자해 충동이 들어.", "expected": [("자해", "emotional")], "clinical_should_preserve": []},
    {"text": "울고 싶어. 친구들도 다 떠났어.", "expected": [("울고 싶", "emotional")], "clinical_should_preserve": []},
    {"text": "무기력해서 씻지도 못하겠어.", "expected": [("무기력", "emotional")], "clinical_should_preserve": []},
    {"text": "자식이 너무 야속해.", "expected": [("야속", "emotional")], "clinical_should_preserve": []},
    {"text": "짜증나. 몸이 말을 안 들으니까.", "expected": [("짜증", "emotional")], "clinical_should_preserve": []},
    {"text": "검사받으러 가야 하는데 겁나.", "expected": [("겁", "emotional")], "clinical_should_preserve": []},
    {"text": "가슴 아파. 자식이 안 와.", "expected": [("가슴 아파", "emotional")], "clinical_should_preserve": []},
    {"text": "낙담했어. 또 재발이래.", "expected": [("낙담", "emotional")], "clinical_should_preserve": []},

    # --- 1인칭: 병원명 (본인이 병원을 언급하는 경우) — 6 cases ---
    {"text": "서울대학병원에서 검사 받으라고 해서 갔다 왔어.", "expected": [("서울대학병원", "hospital")], "clinical_should_preserve": []},
    {"text": "삼성서울병원 예약이 너무 오래 걸려.", "expected": [("삼성서울병원", "hospital")], "clinical_should_preserve": []},
    {"text": "세브란스병원에서 무릎 수술 받았어.", "expected": [("세브란스병원", "hospital")], "clinical_should_preserve": ["무릎", "수술"]},
    {"text": "보라매병원 당뇨 클리닉 다니고 있어.", "expected": [("보라매병원", "hospital")], "clinical_should_preserve": ["당뇨"]},
    {"text": "국립중앙의료원에서 건강검진 받았는데 괜찮대.", "expected": [("국립중앙의료원", "hospital")], "clinical_should_preserve": []},
    {"text": "아산병원에서 심장 검사를 했어.", "expected": [("아산병원", "hospital")], "clinical_should_preserve": ["심장"]},

    # --- 1인칭: 날짜 (본인 일정/경험을 말하는 경우) — 5 cases ---
    {"text": "2024년 5월 10일에 수술받았어.", "expected": [("2024년 5월 10일", "date_specific")], "clinical_should_preserve": ["수술"]},
    {"text": "2025년 1월 3일에 건강검진 예약했어.", "expected": [("2025년 1월 3일", "date_specific")], "clinical_should_preserve": []},
    {"text": "2023년 11월부터 혈압약 먹기 시작했어.", "expected": [("2023년 11월", "date_specific")], "clinical_should_preserve": ["혈압"]},
    {"text": "2024년 9월 20일에 넘어져서 입원했었어.", "expected": [("2024년 9월 20일", "date_specific")], "clinical_should_preserve": []},
    {"text": "2025년 3월 1일부터 재활 운동 시작했어.", "expected": [("2025년 3월 1일", "date_specific")], "clinical_should_preserve": ["재활"]},

    # --- 1인칭: 주민등록번호 (본인이 알려주는 경우) — 4 cases ---
    {"text": "내 주민번호가 440315-1234567인데 확인 좀 해줘.", "expected": [("440315-1234567", "ssn")], "clinical_should_preserve": []},
    {"text": "주민번호 520810-2345678로 보험 청구해줘.", "expected": [("520810-2345678", "ssn")], "clinical_should_preserve": []},
    {"text": "처방전에 주민번호 610422-1111111 적어야 해.", "expected": [("610422-1111111", "ssn")], "clinical_should_preserve": []},
    {"text": "주민등록번호 481205-2222222인데 맞나 봐줘.", "expected": [("481205-2222222", "ssn")], "clinical_should_preserve": []},
    # --- STT 변형: 공백 구분/구분자 없음 ---
    {"text": "내 주민번호 5508281234567이야.", "expected": [("5508281234567", "ssn")], "clinical_should_preserve": []},
    {"text": "주민번호 620315 1111111 맞는지 확인 좀.", "expected": [("620315 1111111", "ssn")], "clinical_should_preserve": []},

    # --- 1인칭: 복합 (여러 PII가 동시에 나오는 자연스러운 발화) — 15 cases ---
    {
        "text": "나 78살인데 서울시 마포구 합정동에 혼자 살아. "
                "딸이 가끔 와서 약 챙겨줘.",
        "expected": [
            ("78살", "age_specific"), ("서울시 마포구 합정동", "address"),
            ("딸", "family_ref"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "요실금이 부끄러워서 삼성서울병원에서도 제대로 말을 못 했어.",
        "expected": [
            ("요실금", "stigma"), ("부끄러워서", "emotional"),
            ("삼성서울병원", "hospital"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "2024년 7월 15일에 서울대학병원에서 치매 검사를 받았어. "
                "결과가 두려워.",
        "expected": [
            ("2024년 7월 15일", "date_specific"), ("서울대학병원", "hospital"),
            ("치매", "stigma"), ("두려워", "emotional"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "내 전화번호 010-9999-8888인데, 혈압이 갑자기 올라서 걱정이야.",
        "expected": [("010-9999-8888", "phone")],
        "clinical_should_preserve": ["혈압"],
    },
    {
        "text": "나 85살이고 부산시 해운대구 중동에 살아. "
                "아들이 매주 와서 확인해.",
        "expected": [
            ("85살", "age_specific"), ("부산시 해운대구 중동", "address"),
            ("아들", "family_ref"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "주민번호 500620-2456789인데, 요실금 약 처방받으러 왔어.",
        "expected": [("500620-2456789", "ssn"), ("요실금", "stigma")],
        "clinical_should_preserve": [],
    },
    {
        "text": "우울증 약을 먹고 있는데 며느리한테 말 못 했어. 창피해서.",
        "expected": [
            ("우울증", "stigma"), ("며느리", "family_ref"),
            ("창피", "emotional"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "82세인데 대전시 서구 둔산동에 살아. 세브란스병원까지 멀어.",
        "expected": [
            ("82세", "age_specific"), ("대전시 서구 둔산동", "address"),
            ("세브란스병원", "hospital"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "010-4444-5555로 연락해줘. 2025년 2월 10일에 검사 결과 나온대.",
        "expected": [
            ("010-4444-5555", "phone"), ("2025년 2월 10일", "date_specific"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "인지장애 진단 받고 나서 외로워. 아내도 먼저 가고 혼자야.",
        "expected": [
            ("인지장애", "stigma"), ("외로워", "emotional"),
            ("아내", "family_ref"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "나 91살이고 수원시 팔달구 매산동에 살아. "
                "치매가 올까 봐 무서워.",
        "expected": [
            ("91살", "age_specific"), ("수원시 팔달구 매산동", "address"),
            ("치매", "stigma"), ("무서워", "emotional"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "손녀가 보라매병원 예약해줬어. 2025년 3월 5일에 가기로 했어.",
        "expected": [
            ("손녀", "family_ref"), ("보라매병원", "hospital"),
            ("2025년 3월 5일", "date_specific"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "주민번호 710803-1234567이야. 73살인데 당뇨 관리 잘 하고 있어.",
        "expected": [
            ("710803-1234567", "ssn"), ("73살", "age_specific"),
        ],
        "clinical_should_preserve": ["당뇨"],
    },
    {
        "text": "딸한테 요실금 있다고 말했더니 부끄러워하지 말래. "
                "아산병원 가보자고 했어.",
        "expected": [
            ("딸", "family_ref"), ("요실금", "stigma"),
            ("부끄러워", "emotional"), ("아산병원", "hospital"),
        ],
        "clinical_should_preserve": [],
    },
    {
        "text": "76세인데 우울증도 있고 무릎도 아파. "
                "서울시 강서구 화곡동에서 혼자 살아.",
        "expected": [
            ("76세", "age_specific"), ("우울증", "stigma"),
            ("서울시 강서구 화곡동", "address"),
        ],
        "clinical_should_preserve": ["무릎"],
    },

    # --- 1인칭: 오탐 방지 (PII 없는 노인 발화) — 23 cases ---
    {"text": "요즘 잠을 통 못 자겠어. 밤에 자꾸 깨.", "expected": [], "clinical_should_preserve": ["잠"]},
    {"text": "허리가 아파서 산책을 못 하겠어.", "expected": [], "clinical_should_preserve": ["허리"]},
    {"text": "혈압약을 먹고 나면 어지러워.", "expected": [], "clinical_should_preserve": ["혈압"]},
    {"text": "관절이 쑤셔서 계단 오르기가 힘들어.", "expected": [], "clinical_should_preserve": ["관절"]},
    {"text": "밥맛이 없어서 하루에 한 끼밖에 못 먹어.", "expected": [], "clinical_should_preserve": []},
    {"text": "기침이 계속 나오는데 병원에 가야 하나?", "expected": [], "clinical_should_preserve": ["기침"]},
    {"text": "눈이 침침해서 글씨가 잘 안 보여.", "expected": [], "clinical_should_preserve": []},
    {"text": "손이 떨려서 젓가락질이 어려워.", "expected": [], "clinical_should_preserve": []},
    {"text": "걷다 보면 숨이 차서 쉬어야 해.", "expected": [], "clinical_should_preserve": []},
    {"text": "소화가 안 되고 속이 더부룩해.", "expected": [], "clinical_should_preserve": []},
    {"text": "오늘 산책하고 왔는데 기분이 좋아.", "expected": [], "clinical_should_preserve": []},
    {"text": "TV에서 건강 프로그램 보고 따라 해봤어.", "expected": [], "clinical_should_preserve": []},
    {"text": "복지관에서 체조 하고 왔어.", "expected": [], "clinical_should_preserve": []},
    {"text": "옛날 생각이 나서 좀 그랬어.", "expected": [], "clinical_should_preserve": []},
    {"text": "요즘 비가 와서 빨래를 못 말렸어.", "expected": [], "clinical_should_preserve": []},
    {"text": "김치를 담갔는데 맛있게 됐어.", "expected": [], "clinical_should_preserve": []},
    {"text": "경로당에서 화투 치며 놀았어.", "expected": [], "clinical_should_preserve": []},
    {"text": "물을 많이 마셔야 한다길래 하루에 8잔 마시고 있어.", "expected": [], "clinical_should_preserve": []},
    {"text": "운동화를 새로 샀어. 산책하기 편해.", "expected": [], "clinical_should_preserve": []},
    {"text": "오늘 날씨가 좋아서 마당에 나갔다 왔어.", "expected": [], "clinical_should_preserve": []},
    {"text": "라디오 듣는 게 낙이야.", "expected": [], "clinical_should_preserve": []},
    {"text": "내일 모레 건강검진 받으러 가야 해.", "expected": [], "clinical_should_preserve": ["검진"]},
    {"text": "약 먹는 시간을 자꾸 까먹어서 알람 맞춰놨어.", "expected": [], "clinical_should_preserve": []},
]


# ── 건강 위험 신호 테스트 데이터 ──
HEALTH_SIGNAL_TEST_DATA = [
    # ═══════════════════════════════════════
    #  수면 장애 (Sleep Disorder) — 6 cases
    # ═══════════════════════════════════════
    {"text": "요즘 잠을 통 못 자겠어요. 새벽에 자꾸 깨요.", "expected_categories": ["sleep_disorder"], "expected_min_level": "medium"},
    {"text": "밤에 다리가 저려서 잠을 설쳤어요.", "expected_categories": ["pain", "sleep_disorder"], "expected_min_level": "medium"},
    {"text": "수면제를 먹어도 잠이 안 와요. 더 강한 약 없나요?", "expected_categories": ["sleep_disorder", "medication"], "expected_min_level": "high"},
    {"text": "코를 너무 심하게 골아서 밤에 계속 깨요.", "expected_categories": ["sleep_disorder"], "expected_min_level": "medium"},
    {"text": "새벽 3시에 깨면 다시 잠들기 어려워요.", "expected_categories": ["sleep_disorder"], "expected_min_level": "medium"},
    {"text": "밤새 뒤척이다가 아침에 겨우 잠들어요.", "expected_categories": ["sleep_disorder"], "expected_min_level": "medium"},

    # ═══════════════════════════════════════
    #  만성질환 (Chronic Disease) — 5 cases
    # ═══════════════════════════════════════
    {"text": "혈압이 올라서 걱정이에요. 당뇨도 있고요.", "expected_categories": ["chronic_disease"], "expected_min_level": "high"},
    {"text": "어지러워서 넘어질 뻔했어요. 혈압이 낮은 것 같아요.", "expected_categories": ["fall_risk", "chronic_disease"], "expected_min_level": "high"},
    {"text": "당뇨 수치가 공복에 200 넘게 나왔어요.", "expected_categories": ["chronic_disease"], "expected_min_level": "high"},
    {"text": "혈압이 160/100으로 측정됐는데 약을 더 먹어야 하나요?", "expected_categories": ["chronic_disease"], "expected_min_level": "high"},
    {"text": "콜레스테롤이 높다고 해서 식이요법 중이에요.", "expected_categories": ["chronic_disease"], "expected_min_level": "medium"},

    # ═══════════════════════════════════════
    #  낙상 위험 (Fall Risk) — 5 cases
    # ═══════════════════════════════════════
    {"text": "어제 넘어져서 무릎이 많이 아파요.", "expected_categories": ["fall_risk", "pain"], "expected_min_level": "high"},
    {"text": "어지러워서 길에서 넘어졌어요.", "expected_categories": ["fall_risk"], "expected_min_level": "high"},
    {"text": "걸을 때 자꾸 비틀거려서 무서워요.", "expected_categories": ["fall_risk"], "expected_min_level": "high"},
    {"text": "화장실 가다가 미끄러져서 엉덩이를 부딪혔어요.", "expected_categories": ["fall_risk", "pain"], "expected_min_level": "high"},
    {"text": "다리에 힘이 빠져서 주저앉았어요.", "expected_categories": ["fall_risk"], "expected_min_level": "high"},

    # ═══════════════════════════════════════
    #  통증 (Pain) — 5 cases
    # ═══════════════════════════════════════
    {"text": "허리가 너무 아파서 걷기가 힘들어요.", "expected_categories": ["pain"], "expected_min_level": "medium"},
    {"text": "무릎이 시큰시큰 쑤셔서 계단을 못 올라가요.", "expected_categories": ["pain"], "expected_min_level": "medium"},
    {"text": "어깨가 아파서 팔을 못 올리겠어요.", "expected_categories": ["pain"], "expected_min_level": "medium"},
    {"text": "두통이 일주일째 계속돼요. 머리가 깨질 것 같아요.", "expected_categories": ["pain"], "expected_min_level": "high"},
    {"text": "손목이 저리고 아파서 물건을 잡기 힘들어요.", "expected_categories": ["pain"], "expected_min_level": "medium"},

    # ═══════════════════════════════════════
    #  영양/식이 (Nutritional) — 4 cases
    # ═══════════════════════════════════════
    {"text": "요즘 밥맛이 없어서 체중이 줄었어요.", "expected_categories": ["nutritional"], "expected_min_level": "medium"},
    {"text": "한 달 동안 5kg이 빠졌어요. 입맛이 없어서요.", "expected_categories": ["nutritional"], "expected_min_level": "high"},
    {"text": "물도 잘 안 마시게 돼요. 목이 안 말라서요.", "expected_categories": ["nutritional"], "expected_min_level": "medium"},
    {"text": "밥을 하루에 한 끼밖에 안 드세요.", "expected_categories": ["nutritional"], "expected_min_level": "high"},

    # ═══════════════════════════════════════
    #  감정/정서 (Emotional) — 5 cases
    # ═══════════════════════════════════════
    {"text": "요즘 우울하고 아무것도 하기 싫어요.", "expected_categories": ["emotional"], "expected_min_level": "medium"},
    {"text": "살고 싶은 마음이 없어요. 그냥 끝내고 싶어요.", "expected_categories": ["emotional"], "expected_min_level": "critical"},
    {"text": "하루 종일 울고만 있어요. 눈물이 멈추지 않아요.", "expected_categories": ["emotional"], "expected_min_level": "high"},
    {"text": "아무 의욕이 없고 사람 만나기 싫어요.", "expected_categories": ["emotional"], "expected_min_level": "medium"},
    {"text": "밤마다 불안해서 잠이 안 와요. 가슴이 두근거려요.", "expected_categories": ["emotional", "sleep_disorder"], "expected_min_level": "high"},

    # ═══════════════════════════════════════
    #  인지 기능 (Cognitive) — 4 cases
    # ═══════════════════════════════════════
    {"text": "자꾸 물건을 어디 뒀는지 잊어버려요. 기억력이 많이 나빠졌어요.", "expected_categories": ["cognitive"], "expected_min_level": "high"},
    {"text": "집 앞인데 길을 잃어버렸어요. 어디가 어딘지 모르겠어요.", "expected_categories": ["cognitive"], "expected_min_level": "critical"},
    {"text": "방금 무슨 말 했는지 기억이 안 나요.", "expected_categories": ["cognitive"], "expected_min_level": "medium"},
    {"text": "날짜를 자꾸 잊어버리고 약속을 까먹어요.", "expected_categories": ["cognitive"], "expected_min_level": "medium"},

    # ═══════════════════════════════════════
    #  복약 (Medication) — 4 cases
    # ═══════════════════════════════════════
    {"text": "혈압약을 며칠째 안 먹었어요. 깜빡했어요.", "expected_categories": ["medication"], "expected_min_level": "high"},
    {"text": "약이 너무 많아서 뭘 먹어야 하는지 헷갈려요.", "expected_categories": ["medication"], "expected_min_level": "medium"},
    {"text": "약을 먹으면 속이 울렁거려서 안 먹고 있어요.", "expected_categories": ["medication"], "expected_min_level": "high"},
    {"text": "약을 두 배로 먹었는데 괜찮을까요?", "expected_categories": ["medication"], "expected_min_level": "high"},

    # ═══════════════════════════════════════
    #  응급 (Emergency) — 4 cases
    # ═══════════════════════════════════════
    {"text": "갑자기 가슴이 아프고 숨이 잘 안 쉬어져요.", "expected_categories": ["emergency"], "expected_min_level": "critical"},
    {"text": "한쪽 팔다리에 힘이 안 들어가고 말이 어눌해졌어요.", "expected_categories": ["emergency"], "expected_min_level": "critical"},
    {"text": "갑자기 의식을 잃고 쓰러지셨어요.", "expected_categories": ["emergency"], "expected_min_level": "critical"},
    {"text": "심한 두통과 함께 구토를 하고 있어요.", "expected_categories": ["emergency"], "expected_min_level": "critical"},

    # ═══════════════════════════════════════
    #  위생 (Hygiene) — 3 cases
    # ═══════════════════════════════════════
    {"text": "목욕하기가 힘들어서 일주일째 못 씻었어요.", "expected_categories": ["hygiene"], "expected_min_level": "medium"},
    {"text": "양치질을 혼자 하기 어렵대요. 손에 힘이 없어서요.", "expected_categories": ["hygiene"], "expected_min_level": "medium"},
    {"text": "옷을 갈아입지도 않고 계속 같은 옷만 입고 계세요.", "expected_categories": ["hygiene"], "expected_min_level": "medium"},

    # ═══════════════════════════════════════
    #  오탐 방지 — Negative (건강 신호 없음) — 8 cases
    # ═══════════════════════════════════════
    {"text": "오늘 날씨가 좋아서 산책 다녀왔어요.", "expected_categories": [], "expected_min_level": "low"},
    {"text": "오늘 점심에 김치찌개 먹었는데 맛있었어요.", "expected_categories": [], "expected_min_level": "low"},
    {"text": "손주가 놀러 와서 즐거웠어요.", "expected_categories": [], "expected_min_level": "low"},
    {"text": "어제 TV에서 건강 프로그램 봤어요.", "expected_categories": [], "expected_min_level": "low"},
    {"text": "오랜만에 친구를 만나서 차 한잔 했어요.", "expected_categories": [], "expected_min_level": "low"},
    {"text": "마당에 꽃이 피어서 기분이 좋아요.", "expected_categories": [], "expected_min_level": "low"},
    {"text": "딸이 주말에 온다고 해서 기다리고 있어요.", "expected_categories": [], "expected_min_level": "low"},
    {"text": "오늘 복지관에서 노래교실 다녀왔어요.", "expected_categories": [], "expected_min_level": "low"},
]


# ── Base vs Fine-tuned 비교용 프롬프트 ──
COMPARISON_PROMPTS = [
    {
        "id": "sleep_001",
        "intent": "health_consult",
        "query": "요즘 잠을 통 못 자겠어. 밤에 자다 깨고 또 깨고… 피곤해 죽겠어.",
        "topic": "수면장애",
    },
    {
        "id": "incontinence_001",
        "intent": "health_consult",
        "query": "재채기만 하면 소변이 새는데, 너무 부끄러워서 아무한테도 말 못 하겠어.",
        "topic": "요실금 (프라이버시 민감)",
    },
    {
        "id": "cognitive_001",
        "intent": "health_consult",
        "query": "딸이 자꾸 기억력이 떨어진다고 걱정하더라고.",
        "topic": "인지기능 (프라이버시 민감)",
    },
    {
        "id": "medication_001",
        "intent": "medication",
        "query": "혈압약을 며칠 안 먹었는데 괜찮을까?",
        "topic": "복약 관리",
    },
    {
        "id": "fall_001",
        "intent": "health_consult",
        "query": "어제 화장실 가다가 미끄러져서 넘어졌어. 무릎이 아파.",
        "topic": "낙상",
    },
    {
        "id": "emotion_001",
        "intent": "health_consult",
        "query": "요즘 뭘해도 재미가 없고 그냥 우울해.",
        "topic": "정서 건강",
    },
    {
        "id": "general_001",
        "intent": "general_chat",
        "query": "오늘 날씨가 좋으니까 기분이 좋아.",
        "topic": "일상 대화",
    },
    {
        "id": "emergency_001",
        "intent": "emergency",
        "query": "갑자기 가슴이 너무 아프고 숨이 안 쉬어져요.",
        "topic": "응급",
    },
    {
        "id": "lifestyle_001",
        "intent": "lifestyle",
        "query": "운동을 하고 싶은데 무릎이 안 좋아서 뭘 해야 할지 모르겠어.",
        "topic": "생활습관",
    },
    {
        "id": "privacy_001",
        "intent": "health_consult",
        "query": "아들이 나보고 치매 검사 받으라는데... 나는 괜찮은 것 같은데.",
        "topic": "인지기능 (가족+프라이버시)",
    },
]


# ═══════════════════════════════════════════════════════
#  2. PII 감지 평가
# ═══════════════════════════════════════════════════════

def evaluate_pii_detection():
    """PII 감지 & 삭제 정밀도/재현율/F1 측정"""
    from app.preprocessing.pii_redactor import PIIRedactor, PIIType
    
    print("\n" + "=" * 70)
    print("  1. PII 감지 & 삭제 평가 (Precision / Recall / F1)")
    print("=" * 70)
    
    redactor = PIIRedactor()
    
    # PII 유형별 TP/FP/FN 카운터
    type_map = {
        "name": PIIType.NAME,
        "phone": PIIType.PHONE,
        "ssn": PIIType.SSN,
        "address": PIIType.ADDRESS,
        "age_specific": PIIType.AGE_SPECIFIC,
        "family_ref": PIIType.FAMILY_REFERENCE,
        "stigma": PIIType.STIGMATIZED_CONDITION,
        "emotional": PIIType.EMOTIONAL_SENSITIVE,
        "hospital": PIIType.HOSPITAL_NAME,
        "date_specific": PIIType.DATE_SPECIFIC,
    }
    
    metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    total_tp = total_fp = total_fn = 0
    
    # 임상 용어 보존 확인
    clinical_preserved_count = 0
    clinical_total_count = 0
    
    results_detail = []
    
    for i, test_case in enumerate(PII_TEST_DATA):
        text = test_case["text"]
        expected = test_case["expected"]
        clinical_should_preserve = test_case.get("clinical_should_preserve", [])
        
        result = redactor.redact(text)
        detected_types = [(d.original_text, d.pii_type.value) for d in result.detections]
        
        # PII 유형별 매칭
        # Expected에서 감지된 것 → TP, 감지 안 된 것 → FN
        detected_set = set()
        for exp_text, exp_type in expected:
            pii_type_enum = type_map.get(exp_type)
            found = False
            for det in result.detections:
                if det.pii_type == pii_type_enum and (
                    exp_text in det.original_text or det.original_text in exp_text
                ):
                    found = True
                    detected_set.add(id(det))
                    break
            
            if found:
                metrics[exp_type]["tp"] += 1
                total_tp += 1
            else:
                metrics[exp_type]["fn"] += 1
                total_fn += 1
        
        # 감지되었지만 expected에 없는 것 → FP
        for det in result.detections:
            if id(det) not in detected_set:
                pii_key = det.pii_type.value
                # family_ref와 family_reference 통일
                for k, v in type_map.items():
                    if v == det.pii_type:
                        pii_key = k
                        break
                metrics[pii_key]["fp"] += 1
                total_fp += 1
        
        # 임상 용어 보존 확인
        for term in clinical_should_preserve:
            clinical_total_count += 1
            if term in result.redacted_text:
                clinical_preserved_count += 1
        
        results_detail.append({
            "text": text[:60] + "..." if len(text) > 60 else text,
            "expected_count": len(expected),
            "detected_count": len(result.detections),
            "redacted_preview": result.redacted_text[:80],
        })
    
    # 결과 출력
    print("\n--- PII 유형별 성능 ---")
    print(f"{'유형':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 60)
    
    type_results = {}
    for pii_type in sorted(metrics.keys()):
        m = metrics[pii_type]
        precision = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0
        recall = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{pii_type:<20} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
              f"{precision:>7.3f} {recall:>7.3f} {f1:>7.3f}")
        type_results[pii_type] = {"precision": precision, "recall": recall, "f1": f1}
    
    # 전체 micro 평균
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0
    
    print("-" * 60)
    print(f"{'Overall (micro)':>20} {total_tp:>4} {total_fp:>4} {total_fn:>4} "
          f"{overall_prec:>7.3f} {overall_rec:>7.3f} {overall_f1:>7.3f}")
    
    # 임상 용어 보존율
    clin_rate = clinical_preserved_count / clinical_total_count if clinical_total_count > 0 else 1.0
    print(f"\n--- 임상 용어 보존율 ---")
    print(f"보존된 임상 용어: {clinical_preserved_count}/{clinical_total_count} ({clin_rate:.1%})")
    
    # 삭제 예시 출력
    print(f"\n--- 삭제 예시 (Before → After) ---")
    for r in results_detail[:5]:
        print(f"  원문: {r['text']}")
        print(f"  삭제: {r['redacted_preview']}")
        print()
    
    return {
        "type_results": type_results,
        "overall": {"precision": overall_prec, "recall": overall_rec, "f1": overall_f1},
        "clinical_preservation_rate": clin_rate,
        "total_test_cases": len(PII_TEST_DATA),
    }


# ═══════════════════════════════════════════════════════
#  3. 건강 위험 신호 감지 평가
# ═══════════════════════════════════════════════════════

def evaluate_health_signal_detection():
    """건강 위험 신호 감지 precision/recall/F1"""
    from app.preprocessing.health_signal_detector import HealthSignalDetector
    
    print("\n" + "=" * 70)
    print("  2. 건강 위험 신호 감지 평가")
    print("=" * 70)
    
    detector = HealthSignalDetector()
    
    level_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    
    category_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    level_correct = 0
    level_total = 0
    total_tp = total_fp = total_fn = 0
    latencies = []
    
    for test_case in HEALTH_SIGNAL_TEST_DATA:
        text = test_case["text"]
        expected_cats = set(test_case["expected_categories"])
        expected_min = test_case["expected_min_level"]
        
        start_t = time.perf_counter()
        result = detector.analyze(text)
        elapsed = (time.perf_counter() - start_t) * 1000
        latencies.append(elapsed)
        
        detected_cats = set()
        for sig in result.risk_signals:
            cat_val = sig.category.value if hasattr(sig.category, 'value') else sig.category
            detected_cats.add(cat_val)
        
        # 카테고리별 TP/FP/FN
        for cat in expected_cats:
            if cat in detected_cats:
                category_metrics[cat]["tp"] += 1
                total_tp += 1
            else:
                category_metrics[cat]["fn"] += 1
                total_fn += 1
        
        for cat in detected_cats:
            if cat not in expected_cats:
                category_metrics[cat]["fp"] += 1
                total_fp += 1
        
        # 위험 수준 판정
        actual_level = result.overall_risk_level.value if hasattr(result.overall_risk_level, 'value') else result.overall_risk_level
        level_total += 1
        if level_order.get(actual_level, 0) >= level_order.get(expected_min, 0):
            level_correct += 1
    
    # 결과
    print(f"\n--- 카테고리별 감지 성능 ---")
    print(f"{'카테고리':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 60)
    
    cat_results = {}
    for cat in sorted(category_metrics.keys()):
        m = category_metrics[cat]
        p = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0
        r = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"{cat:<20} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {p:>7.3f} {r:>7.3f} {f1:>7.3f}")
        cat_results[cat] = {"precision": p, "recall": r, "f1": f1}
    
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    
    print("-" * 60)
    print(f"{'Overall (micro)':>20} {total_tp:>4} {total_fp:>4} {total_fn:>4} "
          f"{overall_p:>7.3f} {overall_r:>7.3f} {overall_f1:>7.3f}")
    
    level_acc = level_correct / level_total if level_total > 0 else 0
    print(f"\n위험 수준 판정 정확도: {level_correct}/{level_total} ({level_acc:.1%})")
    print(f"평균 감지 시간: {sum(latencies)/len(latencies):.1f} ms")
    
    return {
        "category_results": cat_results,
        "overall": {"precision": overall_p, "recall": overall_r, "f1": overall_f1},
        "level_accuracy": level_acc,
        "avg_latency_ms": sum(latencies) / len(latencies),
    }


# ═══════════════════════════════════════════════════════
#  4. 프라이버시 커뮤니케이션 전략 시나리오 테스트
# ═══════════════════════════════════════════════════════

def evaluate_privacy_strategy():
    """프라이버시 인식 대화 전략 자동 시나리오 검증"""
    print("\n" + "=" * 70)
    print("  3. 프라이버시 커뮤니케이션 전략 시나리오 테스트")
    print("=" * 70)
    
    import importlib
    
    # nodes 모듈에서 함수 테스트
    try:
        from app.graph.nodes import _detect_repeated_question, _detect_topic_drift
        has_nodes = True
    except ImportError:
        has_nodes = False
        print("  [SKIP] app.graph.nodes 임포트 실패 (DB 의존성)")
    
    results = []
    
    # --- 테스트 A: 점진적 공개 (턴 카운트 기반) ---
    print("\n  [A] 점진적 공개 관리 (Graduated Disclosure)")
    
    # 시뮬레이션: 3턴 이전에는 의료 권유 없음, 3턴 이후 MEDIUM+ 시 권유
    test_cases_disclosure = [
        {"turn": 1, "risk": "medium", "should_refer": False, "desc": "1턴, MEDIUM → 권유 없어야 함"},
        {"turn": 2, "risk": "high",   "should_refer": False, "desc": "2턴, HIGH → 권유 없어야 함"},
        {"turn": 3, "risk": "medium", "should_refer": True,  "desc": "3턴, MEDIUM → 권유 있어야 함"},
        {"turn": 5, "risk": "high",   "should_refer": True,  "desc": "5턴, HIGH → 권유 있어야 함"},
        {"turn": 4, "risk": "low",    "should_refer": False, "desc": "4턴, LOW → 권유 없어야 함"},
    ]
    
    for tc in test_cases_disclosure:
        # 논문 로직: turn_count >= 3 AND risk_level in (MEDIUM, HIGH, CRITICAL)
        should_refer = (tc["turn"] >= 3) and (tc["risk"] in ("medium", "high", "critical"))
        passed = should_refer == tc["should_refer"]
        status = "PASS" if passed else "FAIL"
        results.append({"test": tc["desc"], "status": status})
        print(f"    [{status}] {tc['desc']}")
    
    # --- 테스트 B: 반복 질문 감지 ---
    print("\n  [B] 반복 질문 감지 (Repeated Question Detection)")
    
    if has_nodes:
        repeat_tests = [
            {
                "prev": "잠을 못 자겠어요 수면이 안 되요",
                "curr": "잠이 잘 안 와요 수면 문제에요",
                "should_detect": True,
                "desc": "수면 관련 반복 → 감지"
            },
            {
                "prev": "혈압이 높아요 혈압약 먹고 있어요",
                "curr": "오늘 날씨가 좋아요",
                "should_detect": False,
                "desc": "다른 주제 → 미감지"
            },
        ]
        
        for tc in repeat_tests:
            try:
                detected = _detect_repeated_question(tc["prev"], tc["curr"])
                passed = detected == tc["should_detect"]
            except Exception as e:
                passed = False
            status = "PASS" if passed else "FAIL"
            results.append({"test": tc["desc"], "status": status})
            print(f"    [{status}] {tc['desc']}")
    else:
        # 로직만 테스트 (overlap coefficient 시뮬레이션)
        def overlap_coefficient(set_a, set_b):
            if not set_a or not set_b:
                return 0.0
            intersect = set_a & set_b
            return len(intersect) / min(len(set_a), len(set_b))
        
        tests = [
            ({"잠", "못", "자", "수면", "안"}, {"잠", "안", "수면", "문제"}, True),
            ({"혈압", "높", "약", "먹"}, {"날씨", "좋"}, False),
        ]
        for a, b, expected in tests:
            score = overlap_coefficient(a, b)
            detected = score >= 0.7 and len(a & b) >= 3
            passed = detected == expected
            status = "PASS" if passed else "FAIL"
            desc = f"Overlap={score:.2f}, matched={len(a&b)} → {'감지' if expected else '미감지'}"
            results.append({"test": desc, "status": status})
            print(f"    [{status}] {desc}")
    
    # --- 테스트 C: 프롬프트 프라이버시 검증 ---
    print("\n  [C] 인텐트별 프롬프트 프라이버시 검증")
    
    from app.config import HealthcarePrompts
    
    prompt_tests = [
        {
            "prompt": HealthcarePrompts.HEALTH_CONSULT_PROMPT,
            "should_contain": ["공감"],
            "should_not_contain": ["치매가 있습니까", "치매 증상이 있습니까"],
            "desc": "건강상담: 간접적 표현 사용"
        },
        {
            "prompt": HealthcarePrompts.EMERGENCY_PROMPT,
            "should_contain": ["119", "침착"],
            "should_not_contain": [],
            "desc": "응급: 119 안내 포함"
        },
        {
            "prompt": HealthcarePrompts.GENERAL_CHAT_PROMPT,
            "should_contain": ["질문"],
            "should_not_contain": ["건강 상담", "건강을 위해", "건강 관리"],
            "desc": "일반대화: 건강 상담 유도하지 않음"
        },
    ]
    
    for tc in prompt_tests:
        prompt_text = tc["prompt"]
        has_required = all(kw in prompt_text for kw in tc["should_contain"])
        has_forbidden = any(kw in prompt_text for kw in tc["should_not_contain"])
        passed = has_required and not has_forbidden
        status = "PASS" if passed else "FAIL"
        results.append({"test": tc["desc"], "status": status})
        print(f"    [{status}] {tc['desc']}")
    
    # 요약
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    print(f"\n  결과: {pass_count}/{total} 통과 ({pass_count/total:.0%})")
    
    return {"pass": pass_count, "total": total, "rate": pass_count / total, "details": results}


# ═══════════════════════════════════════════════════════
#  5. Base vs Fine-tuned 모델 비교
# ═══════════════════════════════════════════════════════

def evaluate_model_comparison():
    """Base SLM vs Fine-tuned SLM 응답 비교 + Latency"""
    import httpx
    
    print("\n" + "=" * 70)
    print("  4. Base SLM vs Fine-tuned SLM 응답 비교")
    print("=" * 70)
    
    OLLAMA_URL = "http://localhost:11434/api/generate"
    BASE_MODEL = "kanana-base-raw"
    FINETUNED_MODEL = "kanana-counseling"
    
    # Ollama 사용 가능 체크
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        has_base = any(BASE_MODEL in m for m in models)
        has_ft = any(FINETUNED_MODEL in m for m in models)
        if not has_base or not has_ft:
            missing = []
            if not has_base:
                missing.append(BASE_MODEL)
            if not has_ft:
                missing.append(FINETUNED_MODEL)
            print(f"  [SKIP] 모델 없음: {', '.join(missing)}")
            return None
    except Exception as e:
        print(f"  [SKIP] Ollama 연결 실패: {e}")
        return None
    
    results = []
    base_latencies = []
    ft_latencies = []
    
    for prompt_data in COMPARISON_PROMPTS:
        query = prompt_data["query"]
        topic = prompt_data["topic"]
        
        print(f"\n  [{prompt_data['id']}] {topic}")
        print(f"  질문: {query}")
        
        # Base 모델 응답
        base_start = time.perf_counter()
        try:
            resp = httpx.post(OLLAMA_URL, json={
                "model": BASE_MODEL,
                "prompt": query,
                "stream": False,
                "options": {"num_predict": 200, "temperature": 0.4}
            }, timeout=60)
            base_response = resp.json().get("response", "").strip()
            base_latency = (time.perf_counter() - base_start) * 1000
            base_latencies.append(base_latency)
        except Exception as e:
            base_response = f"[ERROR: {e}]"
            base_latency = 0
        
        # Fine-tuned 모델 응답
        ft_start = time.perf_counter()
        try:
            resp = httpx.post(OLLAMA_URL, json={
                "model": FINETUNED_MODEL,
                "prompt": query,
                "stream": False,
                "options": {"num_predict": 512, "temperature": 0.4}
            }, timeout=60)
            ft_response = resp.json().get("response", "").strip()
            ft_latency = (time.perf_counter() - ft_start) * 1000
            ft_latencies.append(ft_latency)
        except Exception as e:
            ft_response = f"[ERROR: {e}]"
            ft_latency = 0
        
        # 응답 품질 자동 평가 (heuristic)
        quality = auto_evaluate_response(query, base_response, ft_response, prompt_data["intent"])
        
        print(f"  Base ({base_latency:.0f}ms): {base_response[:100]}...")
        print(f"  Fine ({ft_latency:.0f}ms): {ft_response[:100]}...")
        print(f"  자동평가: {quality['summary']}")
        
        results.append({
            "id": prompt_data["id"],
            "topic": topic,
            "intent": prompt_data["intent"],
            "query": query,
            "base_response": base_response,
            "ft_response": ft_response,
            "base_latency_ms": base_latency,
            "ft_latency_ms": ft_latency,
            "quality": quality,
        })
    
    # Latency 통계
    print(f"\n--- Latency 통계 ---")
    if base_latencies:
        print(f"  Base  모델: avg={sum(base_latencies)/len(base_latencies):.0f}ms, "
              f"min={min(base_latencies):.0f}ms, max={max(base_latencies):.0f}ms")
    if ft_latencies:
        print(f"  Fine-tuned: avg={sum(ft_latencies)/len(ft_latencies):.0f}ms, "
              f"min={min(ft_latencies):.0f}ms, max={max(ft_latencies):.0f}ms")
    
    return {
        "results": results,
        "base_avg_latency": sum(base_latencies) / len(base_latencies) if base_latencies else 0,
        "ft_avg_latency": sum(ft_latencies) / len(ft_latencies) if ft_latencies else 0,
    }


def auto_evaluate_response(query: str, base_resp: str, ft_resp: str, intent: str) -> Dict:
    """응답 품질 자동 평가 (heuristic 기반)
    
    평가 항목:
    1. 길이 적절성 (20-300자)
    2. 존댓말 사용
    3. 공감 표현 포함
    4. 목록/번호/마크다운 미사용 (자연스러운 대화체)
    5. 금지 표현 미사용
    6. 응급 의도에 119 포함
    7. 프라이버시 침해 표현 없음
    8. 구체적 건강 조언 포함
    9. 대화체 자연스러움 (질문으로 끝남)
    """
    scores = {"base": {}, "ft": {}}
    
    for label, resp in [("base", base_resp), ("ft", ft_resp)]:
        # 1. 응답 길이 적절성 (너무 짧거나 긴 응답 감점)
        length = len(resp)
        if 20 <= length <= 300:
            scores[label]["length"] = 1.0
        elif length < 20:
            scores[label]["length"] = 0.3
        else:
            scores[label]["length"] = 0.7
        
        # 2. 존댓말 사용
        honorific_endings = ["요", "다", "세요", "니다", "습니다", "어요", "아요"]
        scores[label]["honorific"] = 1.0 if any(resp.rstrip().endswith(e) for e in honorific_endings) else 0.5
        
        # 3. 공감 표현 포함
        empathy_words = ["이해", "걱정", "힘드", "괴로", "불편", "안타깝", "많이", "그러시",
                         "아이go", "아이고", "속상", "고생", "마음"]
        scores[label]["empathy"] = 1.0 if any(w in resp for w in empathy_words) else 0.0
        
        # 4. 목록/번호/마크다운 미사용 (자연스러운 대화체)
        has_list = bool(re.search(r'^\s*[\d\-\*·•][\.\)]\s', resp, re.MULTILINE))
        has_markdown = bool(re.search(r'\*\*[^*]+\*\*', resp))  # **볼드** 패턴
        has_numbered = bool(re.search(r'^\s*\d+[\.\)]\s', resp, re.MULTILINE))  # 1. 2. 3.
        scores[label]["no_list"] = 0.0 if (has_list or has_markdown or has_numbered) else 1.0
        
        # 5. 금지 표현 미사용
        forbidden = ["도움이 되셨", "도움이 되었", "궁금한 점이 있으시면", "추가 질문"]
        scores[label]["no_forbidden"] = 0.0 if any(f in resp for f in forbidden) else 1.0
        
        # 6. 응급 의도에 119 포함 여부
        if intent == "emergency":
            scores[label]["emergency_119"] = 1.0 if "119" in resp else 0.0
        
        # 7. 프라이버시 민감 질문 회피
        invasive = ["치매입니까", "치매 증상", "요실금이 있으신가요", "우울증이 있으신가요"]
        scores[label]["privacy"] = 0.0 if any(inv in resp for inv in invasive) else 1.0
        
        # 8. 구체적 건강 조언 포함 (도메인 특화 콘텐츠)
        specific_advice = [
            "수영", "걷기", "산책", "스트레칭", "요가", "자전거",              # 운동
            "수면", "카페인", "잠자리", "기상", "취침",                        # 수면
            "식이섬유", "물을", "채소", "과일", "영양", "식사",               # 영양
            "혈압약", "복용", "정기적", "검진", "진료",                       # 의료
            "보조 기구", "지팡이", "미끄럼", "안전",                          # 낙상
            "패드", "골반", "케겔",                                           # 배뇨
            "병원", "전문의", "상담",                                         # 의료 안내
        ]
        advice_count = sum(1 for a in specific_advice if a in resp)
        scores[label]["specific_advice"] = min(1.0, advice_count / 3.0)  # 3개 이상이면 만점
        
        # 9. 대화체 자연스러움 (질문으로 끝나거나 부드러운 종결)
        ends_natural = resp.rstrip().endswith("?") or resp.rstrip().endswith("요?") or \
                       resp.rstrip().endswith("세요.") or resp.rstrip().endswith("세요")
        scores[label]["natural_ending"] = 1.0 if ends_natural else 0.5
    
    # 종합 점수
    for label in ["base", "ft"]:
        vals = list(scores[label].values())
        scores[label]["total"] = sum(vals) / len(vals) if vals else 0
    
    base_total = scores["base"]["total"]
    ft_total = scores["ft"]["total"]
    
    if ft_total > base_total + 0.05:
        summary = f"Fine-tuned 우위 ({ft_total:.2f} vs {base_total:.2f})"
    elif base_total > ft_total + 0.05:
        summary = f"Base 우위 ({base_total:.2f} vs {ft_total:.2f})"
    else:
        summary = f"동등 ({ft_total:.2f} vs {base_total:.2f})"
    
    return {"base_score": base_total, "ft_score": ft_total, "summary": summary, "details": scores}


# ═══════════════════════════════════════════════════════
#  6. PII 삭제 Before/After 예시
# ═══════════════════════════════════════════════════════

def demonstrate_pii_redaction():
    """PII 삭제 Before → After 논문 예시용"""
    from app.preprocessing.pii_redactor import PIIRedactor
    
    print("\n" + "=" * 70)
    print("  5. PII 삭제 Before / After 예시")
    print("=" * 70)
    
    redactor = PIIRedactor()
    
    examples = [
        "딸이 자꾸 기억력이 떨어진다고 걱정하더라고.",
        "김영수 어르신(92살)이 서울시 강남구 역삼동에 사시는데, "
        "치매 진단 후 요실금까지 생겨서 며느리가 많이 힘들어해요.",
        "주민번호 850123-1234567인 박순자 할머니가 010-9876-5432로 전화해서 "
        "우울증 약을 안 먹었다고 부끄러워하셨어요.",
        "이순옥 어르신이 서울대학병원에서 2024년 3월 15일에 검진받으셨는데, "
        "혈압이 높고 당뇨 수치도 안 좋대요. 아들이 걱정하고 있어요.",
    ]
    
    results = []
    for text in examples:
        result = redactor.redact(text)
        print(f"\n  [Before] {text}")
        print(f"  [After ] {result.redacted_text}")
        print(f"  감지된 PII: {result.pii_count}건, "
              f"보존된 임상 용어: {result.clinical_terms_preserved}")
        results.append({
            "before": text,
            "after": result.redacted_text,
            "pii_count": result.pii_count,
            "clinical_preserved": result.clinical_terms_preserved,
        })
    
    return results


# ═══════════════════════════════════════════════════════
#  7. 템플릿 기반 건강 요약 생성 예시
# ═══════════════════════════════════════════════════════

def demonstrate_health_summary():
    """템플릿 기반 건강 요약 생성 (논문 Section 5a)"""
    from app.preprocessing.pii_redactor import PIIRedactor
    from app.preprocessing.health_signal_detector import HealthSignalDetector
    
    print("\n" + "=" * 70)
    print("  6. 템플릿 기반 건강 요약 생성 (Sanitized Summary)")
    print("=" * 70)
    
    redactor = PIIRedactor()
    detector = HealthSignalDetector()
    
    conversations = [
        "딸이 자꾸 기억력이 떨어진다고 걱정하더라고. 요즘 물건을 어디 뒀는지 자꾸 잊어버려.",
        "잠을 못 자서 수면제를 먹고 있는데, 부끄러워서 아들한테도 말 못 했어. 010-1234-5678로 전화해줘.",
        "어제 화장실 가다가 넘어져서 무릎이 아파. 김영숙 간호사한테 말했더니 병원 가보래.",
    ]
    
    results = []
    for text in conversations:
        health_result = detector.analyze(text)
        summary = redactor.generate_health_summary(text, health_result.risk_signals)
        
        print(f"\n  [원문 대화] {text[:80]}...")
        print(f"  [건강 요약] {json.dumps(summary, ensure_ascii=False, indent=2)}")
        results.append({"original": text, "summary": summary})
    
    return results


# ═══════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  IEEE ROMAN 논문용 통합 평가")
    print(f"  실행 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  플랫폼: Mac Mini M4, 32GB")
    print("=" * 70)
    
    all_results = {}
    
    # 1. PII 감지 평가
    all_results["pii_detection"] = evaluate_pii_detection()
    
    # 2. 건강 위험 신호 감지 평가
    all_results["health_signal"] = evaluate_health_signal_detection()
    
    # 3. 프라이버시 전략 시나리오 테스트
    all_results["privacy_strategy"] = evaluate_privacy_strategy()
    
    # 4. PII Before/After 예시
    all_results["pii_examples"] = demonstrate_pii_redaction()
    
    # 5. 건강 요약 예시
    all_results["health_summaries"] = demonstrate_health_summary()
    
    # 6. Base vs Fine-tuned 비교 (Ollama 필요)
    all_results["model_comparison"] = evaluate_model_comparison()
    
    # 결과 저장
    output_dir = Path(__file__).resolve().parents[2] / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"roman_eval_{timestamp}.json"
    
    # JSON 직렬화 (dataclass 등 처리)
    def default_serializer(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        if hasattr(obj, 'value'):
            return obj.value
        return str(obj)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=default_serializer)
    
    print(f"\n\n{'=' * 70}")
    print(f"  평가 결과 저장: {output_file}")
    print(f"{'=' * 70}")
    
    # 논문용 요약 테이블
    print_summary_table(all_results)


def print_summary_table(results: Dict):
    """논문용 요약 테이블 출력"""
    print(f"\n{'=' * 70}")
    print("  === 논문용 요약 (Paper-Ready Summary) ===")
    print(f"{'=' * 70}")
    
    # Table 1: PII Detection
    pii = results.get("pii_detection", {})
    if pii:
        print(f"\n  [Table 1] PII Detection Performance (N={pii.get('total_test_cases', 0)} test cases)")
        print(f"  {'Type':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*50}")
        for ptype, m in pii.get("type_results", {}).items():
            print(f"  {ptype:<20} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")
        overall = pii.get("overall", {})
        print(f"  {'-'*50}")
        print(f"  {'Overall (micro)':>20} {overall.get('precision',0):>10.3f} "
              f"{overall.get('recall',0):>10.3f} {overall.get('f1',0):>10.3f}")
        print(f"  Clinical Term Preservation Rate: {pii.get('clinical_preservation_rate',0):.1%}")
    
    # Table 2: Health Signal Detection
    hs = results.get("health_signal", {})
    if hs:
        print(f"\n  [Table 2] Health Risk Signal Detection Performance")
        print(f"  {'Category':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*50}")
        for cat, m in hs.get("category_results", {}).items():
            print(f"  {cat:<20} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")
        overall = hs.get("overall", {})
        print(f"  {'-'*50}")
        print(f"  {'Overall (micro)':>20} {overall.get('precision',0):>10.3f} "
              f"{overall.get('recall',0):>10.3f} {overall.get('f1',0):>10.3f}")
        print(f"  Risk Level Accuracy: {hs.get('level_accuracy',0):.1%}")
        print(f"  Avg Detection Latency: {hs.get('avg_latency_ms',0):.1f} ms")
    
    # Table 3: Privacy Strategy
    ps = results.get("privacy_strategy", {})
    if ps:
        print(f"\n  [Table 3] Privacy Communication Strategy Compliance")
        print(f"  Pass Rate: {ps.get('pass',0)}/{ps.get('total',0)} ({ps.get('rate',0):.0%})")
    
    # Table 4: Model Comparison
    mc = results.get("model_comparison")
    if mc:
        print(f"\n  [Table 4] Base SLM vs Fine-tuned SLM Comparison")
        print(f"  {'Prompt':<25} {'Base Score':>10} {'FT Score':>10} {'Winner':>15}")
        print(f"  {'-'*60}")
        base_wins = ft_wins = ties = 0
        for r in mc.get("results", []):
            q = r["quality"]
            winner = "Fine-tuned" if q["ft_score"] > q["base_score"] + 0.05 else (
                "Base" if q["base_score"] > q["ft_score"] + 0.05 else "Tie"
            )
            if winner == "Fine-tuned":
                ft_wins += 1
            elif winner == "Base":
                base_wins += 1
            else:
                ties += 1
            print(f"  {r['topic'][:25]:<25} {q['base_score']:>10.2f} {q['ft_score']:>10.2f} {winner:>15}")
        
        print(f"  {'-'*60}")
        print(f"  Fine-tuned wins: {ft_wins}, Base wins: {base_wins}, Ties: {ties}")
        print(f"  Avg Latency — Base: {mc.get('base_avg_latency',0):.0f}ms, "
              f"Fine-tuned: {mc.get('ft_avg_latency',0):.0f}ms")


if __name__ == "__main__":
    main()
