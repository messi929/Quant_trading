# -*- coding: utf-8 -*-
"""Generate kr_sector_map.yaml from ticker_info + expanded keywords + manual overrides."""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
ti = pd.read_parquet(ROOT / "data/raw/ticker_info.parquet")
kr = ti[ti["market"].isin(["KOSPI", "KOSDAQ"])].copy()

# ===== Expanded keyword mapping =====
expanded_keywords = {
    "energy": [
        "에너지", "석유", "가스", "정유", "오일", "연료", "광물", "탄광", "발전",
        "원유", "LNG", "LPG", "석탄", "플랜트", "에탄올",
    ],
    "materials": [
        "화학", "철강", "소재", "비철", "금속", "알루미늄", "구리", "아연", "니켈",
        "시멘트", "유리", "도료", "접착제", "플라스틱", "제강", "제지", "펄프",
        "케미칼", "첨단소재", "스틸", "하이텍", "세라믹", "고무", "섬유", "실리콘",
        "폴리", "페인트", "카본", "석회", "광업", "타이어", "화성", "수지",
        "포장재", "필름", "코팅", "합성", "섬유", "원단",
    ],
    "industrials": [
        "건설", "기계", "조선", "운송", "항공", "물류", "엔지니어링", "설비",
        "장비", "도로", "철도", "항만", "중공업", "방산", "군수", "해운",
        "모빌리티", "오토모티브", "모터스", "이앤씨", "전기", "일렉트릭",
        "마린", "선박", "산업", "교통", "택배", "네트웍스", "인터내셔널",
        "건설기계", "산업개발", "전선", "중전기", "엔진", "터빈", "밸브",
        "배관", "파이프", "크레인", "승강기", "엘리베이터", "냉동", "공조",
        "보일러", "펌프", "압축기", "측량", "토목", "강업", "주철관",
        "토건", "방직", "제분", "연마", "실업",
    ],
    "consumer_discretionary": [
        "자동차", "의류", "호텔", "레저", "백화점", "패션",
        "스포츠", "여행", "관광", "음식점", "외식", "가구", "인테리어",
        "쇼핑", "면세", "렌탈", "리테일", "홈쇼핑", "뷰티", "아모레",
        "코스메틱", "교육", "완구", "카지노", "시계", "보석", "귀금속",
        "골프", "피트니스", "엔터테인",
    ],
    "consumer_staples": [
        "식품", "음료", "담배", "가정용품", "화장품", "생활용품",
        "농업", "수산", "유제품", "과자", "라면", "씨푸드", "축산", "사료",
        "제당", "제과", "유업", "양조", "주류", "맥주", "소주", "커피",
    ],
    "healthcare": [
        "제약", "바이오", "의료기기", "헬스케어", "병원", "진단", "치료제", "백신",
        "의약품", "임상", "신약", "유전자", "세포치료", "생명과학", "메디", "약품",
        "팜", "파마", "헬스", "의료", "덴탈", "치과", "안과", "피부",
    ],
    "financials": [
        "은행", "증권", "보험", "금융", "카드", "캐피탈", "저축", "자산운용",
        "신탁", "리스", "여신", "신용", "홀딩스", "지주",
    ],
    "information_technology": [
        "반도체", "전자", "소프트웨어", "IT", "통신장비", "디스플레이",
        "배터리", "OLED", "LCD", "메모리", "시스템반도체", "파운드리",
        "PCB", "카메라모듈", "세미콘", "데이터", "클라우드",
        "테크", "이노텍", "SDS", "SDI", "플랫폼", "소프트", "컴퓨터",
        "서버", "네트워크", "보안", "암호", "AI", "로봇",
    ],
    "communication_services": [
        "통신", "엔터", "게임", "인터넷", "방송", "포털", "콘텐츠",
        "OTT", "SNS", "광고", "영화", "음악", "미디어", "텔레비전",
    ],
    "utilities": [
        "전력", "수도", "유틸리티", "원자력", "신재생", "태양광", "풍력",
        "에너지솔루션", "지역난방",
    ],
    "real_estate": [
        "부동산", "리츠", "REITs", "건물관리", "임대", "분양", "아파트",
        "오피스", "물류센터", "스퀘어리츠", "스타리츠",
    ],
}

# ===== Manual overrides for major companies =====
manual_map = {
    # 삼성 그룹
    "005930.KS": "information_technology",
    "005935.KS": "information_technology",
    "006400.KS": "information_technology",  # 삼성SDI
    "009150.KS": "information_technology",  # 삼성전기
    "018260.KS": "information_technology",  # 삼성SDS
    "207940.KS": "healthcare",  # 삼성바이오로직스
    "032830.KS": "financials",  # 삼성생명
    "000810.KS": "financials",  # 삼성화재
    "016360.KS": "financials",  # 삼성증권
    "028260.KS": "industrials",  # 삼성물산
    "010140.KS": "industrials",  # 삼성중공업
    # SK 그룹
    "034730.KS": "information_technology",  # SK
    "000660.KS": "information_technology",  # SK하이닉스
    "017670.KS": "communication_services",  # SK텔레콤
    "034020.KS": "industrials",  # 두산에너빌리티
    "326030.KS": "healthcare",  # SK바이오팜
    "302440.KS": "healthcare",  # SK바이오사이언스
    # 현대/기아
    "005380.KS": "consumer_discretionary",
    "005385.KS": "consumer_discretionary",
    "005387.KS": "consumer_discretionary",
    "000270.KS": "consumer_discretionary",  # 기아
    "012330.KS": "consumer_discretionary",  # 현대모비스
    "267250.KS": "industrials",  # HD현대
    "329180.KS": "industrials",  # HD현대중공업
    "009540.KS": "industrials",  # HD한국조선해양
    "267270.KS": "industrials",  # HD건설기계
    "267260.KS": "industrials",  # HD현대일렉트릭
    "443060.KS": "industrials",  # HD현대마린솔루션
    "071970.KS": "industrials",  # HD현대마린엔진
    "294870.KS": "industrials",  # HDC현대산업개발
    "204320.KS": "consumer_discretionary",  # HL만도
    # LG
    "003550.KS": "industrials",  # LG
    "066570.KS": "information_technology",  # LG전자
    "051910.KS": "materials",  # LG화학
    "051915.KS": "materials",  # LG화학우
    "373220.KS": "information_technology",  # LG에너지솔루션
    "011070.KS": "information_technology",  # LG이노텍
    "032640.KS": "communication_services",  # LG유플러스
    "051900.KS": "consumer_staples",  # LG생활건강
    "051905.KS": "consumer_staples",  # LG생활건강우
    "064400.KS": "information_technology",  # LG씨엔에스
    "037560.KS": "communication_services",  # LG헬로비전
    "383800.KS": "industrials",  # LX홀딩스
    "108320.KS": "information_technology",  # LX세미콘
    "001120.KS": "industrials",  # LX인터내셔널
    "108670.KS": "industrials",  # LX하우시스
    # POSCO
    "005490.KS": "materials",  # POSCO홀딩스
    "003670.KS": "materials",  # 포스코퓨처엠
    "047050.KS": "industrials",  # 포스코인터내셔널
    # 금융
    "105560.KS": "financials",  # KB금융
    "055550.KS": "financials",  # 신한지주
    "086790.KS": "financials",  # 하나금융지주
    "316140.KS": "financials",  # 우리금융지주
    "024110.KS": "financials",  # 기업은행
    "138930.KS": "financials",  # BNK금융지주
    "175330.KS": "financials",  # JB금융지주
    "139130.KS": "financials",  # DGB금융지주
    # 통신/인터넷
    "035420.KS": "communication_services",  # NAVER
    "035720.KS": "communication_services",  # 카카오
    "030200.KS": "communication_services",  # KT
    "036570.KS": "communication_services",  # 엔씨소프트
    "181710.KS": "information_technology",  # NHN
    "352820.KS": "communication_services",  # 하이브
    "259960.KS": "communication_services",  # 크래프톤
    # CJ
    "001040.KS": "consumer_staples",  # CJ
    "097950.KS": "consumer_staples",  # CJ제일제당
    "079160.KS": "communication_services",  # CJ CGV
    "000120.KS": "industrials",  # CJ대한통운
    # 한화
    "000880.KS": "industrials",  # 한화
    "012450.KS": "industrials",  # 한화에어로스페이스
    "009830.KS": "materials",  # 한화솔루션
    "042660.KS": "industrials",  # 한화오션
    "272210.KS": "industrials",  # 한화시스템
    # GS
    "078930.KS": "energy",  # GS
    "006360.KS": "industrials",  # GS건설
    "001250.KS": "industrials",  # GS글로벌
    "007070.KS": "consumer_staples",  # GS리테일
    # LS
    "006260.KS": "industrials",  # LS
    "010120.KS": "industrials",  # LS ELECTRIC
    "000680.KS": "industrials",  # LS네트웍스
    # 기타 대형주
    "011200.KS": "industrials",  # HMM
    "003490.KS": "industrials",  # 대한항공
    "180640.KS": "industrials",  # 한진칼
    "010130.KS": "materials",  # 고려아연
    "017940.KS": "energy",  # E1
    "010950.KS": "energy",  # S-Oil
    "010955.KS": "energy",  # S-Oil우
    "000210.KS": "industrials",  # DL
    "375500.KS": "industrials",  # DL이앤씨
    "000100.KS": "healthcare",  # 유한양행
    "128940.KS": "healthcare",  # 한미약품
    "068270.KS": "healthcare",  # 셀트리온
    "383220.KS": "consumer_discretionary",  # F&F
    "007700.KS": "consumer_discretionary",  # F&F홀딩스
    "004170.KS": "consumer_discretionary",  # 신세계
    "139480.KS": "consumer_discretionary",  # 이마트
    "090430.KS": "consumer_discretionary",  # 아모레퍼시픽
    "002790.KS": "consumer_discretionary",  # 아모레G
    "282330.KS": "consumer_staples",  # BGF리테일
    "027410.KS": "consumer_staples",  # BGF
    "034120.KS": "communication_services",  # SBS
    "036460.KS": "utilities",  # 한국가스공사
    "015760.KS": "utilities",  # 한국전력
    "114090.KS": "consumer_discretionary",  # GKL
    "079550.KS": "industrials",  # LIG넥스원
    "456040.KS": "materials",  # OCI
    "010060.KS": "materials",  # OCI홀딩스
    "002380.KS": "materials",  # KCC
    "344820.KS": "materials",  # KCC글라스
    "093050.KS": "consumer_discretionary",  # LF
    "034310.KS": "information_technology",  # NICE
    "030190.KS": "information_technology",  # NICE평가정보
    "007340.KS": "consumer_discretionary",  # DN오토모티브
    "003580.KS": "healthcare",  # HLB글로벌
    "060980.KS": "industrials",  # HL홀딩스
    "014790.KS": "industrials",  # HL D&I
    "012030.KS": "industrials",  # DB (DB그룹 지주)
    "005830.KS": "financials",  # DB손해보험
    "016610.KS": "financials",  # DB증권
    "000990.KS": "information_technology",  # DB하이텍
    "095570.KS": "industrials",  # AJ네트웍스
    "006840.KS": "industrials",  # AK홀딩스
    "001460.KS": "consumer_discretionary",  # BYC
    "000480.KS": "industrials",  # CR홀딩스
    "000590.KS": "industrials",  # CS홀딩스
    "001530.KS": "industrials",  # DI동일
    "015590.KS": "industrials",  # DKME
    "004840.KS": "industrials",  # DRB동일
    "155660.KS": "industrials",  # DSR
    "017860.KS": "energy",  # DS단석
    "092780.KS": "materials",  # DYP
    "365550.KS": "real_estate",  # ESR켄달스퀘어리츠
    "012630.KS": "industrials",  # HDC
    "039570.KS": "information_technology",  # HDC랩스
    "089470.KS": "materials",  # HDC현대EP
    "035000.KS": "communication_services",  # HS애드
    "002460.KS": "materials",  # HS화성
    "487570.KS": "industrials",  # HS효성
    "298050.KS": "materials",  # HS효성첨단소재
    "015360.KS": "industrials",  # INVENI
    "234080.KS": "healthcare",  # JW생명과학
    "001060.KS": "healthcare",  # JW중외제약
    "096760.KS": "healthcare",  # JW홀딩스
    "009070.KS": "industrials",  # KCTC
    "009440.KS": "industrials",  # KC그린홀딩스
    "119650.KS": "industrials",  # KC코트렐
    "092220.KS": "information_technology",  # KEC
    "003620.KS": "consumer_discretionary",  # KG모빌리티
    "016380.KS": "materials",  # KG스틸
    "001390.KS": "materials",  # KG케미칼
    "001940.KS": "materials",  # KISCO홀딩스
    "025000.KS": "materials",  # KPX케미칼
    "092230.KS": "materials",  # KPX홀딩스
    "000040.KS": "consumer_discretionary",  # KR모터스
    "044450.KS": "industrials",  # KSS해운
    "008970.KS": "materials",  # KBI동양철관
    "415640.KS": "real_estate",  # KB발해인프라
    "432320.KS": "real_estate",  # KB스타리츠
    "023150.KS": "materials",  # MH에탄올
    "008260.KS": "materials",  # NI스틸
    "004250.KS": "materials",  # NPC
    "001340.KS": "industrials",  # PKC
    "003080.KS": "materials",  # SB성보
    "001380.KS": "industrials",  # SG글로벌
    "322000.KS": "energy",  # HD현대에너지솔루션
    # 추가 대형주
    "000050.KS": "consumer_discretionary",  # 경방 (섬유→소비재)
    "000070.KS": "consumer_staples",  # 삼양홀딩스
    "000080.KS": "consumer_staples",  # 하이트진로
    "000140.KS": "consumer_staples",  # 하이트진로홀딩스
    "000150.KS": "industrials",  # 두산
    "000180.KS": "industrials",  # 성창기업지주
    "000230.KS": "healthcare",  # 일동홀딩스
    "000240.KS": "consumer_discretionary",  # 한국앤컴퍼니
    "000320.KS": "materials",  # 노루홀딩스 (페인트)
    "000490.KS": "industrials",  # 대동
    "000540.KS": "financials",  # 흥국화재
    "000640.KS": "healthcare",  # 동아쏘시오홀딩스
    "000650.KS": "industrials",  # 천일고속
    "000670.KS": "materials",  # 영풍
    "000700.KS": "financials",  # 유수홀딩스
    "000850.KS": "industrials",  # 화천기공
    "000860.KS": "industrials",  # 강남제비스코
    "000910.KS": "materials",  # 유니온
    "000950.KS": "materials",  # 전방 (섬유)
    "000970.KS": "materials",  # 한국주철관
    "001020.KS": "materials",  # 페이퍼코리아
    "001070.KS": "consumer_discretionary",  # 대한방직
    "001130.KS": "consumer_staples",  # 대한제분
    "001230.KS": "materials",  # 동국홀딩스 (철강)
    "001260.KS": "industrials",  # 남광토건
    "001420.KS": "industrials",  # 태원물산
    "001450.KS": "financials",  # 현대해상
    "001470.KS": "industrials",  # 삼부토건
    "001520.KS": "industrials",  # 동양
    "001550.KS": "materials",  # 조비
    "001560.KS": "materials",  # 제일연마
    "001570.KS": "materials",  # 금양
    "001620.KS": "industrials",  # 케이비아이동국실업
    "001680.KS": "consumer_staples",  # 대상
    "001740.KS": "consumer_staples",  # SK네트웍스
    "001800.KS": "consumer_staples",  # 오리온홀딩스
    "002020.KS": "materials",  # 코오롱
    "002030.KS": "consumer_staples",  # 아세아
    "002070.KS": "consumer_staples",  # 남영비비안
    "002170.KS": "industrials",  # 삼양통상
    "002270.KS": "consumer_staples",  # 롯데푸드 -> 롯데웰푸드
    "002350.KS": "materials",  # 넥센타이어
    "002600.KS": "consumer_discretionary",  # 조흥
    "002700.KS": "materials",  # 신일전자
    "002710.KS": "information_technology",  # TCC스틸
    "002710.KS": "materials",
    "002840.KS": "materials",  # 미원상사
    "002990.KS": "industrials",  # 금호건설
    "003000.KS": "materials",  # 부광약품
    "003000.KS": "healthcare",
    "003030.KS": "industrials",  # 세아제강지주
    "003090.KS": "industrials",  # 대웅
    "003200.KS": "industrials",  # 일신방직
    "003220.KS": "consumer_staples",  # 대원미디어
    "003220.KS": "communication_services",
    "003230.KS": "industrials",  # 삼양식품
    "003230.KS": "consumer_staples",
    "003240.KS": "industrials",  # 태광산업
    "003300.KS": "consumer_staples",  # 한일시멘트
    "003300.KS": "materials",
    "003350.KS": "consumer_staples",  # 한국식품
    "003350.KS": "consumer_staples",
    "003460.KS": "materials",  # 유화증권
    "003460.KS": "financials",
    "003540.KS": "consumer_staples",  # 대신증권
    "003540.KS": "financials",
    "003610.KS": "industrials",  # 방림
    "003720.KS": "industrials",  # 삼영전자
    "003720.KS": "information_technology",
    "003850.KS": "materials",  # 보령
    "003850.KS": "healthcare",
    "004000.KS": "materials",  # 롯데정밀화학
    "004020.KS": "materials",  # 현대제철
    "004060.KS": "materials",  # SG세계물산
    "004090.KS": "consumer_staples",  # 한국석유공업
    "004090.KS": "energy",
    "004150.KS": "consumer_discretionary",  # 한솔케미칼
    "004150.KS": "materials",
    "004370.KS": "materials",  # 농심홀딩스
    "004370.KS": "consumer_staples",
    "004490.KS": "industrials",  # 세방전지
    "004490.KS": "information_technology",
    "004560.KS": "materials",  # 현대비앤지스틸
    "004800.KS": "materials",  # 효성
    "004800.KS": "industrials",
    "004910.KS": "consumer_staples",  # 조광페인트
    "004910.KS": "materials",
    "004970.KS": "consumer_staples",  # 신라교역
    "005070.KS": "industrials",  # 코스모신소재
    "005070.KS": "materials",
    "005180.KS": "financials",  # 빙그레
    "005180.KS": "consumer_staples",
    "005250.KS": "industrials",  # 녹십자홀딩스
    "005250.KS": "healthcare",
    "005300.KS": "industrials",  # 롯데칠성음료
    "005300.KS": "consumer_staples",
    "005440.KS": "industrials",  # 현대그린푸드
    "005440.KS": "consumer_staples",
    "005610.KS": "industrials",  # SPC삼립
    "005610.KS": "consumer_staples",
    "005680.KS": "industrials",  # 삼보산업
    "005740.KS": "industrials",  # 크라운해태홀딩스
    "005740.KS": "consumer_staples",
    "005850.KS": "industrials",  # 에스엘
    "005850.KS": "consumer_discretionary",
    "005940.KS": "industrials",  # NH투자증권
    "005940.KS": "financials",
    "006280.KS": "financials",  # 녹십자
    "006280.KS": "healthcare",
    "006650.KS": "energy",  # 대한유화
    "006650.KS": "materials",
    # KOSDAQ 주요
    "247540.KQ": "materials",  # 에코프로비엠
    "086520.KQ": "materials",  # 에코프로
    "196170.KQ": "healthcare",  # 알테오젠
    "403870.KQ": "information_technology",  # HPSP
    "058470.KQ": "information_technology",  # 리노공업
    "263750.KQ": "communication_services",  # 펄어비스
    "112040.KQ": "communication_services",  # 위메이드
    "293490.KQ": "communication_services",  # 카카오게임즈
    "357780.KQ": "healthcare",  # 솔바이오
    "950160.KQ": "healthcare",  # 코오롱티슈진
    "041510.KQ": "information_technology",  # 에스엠
    "035900.KQ": "communication_services",  # JYP Ent.
    "122870.KQ": "information_technology",  # 와이지엔터테인먼트
    "122870.KQ": "communication_services",
    "328130.KQ": "information_technology",  # 루닛
    "039030.KQ": "information_technology",  # 이오테크닉스
    "240810.KQ": "information_technology",  # 원익IPS
    "078600.KQ": "information_technology",  # 대주전자재료
    "222160.KQ": "materials",  # NPX
    "950210.KQ": "information_technology",  # 프레스티지바이오파마
    "950210.KQ": "healthcare",
    "145020.KQ": "healthcare",  # 휴젤
    "067160.KQ": "healthcare",  # 아프리카TV -> communication
    "067160.KQ": "communication_services",
    "036930.KQ": "information_technology",  # 주성엔지니어링
    "336260.KQ": "information_technology",  # 두산테스나
    "131970.KQ": "information_technology",  # 테스나
    "178920.KQ": "information_technology",  # PI첨단소재 -> materials
    "178920.KQ": "materials",
    "060310.KQ": "information_technology",  # 3S
    "095340.KQ": "information_technology",  # ISC
    "357550.KQ": "information_technology",  # 석경에이티
    "272110.KQ": "information_technology",  # 케이엔솔
    "054950.KQ": "information_technology",  # 제비스코
    "060250.KQ": "information_technology",  # NHN KCP
    "251970.KQ": "information_technology",  # 펌텍코리아
    "251970.KQ": "materials",
    "323990.KQ": "information_technology",  # 박셀바이오
    "323990.KQ": "healthcare",
    "141080.KQ": "information_technology",  # 레고켐바이오
    "141080.KQ": "healthcare",
    "214150.KQ": "healthcare",  # 클래시스
    "137310.KQ": "healthcare",  # 에스디바이오센서
    "314930.KQ": "healthcare",  # 바이오노트
    "226330.KQ": "information_technology",  # 쏠리드
    "033640.KQ": "information_technology",  # 네패스
    "036540.KQ": "information_technology",  # SFA반도체
    "253450.KQ": "information_technology",  # 스튜디오드래곤
    "253450.KQ": "communication_services",
    "215600.KQ": "information_technology",  # 신라젠
    "215600.KQ": "healthcare",
}


def classify_expanded(name):
    if not isinstance(name, str) or not name:
        return "unknown"
    best, best_c = "unknown", 0
    for sk, kws in expanded_keywords.items():
        c = sum(1 for kw in kws if kw in name)
        if c > best_c:
            best_c = c
            best = sk
    return best


# First pass: expanded keywords
kr["sector"] = kr["name"].apply(classify_expanded)

# Second pass: apply manual overrides
for yf_ticker, sector in manual_map.items():
    mask = kr["yf_ticker"] == yf_ticker
    if mask.any():
        kr.loc[mask, "sector"] = sector

# Third pass: preferred stock inheritance
# 우선주(우, 1우, 2우B, 3우B, 4우(전환)) inherits sector from common stock
import re

# Build base ticker -> sector map (common stocks only)
base_sectors = {}
for _, r in kr.iterrows():
    if r["sector"] != "unknown":
        base_sectors[r["yf_ticker"]] = r["sector"]

# For each unknown, try to find its common stock
for idx, r in kr[kr["sector"] == "unknown"].iterrows():
    yf = r["yf_ticker"]
    # Extract base code: "000815.KS" -> try "000810.KS" (strip last digit variants)
    # Korean preferred stock patterns: code ends with 5,7,K,L or has 우/B in name
    code, suffix = yf.split(".")
    name = r["name"] if isinstance(r["name"], str) else ""

    # Try common patterns for preferred stocks
    base_candidates = []
    if code.endswith("5") or code.endswith("7"):
        base_candidates.append(code[:-1] + "0." + suffix)
    if code.endswith("K") or code.endswith("L"):
        base_candidates.append(code[:-1] + "0." + suffix)
    # Check if name contains 우 (preferred) - inherit from name without 우
    if "우" in name or "B" in name:
        # Find any ticker with same prefix
        prefix = code[:5]
        for base_yf, sec in base_sectors.items():
            if base_yf.startswith(prefix) and base_yf.endswith(suffix):
                base_candidates.append(base_yf)

    for candidate in base_candidates:
        if candidate in base_sectors:
            kr.loc[idx, "sector"] = base_sectors[candidate]
            break

# Stats after
for m in ["KOSPI", "KOSDAQ"]:
    sub = kr[kr["market"] == m]
    classified = (sub["sector"] != "unknown").sum()
    print(f"{m}: {classified}/{len(sub)} classified ({classified/len(sub)*100:.1f}%)")

# Generate YAML-compatible dict: {yf_ticker: sector}
mapping = {}
for _, r in kr.iterrows():
    if r["sector"] != "unknown":
        mapping[r["yf_ticker"]] = r["sector"]

print(f"\nTotal mapped: {len(mapping)}")

# Save as YAML
output_path = ROOT / "config" / "kr_sector_map.yaml"
output = {
    "_description": "Static ticker -> GICS sector mapping for KOSPI/KOSDAQ",
    "_generated": "2026-03-18",
    "_note": "Primary source for sector classification. Keywords fallback for unmapped tickers.",
    "mapping": mapping,
}

with open(output_path, "w", encoding="utf-8") as f:
    yaml.dump(output, f, allow_unicode=True, default_flow_style=False, sort_keys=True)

print(f"Saved to {output_path}")

# Also show sector distribution
print("\nSector distribution (mapped tickers):")
sector_counts = kr[kr["sector"] != "unknown"].groupby("sector").size().sort_values(ascending=False)
for s, c in sector_counts.items():
    print(f"  {s}: {c}")
