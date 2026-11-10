# myocr — PaddleOCR로 표 이미지를 검증 가능한 표로

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jkspawn/myocr/blob/main/paddleocr_table.ipynb)

스캔한 표 이미지를 PaddleOCR(영어 모델)로 읽고, 글자를 **행·열에 연결한 표**로 만든 뒤
원본 대조 결과와 실행 기록까지 남기는 Colab 노트북입니다.

## 바로 실행

1. 위 **Open in Colab** 배지를 누릅니다.
2. **런타임 → 모두 실행**을 누릅니다. CPU 런타임이면 충분합니다.
3. 마지막 셀이 `outputs/` 전체를 `ocr_outputs.zip`으로 내려받습니다.

입력 이미지는 노트북 안에 들어 있어 따로 올릴 필요가 없습니다.

## 노트북 단계

| 단계 | 하는 일 |
|---|---|
| 1 | `paddlepaddle`, `paddleocr` 설치, 시작 시각 기록 |
| 2 · 2-1 | 입력 이미지 풀기, **출처와 원문 페이지** 적기 (`SOURCES`) |
| 3 | 영어 인식 모델 내려받기 (`lang="en"`) |
| 4 | 글자 인식, 결과 저장 (`outputs/ocr/`) |
| 5 | 상자를 행 단위로 묶기 |
| 6-1 · 6-2 | **원본 대조 기록** (`REVIEW`), Table 1 행·열 연결과 문서 만들기 (`outputs/table1/`) |
| 7 | 손으로 검증한 정답과 비교 |
| 8 | 실행 기록 `outputs/manifest.json` |
| 9 | 결과 zip 내려받기 |

## 만들어지는 파일

```
outputs/
├── manifest.json               출처·원문 페이지·입력 파일 해시·모델·패키지 버전·실행 위치·설정
├── ocr/
│   ├── input/                  입력 이미지 사본
│   ├── raw/<이름>_ocr_raw.json  PaddleOCR 원시 결과
│   ├── text/<이름>_text_boxes.csv  글자 덩어리마다 id·텍스트·신뢰도·좌표
│   ├── vis/<이름>_regions.png   인식 영역에 id를 붙여 그린 이미지
│   └── tables/<이름>.csv        행 단위로 묶은 표
└── table1/
    ├── table.csv               행(parent_id)·열(머리글)을 연결한 Table 1
    ├── review_items.csv        확인 필요 항목과 상태
    ├── changes.csv             수정 내역 (자동·검토자)
    └── document.md             표 · 합계 검증 · 확인 필요 · 수정 내역 · OCR 출처
```

## 원본 대조 흐름

1. 처음에는 6-1 셀의 `REVIEW`를 비워 둔 채 실행합니다.
2. 6-2 셀이 확인이 필요한 칸을 출력합니다. 기준은 신뢰도 0.9 미만, 빈 칸, 상자 합침, 각주 번호 자동 분리, 합계 불일치입니다. 같은 출력에 `REVIEW`에 붙여 넣을 줄도 나옵니다.
3. 원본 이미지와 대조한 뒤 조치를 적습니다.
   - `"확인"`: 원본과 같음
   - `"수정"`: 새 값으로 고침
   - `"확인 필요"`: 직접 표시
4. 6-1 셀과 6-2 셀을 다시 실행하면 `document.md`의 확인 필요·수정 내역과 `table.csv`가 바뀝니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `paddleocr_table.ipynb` | Colab 노트북 (입력 이미지를 base64로 포함) |
| `input/` | 입력 표 이미지 (`census-table.png`, `fraser-table.png`) |
| `output/census-table.{md,csv}` | 사람이 옮기고 합계를 검산한 Table 1 (정답 기준) |
| `tools/build_notebook.py` | 노트북을 다시 만드는 스크립트: `python tools/build_notebook.py .` |

## 검증 상태

- Colab에서 받은 PaddleOCR 결과(PP-OCRv5 검출 + 영어 인식)를 현재 5~8단계에 넣어 돌리면, Table 1 **11개 행이 모두 정답과 일치**합니다.
  각주 위첨자가 붙어 읽힌 `17,635,426`, `262,810`은 합계 검증으로 `7,635,426`, `62,810`으로 바로잡히고, 수정 내역에 남습니다.
- 현재 노트북 전체를 Colab에서 처음부터 끝까지 돌려 본 적은 아직 없습니다.
- **출처 미기재:** 두 이미지의 원문 자료명, 쪽 번호, 주소는 아직 적혀 있지 않습니다. 2-1 셀 `SOURCES`에 채워 주세요.
- 6단계 구조화는 Table 1(3열, 총계·소계 구조)에 맞춰져 있습니다. `fraser-table`은 4단계까지만 처리합니다.
