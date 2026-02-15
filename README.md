# Colpali-Byaldi-Local-AI

Windows 사용자 기준으로 **설치 GUI(`installer_gui.pyw`)** 와 **실행 GUI(`runtime_gui.pyw`)** 를 제공하는 로컬 멀티모달 RAG(ColPali + Byaldi) 프로젝트입니다.

## 빠른 시작
1. `installer_gui.pyw` 실행
2. 설치 폴더 선택 후 `원클릭 설치 시작`
3. 설치 완료 후 설치 폴더의 `runtime_gui.pyw` 실행
4. 실행 GUI에서 원본 문서 폴더 선택 후 인덱싱 시작

## 권장 로컬 스택 (무료)
- Indexing/Retrieval: `Byaldi + ColPali`
- Vector store: `Qdrant Local`
- Local LLM serving: `Ollama`

## 기본 포함 파일
- `installer_gui.pyw`: 설치 전용 GUI
- `program_files/runtime_gui.pyw`: 실행 전용 GUI
- `program_files/rag_pipeline.py`: ColPali + Byaldi 인덱싱 파이프라인
- `program_files/patch_notes.py`: 버전별 패치/업데이트 상세 내역
- `program_files/requirements.txt`: 자동 설치 라이브러리 목록
