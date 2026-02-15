# Colpali-Byaldi-Local-AI

Windows 사용자 기준으로 **설치 GUI(`installer_gui.pyw`)** 와 **실행 GUI(`runtime_gui.pyw`)** 를 제공하는 로컬 멀티모달 RAG(ColPali + Byaldi + Local LLM) 프로젝트입니다.

## 빠른 시작
1. `installer_gui.pyw` 실행
2. 설치 폴더 선택 후 `원클릭 설치 시작`
3. 설치 완료 후 설치 폴더의 `runtime_gui.pyw` 실행
4. `인덱싱` 탭에서 원본 문서 폴더 선택 후 인덱싱 시작
5. `로컬 LLM 질의응답` 탭에서 백엔드(Ollama/LM Studio)와 모델을 로딩하고 질의 실행

## 권장 로컬 스택 (무료)
- Indexing/Retrieval: `Byaldi + ColPali`
- Local LLM backend:
  - `Ollama` (기본 주소 `http://127.0.0.1:11434`)
  - `LM Studio` OpenAI 호환 API (기본 주소 `http://127.0.0.1:1234`)
- Optional vector stack: `Qdrant Local`

## 기본 포함 파일
- `installer_gui.pyw`: 설치 전용 GUI
- `program_files/runtime_gui.pyw`: 실행 전용 GUI (인덱싱 + 로컬 LLM QA)
- `program_files/rag_pipeline.py`: ColPali + Byaldi 인덱싱/검색 파이프라인
- `program_files/local_llm.py`: Ollama/LM Studio 모델 조회·로딩·질의 함수
- `program_files/patch_notes.py`: 버전별 패치/업데이트 상세 내역
- `program_files/requirements.txt`: 자동 설치 라이브러리 목록

## GitHub 브랜치/PR 업로드 체크리스트
로컬에서 커밋이 있어도 원격 저장소(`origin`)가 없으면 GitHub에서 브랜치/PR을 확인할 수 없습니다.

```bash
git remote -v
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin work
```

- `git remote -v`가 비어 있으면 먼저 원격 URL을 추가해야 합니다.
- 인증 실패 시 GitHub PAT(토큰) 권한(`repo`)과 계정 접근 권한을 확인하세요.
