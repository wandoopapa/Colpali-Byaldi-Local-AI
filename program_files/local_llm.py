"""
4. 구성
* 프로그램 구성
- 프로그램 코드에는 '4. 구성'의 모든 내용이 코드 최상단에 전체 내용 기입 + 관련 코드 항목에 각각 코멘트 기입 필요 : 최종 : 최상단에 구성 전체 내용 기입 + 각 항목별 코멘트 기입 (다른 ai 혹은 프로그래머가 코드만 보고 코드 작성 가이드를 따를 수 있어야 함)
- 사용자는 설치 파일만 실행, 나머지 설치 과정은 파이썬 코드가 자동으로 진행
- '설치 GUI', '실행 GUI' 로 구분
- 모든 과정은 GUI 를 적용하여 .pyw 로 cmd 창 없이 GUI로 진행
- '설치 GUI' 파일을 실행하면, [폴더 생성 -> 파일 생성 -> 필요 라이브러리 자동 다운로드 및 설치] 까지 원클릭 진행
- 설치가 완료 후, 설치된 '실행 GUI' 파일로 프로그램을 실행
- 소스 코드는 반드시 영어(English)로 작성하여 코드 호환성 유지
- GUI 출력(라벨, 버튼, 알림창)은 반드시 한국어(Korean)로 작성하여 사용자 편의성 극대화
- 역할, 파트 별로 파일을 나누어서 1개의 폴더 안에 저장

* '설치 GUI'
- 라이브러리가 설치되는 과정을 검은색 CMD 창이 아닌, '설치 GUI' 로딩 창(Progress Bar)으로 표현
- 모든 설치 과정은 항목 별로 총 작업량 대비 현재 작업량을 퍼센티지화 하여 진척률을 확인할 수 있는 시각적 이미지 적용
- 설치 진행률은 전체 항목과 현재 항목을 구분하여 Progress Bar 를 각각 적용
- 설치 폴더를 '설치 GUI' 안의 버튼을 이용하여 사용자가 지정
- 결과 파일은 지정 된 설치 폴더에 하위폴더 없이 저장

* 실행 GUI
- Indexing 이 진행되는 과정을 검은색 CMD 창이 아닌, '실행 GUI' 로딩 창(Progress Bar)으로 표현
- 작업 과정은 항목 별로 총 작업량 대비 현재 작업량을 퍼센티지화 하여 진척률을 확인할 수 있는 시각적 이미지 적용
- 작업 진행률은 전체 항목과 현재 항목을 구분하여 Progress Bar 를 각각 적용
- 원본 파일을 불러오는 폴더를 '실행 GUI' 상에서 사용자가 지정
- 프로그램을 제작하는 과정에서 누적된 패치 및 업데이트 사항에 대한 모든 내용을 누락없이 표시 (패치 및 업데이트 페이지는 버전별로 폴더형식으로 눌러서 열어볼 수 있또록 누락없이 모든 내용 기입,핵심 요약이 아니라 관련 내용이 자세하게 기입)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import requests


@dataclass
class LLMConfig:
    backend: str
    endpoint: str
    model_name: str
    temperature: float = 0.1


def list_models(backend: str, endpoint: str) -> List[str]:
    # 실행 GUI 요구사항: Local LLM 로딩 전에 백엔드별 모델 목록 조회를 지원.
    endpoint = endpoint.rstrip("/")
    if backend == "ollama":
        response = requests.get(f"{endpoint}/api/tags", timeout=30)
        response.raise_for_status()
        return [m.get("name", "") for m in response.json().get("models", []) if m.get("name")]

    response = requests.get(f"{endpoint}/v1/models", timeout=30)
    response.raise_for_status()
    return [m.get("id", "") for m in response.json().get("data", []) if m.get("id")]


def warmup_model(config: LLMConfig) -> str:
    # 실행 GUI 요구사항: 선택된 Local LLM 모델을 실제 추론 호출로 로딩(웜업).
    if config.backend == "ollama":
        payload = {
            "model": config.model_name,
            "prompt": "안녕하세요",
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": config.temperature, "num_predict": 1},
        }
        response = requests.post(f"{config.endpoint.rstrip('/')}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        return "Ollama 모델 로딩이 완료되었습니다."

    payload = {
        "model": config.model_name,
        "messages": [{"role": "user", "content": "안녕하세요"}],
        "temperature": config.temperature,
        "max_tokens": 1,
    }
    response = requests.post(f"{config.endpoint.rstrip('/')}/v1/chat/completions", json=payload, timeout=120)
    response.raise_for_status()
    return "LM Studio(OpenAI 호환) 모델 로딩이 완료되었습니다."


def chat(config: LLMConfig, user_question: str, context_chunks: List[str]) -> Tuple[str, str]:
    # 실행 GUI 요구사항: 질의응답 페이지에서 RAG 컨텍스트를 포함한 답변 생성.
    context = "\n\n".join(context_chunks) if context_chunks else "검색된 컨텍스트가 없습니다."
    system_prompt = (
        "당신은 로컬 멀티모달 RAG 도우미입니다. "
        "제공된 컨텍스트를 최우선으로 사용하고, 모르면 모른다고 답변하세요."
    )
    user_prompt = f"[질문]\n{user_question}\n\n[컨텍스트]\n{context}"

    if config.backend == "ollama":
        payload = {
            "model": config.model_name,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": config.temperature},
            "prompt": f"시스템: {system_prompt}\n\n사용자: {user_prompt}",
        }
        response = requests.post(f"{config.endpoint.rstrip('/')}/api/generate", json=payload, timeout=300)
        response.raise_for_status()
        answer = response.json().get("response", "응답이 비어 있습니다.")
        return answer, context

    payload = {
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.temperature,
    }
    response = requests.post(f"{config.endpoint.rstrip('/')}/v1/chat/completions", json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "응답이 비어 있습니다.")
    return answer, context
