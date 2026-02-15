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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List


ProgressCallback = Callable[[str, float, float], None]
DEFAULT_INDEX_NAME = "colpali_byaldi_local_index"
DEFAULT_MODEL_NAME = "vidore/colpali-v1.3"
CUDA_ERROR_TEXT = "Torch not compiled with CUDA enabled"


@dataclass
class PipelineConfig:
    source_folder: Path
    index_folder: Path
    index_name: str = DEFAULT_INDEX_NAME
    model_name: str = DEFAULT_MODEL_NAME


def _list_supported_documents(folder: Path) -> List[Path]:
    # 실행 GUI 요구사항: 사용자가 선택한 폴더에서 인덱싱 대상 파일을 항목 단위로 수집.
    supported_ext = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in supported_ext]


def _notify(progress: ProgressCallback, message: str, total_pct: float, item_pct: float) -> None:
    if progress:
        progress(message, total_pct, item_pct)


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load_byaldi_model(model_name: str, progress: ProgressCallback):
    from byaldi import RAGMultiModalModel

    # 사용자 요구사항: 인덱싱은 반드시 GPU(CUDA)로만 진행.
    if not _torch_cuda_available():
        raise RuntimeError(
            "GPU(CUDA)를 사용할 수 없습니다. GPU 드라이버/CUDA 지원 PyTorch를 설치한 뒤 다시 시도해 주세요."
        )

    _notify(progress, "GPU(CUDA) 모드로 모델 로딩 중...", 5, 50)
    cuda_variants: Iterable[dict] = [{"device": "cuda"}, {"device_map": "cuda"}, {}]
    last_error: Exception | None = None
    for kwargs in cuda_variants:
        try:
            return RAGMultiModalModel.from_pretrained(model_name, **kwargs), "cuda"
        except TypeError:
            continue
        except Exception as exc:
            last_error = exc
            if CUDA_ERROR_TEXT in str(exc):
                break

    if last_error and CUDA_ERROR_TEXT in str(last_error):
        raise RuntimeError(
            "Torch CUDA 빌드 오류가 발생했습니다. 설치 GUI를 다시 실행해 CUDA용 PyTorch를 설치한 뒤 재시도해 주세요."
        ) from last_error
    raise RuntimeError("Byaldi GPU 모델 로딩에 실패했습니다. CUDA/드라이버/의존성 버전을 확인해 주세요.") from last_error


def _save_manifest(config: PipelineConfig, docs: List[Path], device_used: str) -> None:
    # 실행 GUI 개선: QA 단계에서 인덱스 메타데이터를 재사용할 수 있도록 매니페스트 저장.
    manifest = {
        "index_name": config.index_name,
        "model_name": config.model_name,
        "device_used": device_used,
        "source_folder": str(config.source_folder),
        "indexed_documents": [str(d) for d in docs],
    }
    (config.index_folder / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_indexing(config: PipelineConfig, progress: ProgressCallback) -> None:
    # 실행 GUI 요구사항: 전체/현재 항목 이중 진척률 콜백 제공.
    docs = _list_supported_documents(config.source_folder)
    if not docs:
        raise ValueError("선택한 폴더에 지원되는 문서 파일(PDF/이미지)이 없습니다.")

    config.index_folder.mkdir(parents=True, exist_ok=True)

    _notify(progress, "환경 점검 중...", 2, 20)
    try:
        from byaldi import RAGMultiModalModel
    except Exception as exc:
        raise RuntimeError(
            "byaldi 로드에 실패했습니다. 설치 GUI에서 의존성 설치를 먼저 완료하세요."
        ) from exc

    _notify(progress, "모델 로딩 중 (최초 1회 오래 걸릴 수 있음)...", 5, 40)
    model, device_used = _load_byaldi_model(config.model_name, progress)

    total_docs = len(docs)
    for idx, doc_path in enumerate(docs, start=1):
        # 실행 GUI 요구사항: 현재 항목 진행률을 파일 단위 퍼센티지로 계산.
        base_total = 10 + ((idx - 1) / total_docs) * 80
        _notify(progress, f"문서 준비 중: {doc_path.name}", base_total, 20)

        # 품질 우선 인덱싱: overwrite=False로 누적 색인 가능 (무한 인덱싱 시간 조건 반영).
        # Byaldi 버전 차이를 고려해 여러 시그니처를 순차 시도.
        indexed = False
        call_variants: Iterable[dict] = [
            {
                "input_path": str(doc_path),
                "index_name": config.index_name,
                "index_root": str(config.index_folder),
                "store_collection_with_index": True,
                "overwrite": False,
            },
            {
                "input_path": str(doc_path),
                "index_name": config.index_name,
                "index_root": str(config.index_folder),
                "overwrite": False,
            },
            {
                "input_path": str(doc_path),
                "index_name": config.index_name,
                "overwrite": False,
            },
        ]

        for kwargs in call_variants:
            try:
                _notify(progress, f"인덱싱 중: {doc_path.name}", base_total + 5, 60)
                model.index(**kwargs)
                indexed = True
                break
            except TypeError:
                continue
            except Exception as exc:
                if CUDA_ERROR_TEXT in str(exc):
                    raise RuntimeError(
                        "현재 Torch가 CUDA를 지원하지 않아 인덱싱에 실패했습니다. 설치 GUI를 다시 실행해 CUDA용 PyTorch를 설치한 뒤 재시도해 주세요."
                    ) from exc
                raise

        if not indexed:
            raise RuntimeError(f"Byaldi index API 호출 실패: {doc_path.name}")

        doc_total = 10 + (idx / total_docs) * 80
        _notify(progress, f"완료: {doc_path.name}", doc_total, 100)

    _save_manifest(config, docs, device_used)
    _notify(progress, "최종 저장 및 검증 중...", 98, 100)
    _notify(progress, f"인덱싱이 완료되었습니다. (사용 장치: {device_used.upper()})", 100, 100)


def retrieve_context(index_folder: Path, question: str, top_k: int = 3) -> List[str]:
    # 실행 GUI 개선: Local LLM QA 페이지에서 Byaldi 검색 결과를 컨텍스트 텍스트로 변환.
    manifest_file = index_folder / "index_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError("index_manifest.json이 없습니다. 먼저 인덱싱을 실행해 주세요.")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    index_name = manifest.get("index_name", DEFAULT_INDEX_NAME)

    from byaldi import RAGMultiModalModel

    model = None
    loader_variants = [
        {"index_name": index_name, "index_root": str(index_folder)},
        {"index_name": index_name},
        {"index_path": str(index_folder / index_name)},
    ]
    for kwargs in loader_variants:
        try:
            model = RAGMultiModalModel.from_index(**kwargs)
            break
        except TypeError:
            continue

    if model is None:
        raise RuntimeError("Byaldi 인덱스 로딩 실패: 버전에 맞는 from_index 시그니처를 확인해 주세요.")

    hits = model.search(question, k=top_k)
    contexts: List[str] = []
    for i, hit in enumerate(hits, start=1):
        contexts.append(
            f"[검색결과 {i}] 파일: {hit.get('doc_name', 'unknown')} | 페이지: {hit.get('page_num', 'unknown')} | 점수: {hit.get('score', 'n/a')}"
        )
    return contexts
