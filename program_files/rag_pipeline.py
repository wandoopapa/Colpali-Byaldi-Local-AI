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
from typing import Callable, Iterable, List, Set


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


def _normalize_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _list_supported_documents(folder: Path) -> List[Path]:
    # 실행 GUI 요구사항: 사용자가 선택한 폴더에서 인덱싱 대상 파일을 항목 단위로 수집.
    supported_ext = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    docs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in supported_ext and p.stat().st_size > 0]
    # 안정성 개선: 문서 순서를 고정(재현성)하고 중복(realpath 동일) 제거.
    docs.sort(key=lambda p: p.name.lower())
    uniq: List[Path] = []
    seen: Set[str] = set()
    for p in docs:
        rp = _normalize_path(p)
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def _notify(progress: ProgressCallback, message: str, total_pct: float, item_pct: float) -> None:
    if progress:
        progress(message, total_pct, item_pct)


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _is_index_exists_error(text: str) -> bool:
    lowered = text.lower()
    return "already exists" in lowered and "overwrite=true" in lowered


def _is_missing_index_artifact_error(text: str) -> bool:
    lowered = text.lower()
    return "no such file or directory" in lowered and "index_config" in lowered


def _normalize_indexing_error(exc: Exception) -> Exception:
    text = str(exc)
    lowered = text.lower()

    if _is_index_exists_error(text):
        return RuntimeError(
            "동일한 인덱스가 이미 존재합니다. 기존 인덱스를 로딩해 append 모드로 처리해야 합니다. 인덱스 폴더 권한/손상 여부를 확인해 주세요."
        )
    if _is_missing_index_artifact_error(text):
        return RuntimeError(
            "기존 인덱스 메타파일(index_config)이 누락되었습니다. 손상된 인덱스 폴더를 정리한 뒤 다시 인덱싱을 실행해 주세요."
        )
    if "already loaded" in lowered and "add_to_index" in lowered:
        return RuntimeError(
            "동일한 인덱스가 이미 로딩되어 있습니다. add_to_index 모드로 재시도 중 오류가 발생했습니다. 인덱스 파일 무결성을 확인해 주세요."
        )
    if CUDA_ERROR_TEXT in text:
        return RuntimeError(
            "현재 Torch가 CUDA를 지원하지 않아 인덱싱에 실패했습니다. 설치 GUI를 다시 실행해 CUDA용 PyTorch를 설치한 뒤 재시도해 주세요."
        )
    if "out of memory" in lowered:
        return RuntimeError(
            "GPU 메모리 부족으로 인덱싱이 중단되었습니다. 문서 수를 나누어 실행하거나 다른 GPU 작업을 종료한 뒤 다시 시도해 주세요."
        )
    return exc


def _load_byaldi_pretrained(model_name: str, progress: ProgressCallback):
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


def _find_existing_index_path(index_folder: Path, index_name: str) -> Path | None:
    # 근본 원인 대응: manifest 유무가 아니라 index_config 실제 존재 여부로 판단.
    candidates = [index_folder / index_name]
    candidates.extend(p for p in index_folder.glob(f"{index_name}*") if p.is_dir())
    for c in candidates:
        if (c / "index_config.json.gz").exists() or (c / "index_config.json").exists():
            return c
    return None


def _load_byaldi_existing_index(index_folder: Path, index_name: str, progress: ProgressCallback):
    from byaldi import RAGMultiModalModel

    if not _torch_cuda_available():
        raise RuntimeError(
            "GPU(CUDA)를 사용할 수 없습니다. GPU 드라이버/CUDA 지원 PyTorch를 설치한 뒤 다시 시도해 주세요."
        )

    existing_index_path = _find_existing_index_path(index_folder, index_name)
    if existing_index_path is None:
        raise RuntimeError("기존 인덱스 메타파일(index_config.json.gz)을 찾지 못했습니다.")

    _notify(progress, f"기존 인덱스를 GPU 모드로 로딩 중... ({existing_index_path.name})", 6, 55)
    loader_variants: Iterable[dict] = [
        {"index_name": index_name, "index_root": _normalize_path(index_folder), "device": "cuda"},
        {"index_name": index_name, "index_root": _normalize_path(index_folder), "device_map": "cuda"},
        {"index_name": index_name, "index_root": _normalize_path(index_folder)},
        {"index_path": _normalize_path(existing_index_path), "device": "cuda"},
        {"index_path": _normalize_path(existing_index_path), "device_map": "cuda"},
        {"index_path": _normalize_path(existing_index_path)},
    ]

    last_error: Exception | None = None
    for kwargs in loader_variants:
        try:
            return RAGMultiModalModel.from_index(**kwargs), "cuda"
        except TypeError:
            continue
        except Exception as exc:
            last_error = exc
            if CUDA_ERROR_TEXT in str(exc):
                break

    raise _normalize_indexing_error(last_error or RuntimeError("기존 인덱스 로딩 실패"))


def _index_storage_exists(index_folder: Path, index_name: str) -> bool:
    return _find_existing_index_path(index_folder, index_name) is not None


def _load_indexed_document_set(index_folder: Path) -> Set[str]:
    # 사용자 요구사항: 이미 인덱싱 완료된 문서는 pass.
    manifest = index_folder / "index_manifest.json"
    if not manifest.exists():
        return set()

    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return set()

    docs = data.get("indexed_documents", [])
    if not isinstance(docs, list):
        return set()

    normalized: Set[str] = set()
    for item in docs:
        try:
            normalized.add(_normalize_path(Path(item)))
        except Exception:
            continue
    return normalized


def _save_manifest(config: PipelineConfig, docs: List[Path], device_used: str) -> None:
    # 실행 GUI 개선: QA 단계에서 인덱스 메타데이터를 재사용할 수 있도록 매니페스트 저장.
    manifest = {
        "index_name": config.index_name,
        "model_name": config.model_name,
        "device_used": device_used,
        "source_folder": _normalize_path(config.source_folder),
        "indexed_documents": sorted({_normalize_path(d) for d in docs}),
    }
    out = config.index_folder / "index_manifest.json"
    tmp = config.index_folder / "index_manifest.tmp.json"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)


def _index_doc_variants(doc_path: Path, config: PipelineConfig, append_mode: bool) -> Iterable[dict]:
    # API 차이 대응: add_to_index/overwrite/index_root 지원 여부가 버전마다 달라 순차 시도.
    if append_mode:
        return [
            {
                "input_path": _normalize_path(doc_path),
                "index_name": config.index_name,
                "index_root": _normalize_path(config.index_folder),
                "add_to_index": True,
                "store_collection_with_index": True,
                "overwrite": False,
            },
            {
                "input_path": _normalize_path(doc_path),
                "index_name": config.index_name,
                "index_root": _normalize_path(config.index_folder),
                "add_to_index": True,
                "overwrite": False,
            },
            {
                "input_path": _normalize_path(doc_path),
                "index_name": config.index_name,
                "add_to_index": True,
                "overwrite": False,
            },
        ]

    return [
        {
            "input_path": _normalize_path(doc_path),
            "index_name": config.index_name,
            "index_root": _normalize_path(config.index_folder),
            "store_collection_with_index": True,
            "overwrite": False,
        },
        {
            "input_path": _normalize_path(doc_path),
            "index_name": config.index_name,
            "index_root": _normalize_path(config.index_folder),
            "overwrite": False,
        },
        {
            "input_path": _normalize_path(doc_path),
            "index_name": config.index_name,
            "overwrite": False,
        },
    ]


def _index_single_doc(model, doc_path: Path, config: PipelineConfig, append_mode: bool, progress: ProgressCallback, base_total: float) -> None:
    last_error: Exception | None = None
    for kwargs in _index_doc_variants(doc_path, config, append_mode):
        try:
            _notify(progress, f"인덱싱 중: {doc_path.name}", base_total + 5, 60)
            model.index(**kwargs)
            return
        except TypeError:
            continue
        except Exception as exc:
            last_error = exc
            if _is_index_exists_error(str(exc)) or _is_missing_index_artifact_error(str(exc)):
                raise exc

    # 일부 버전은 index 대신 add_to_index 메서드를 제공할 수 있음.
    if append_mode and hasattr(model, "add_to_index"):
        _notify(progress, f"기존 인덱스 add_to_index 실행 중: {doc_path.name}", base_total + 6, 70)
        model.add_to_index(input_path=_normalize_path(doc_path), index_name=config.index_name)
        return

    if last_error:
        raise _normalize_indexing_error(last_error)
    raise RuntimeError(f"Byaldi index API 호출 실패: {doc_path.name}")


def _assert_index_folder_writable(index_folder: Path) -> None:
    probe = index_folder / ".write_test.tmp"
    try:
        probe.write_text("ok", encoding="utf-8")
    except Exception as exc:
        raise RuntimeError("인덱스 저장 폴더에 쓰기 권한이 없습니다. 다른 폴더를 선택해 주세요.") from exc
    finally:
        try:
            if probe.exists():
                probe.unlink()
        except Exception:
            pass


def run_indexing(config: PipelineConfig, progress: ProgressCallback) -> None:
    # 실행 GUI 요구사항: 전체/현재 항목 이중 진척률 콜백 제공.
    docs = _list_supported_documents(config.source_folder)
    if not docs:
        raise ValueError("선택한 폴더에 지원되는 문서 파일(PDF/이미지)이 없습니다. (빈 파일은 자동 제외)")

    config.index_folder.mkdir(parents=True, exist_ok=True)
    _assert_index_folder_writable(config.index_folder)

    _notify(progress, "환경 점검 중...", 2, 20)
    try:
        from byaldi import RAGMultiModalModel  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "byaldi 로드에 실패했습니다. 설치 GUI에서 의존성 설치를 먼저 완료하세요."
        ) from exc

    append_mode = _index_storage_exists(config.index_folder, config.index_name)
    if append_mode:
        _notify(progress, "기존 인덱스를 감지했습니다. 기존 인덱스를 직접 로딩합니다...", 4, 30)
        model, device_used = _load_byaldi_existing_index(config.index_folder, config.index_name, progress)
    else:
        _notify(progress, "새 인덱스를 생성하기 위해 모델 로딩 중 (최초 1회 오래 걸릴 수 있음)...", 5, 40)
        model, device_used = _load_byaldi_pretrained(config.model_name, progress)

    indexed_set = _load_indexed_document_set(config.index_folder)
    pending_docs: List[Path] = []
    for doc in docs:
        if _normalize_path(doc) in indexed_set:
            _notify(progress, f"PASS(기존 DB 존재): {doc.name}", 8, 100)
            continue
        pending_docs.append(doc)

    if not pending_docs:
        _notify(progress, "모든 문서가 이미 인덱싱되어 PASS 처리되었습니다.", 100, 100)
        return

    total_docs = len(pending_docs)
    for idx, doc_path in enumerate(pending_docs, start=1):
        base_total = 10 + ((idx - 1) / total_docs) * 80
        _notify(progress, f"문서 준비 중: {doc_path.name}", base_total, 20)

        try:
            _index_single_doc(model, doc_path, config, append_mode, progress, base_total)
        except Exception as exc:
            text = str(exc)
            if _is_index_exists_error(text) or _is_missing_index_artifact_error(text):
                _notify(progress, f"기존 인덱스 충돌/손상 감지, 기존 인덱스 재로딩 후 재시도: {doc_path.name}", base_total + 6, 70)
                model, device_used = _load_byaldi_existing_index(config.index_folder, config.index_name, progress)
                append_mode = True
                _index_single_doc(model, doc_path, config, append_mode, progress, base_total)
            else:
                raise _normalize_indexing_error(exc) from exc

        indexed_set.add(_normalize_path(doc_path))
        doc_total = 10 + (idx / total_docs) * 80
        _notify(progress, f"완료: {doc_path.name}", doc_total, 100)

    all_docs = [Path(p) for p in sorted(indexed_set)]
    _save_manifest(config, all_docs, device_used)
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
        {"index_name": index_name, "index_root": _normalize_path(index_folder)},
        {"index_name": index_name},
        {"index_path": _normalize_path(index_folder / index_name)},
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
