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

import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from local_llm import LLMConfig, chat, list_models, warmup_model
from patch_notes import PATCH_NOTES
from rag_pipeline import DEFAULT_INDEX_NAME, PipelineConfig, retrieve_context, run_indexing


class RuntimeApp(tk.Tk):
    # 실행 GUI 요구사항: 인덱싱/로컬 LLM 질의응답/업데이트 페이지를 GUI 탭으로 제공.
    def __init__(self) -> None:
        super().__init__()
        self.title("ColPali + Byaldi 실행 GUI")
        self.geometry("1100x760")

        self.event_queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None

        self.source_dir = tk.StringVar(value="")
        self.index_dir = tk.StringVar(value=str((Path.cwd() / "index_output").resolve()))
        self.index_name = tk.StringVar(value=DEFAULT_INDEX_NAME)

        self.backend = tk.StringVar(value="ollama")
        self.endpoint = tk.StringVar(value="http://127.0.0.1:11434")
        self.model_name = tk.StringVar(value="")

        self.status_text = tk.StringVar(value="대기 중")
        self.total_progress = tk.DoubleVar(value=0)
        self.item_progress = tk.DoubleVar(value=0)

        self.qa_status = tk.StringVar(value="로컬 LLM 대기 중")
        self._build_layout()
        self.after(100, self._poll_events)

    def _build_layout(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=12, pady=12)

        index_tab = ttk.Frame(notebook)
        qa_tab = ttk.Frame(notebook)
        patch_tab = ttk.Frame(notebook)
        notebook.add(index_tab, text="인덱싱")
        notebook.add(qa_tab, text="로컬 LLM 질의응답")
        notebook.add(patch_tab, text="패치/업데이트")

        self._build_index_tab(index_tab)
        self._build_qa_tab(qa_tab)
        self._build_patch_tab(patch_tab)

    def _build_index_tab(self, parent: ttk.Frame) -> None:
        frm = ttk.Frame(parent, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="원본 문서 폴더").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.source_dir, width=90).grid(row=1, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(frm, text="폴더 선택", command=self._choose_source).grid(row=1, column=1)

        ttk.Label(frm, text="인덱스 저장 폴더").grid(row=2, column=0, pady=(12, 0), sticky="w")
        ttk.Entry(frm, textvariable=self.index_dir, width=90).grid(row=3, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(frm, text="폴더 선택", command=self._choose_index).grid(row=3, column=1)

        # 사용자 요청 반영: 인덱스 이름은 입력칸이 아닌 출력(고정) 정보로 제공.
        ttk.Label(frm, text="인덱스 이름 (출력)").grid(row=4, column=0, pady=(12, 0), sticky="w")
        ttk.Entry(frm, textvariable=self.index_name, state="readonly", width=50).grid(row=5, column=0, sticky="w")

        ttk.Label(frm, text="전체 진행률").grid(row=6, column=0, pady=(20, 0), sticky="w")
        ttk.Progressbar(frm, variable=self.total_progress, maximum=100).grid(row=7, column=0, columnspan=2, sticky="ew")

        ttk.Label(frm, text="현재 항목 진행률").grid(row=8, column=0, pady=(12, 0), sticky="w")
        ttk.Progressbar(frm, variable=self.item_progress, maximum=100).grid(row=9, column=0, columnspan=2, sticky="ew")

        ttk.Label(frm, textvariable=self.status_text).grid(row=10, column=0, columnspan=2, pady=(12, 0), sticky="w")

        btn_row = ttk.Frame(frm)
        btn_row.grid(row=11, column=0, columnspan=2, pady=(10, 0), sticky="w")
        ttk.Button(btn_row, text="인덱싱 시작", command=self._start_indexing).pack(side="left")
        ttk.Button(btn_row, text="로그 지우기", command=self._clear_index_log).pack(side="left", padx=(8, 0))

        ttk.Label(frm, text="인덱싱 로그").grid(row=12, column=0, pady=(14, 0), sticky="w")
        log_frame = ttk.Frame(frm)
        log_frame.grid(row=13, column=0, columnspan=2, sticky="nsew")
        self.index_log = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.index_log.yview)
        self.index_log.configure(yscrollcommand=log_scroll.set)
        self.index_log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(13, weight=1)

    def _build_qa_tab(self, parent: ttk.Frame) -> None:
        # 사용자 요청 반영: Local LLM 로딩 + 질의응답 전용 페이지.
        frm = ttk.Frame(parent, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="LLM 백엔드").grid(row=0, column=0, sticky="w")
        backend_combo = ttk.Combobox(frm, textvariable=self.backend, values=["ollama", "lmstudio"], state="readonly", width=20)
        backend_combo.grid(row=1, column=0, sticky="w")
        backend_combo.bind("<<ComboboxSelected>>", self._on_backend_change)

        ttk.Label(frm, text="백엔드 주소").grid(row=0, column=1, sticky="w")
        ttk.Entry(frm, textvariable=self.endpoint, width=46).grid(row=1, column=1, sticky="ew", padx=(8, 8))

        ttk.Button(frm, text="모델 목록 조회", command=self._fetch_models).grid(row=1, column=2)

        ttk.Label(frm, text="모델 선택").grid(row=2, column=0, pady=(12, 0), sticky="w")
        self.model_combo = ttk.Combobox(frm, textvariable=self.model_name, values=[], width=50)
        self.model_combo.grid(row=3, column=0, columnspan=2, sticky="w")

        ttk.Button(frm, text="모델 로딩", command=self._warmup_model).grid(row=3, column=2)

        ttk.Label(frm, text="질문 입력").grid(row=4, column=0, pady=(16, 0), sticky="w")
        self.question_text = tk.Text(frm, height=4)
        self.question_text.grid(row=5, column=0, columnspan=3, sticky="nsew")

        ttk.Button(frm, text="질의 실행", command=self._ask_question).grid(row=6, column=0, pady=(10, 0), sticky="w")
        ttk.Label(frm, textvariable=self.qa_status).grid(row=6, column=1, columnspan=2, padx=(8, 0), sticky="w")

        ttk.Label(frm, text="검색 컨텍스트").grid(row=7, column=0, pady=(16, 0), sticky="w")
        self.context_text = tk.Text(frm, height=8)
        self.context_text.grid(row=8, column=0, columnspan=3, sticky="nsew")

        ttk.Label(frm, text="LLM 응답").grid(row=9, column=0, pady=(16, 0), sticky="w")
        self.answer_text = tk.Text(frm, height=10)
        self.answer_text.grid(row=10, column=0, columnspan=3, sticky="nsew")

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(10, weight=1)

    def _build_patch_tab(self, parent: ttk.Frame) -> None:
        # 실행 GUI 요구사항: 버전 폴더를 펼쳐 상세 변경점 전체를 확인할 수 있게 Treeview 구성.
        frm = ttk.Frame(parent, padding=12)
        frm.pack(fill="both", expand=True)

        tree = ttk.Treeview(frm)
        tree.pack(fill="both", expand=True)
        tree.heading("#0", text="패치/업데이트 내역 (버전 폴더 펼치기)")

        for version, meta in sorted(PATCH_NOTES.items(), reverse=True):
            root = tree.insert("", "end", text=f"{version} | {meta['date']} | {meta['title']}")
            for i, detail in enumerate(meta["details"], start=1):
                tree.insert(root, "end", text=f"{i:02d}. {detail}")

    def _append_index_log(self, message: str) -> None:
        # 실행 GUI 개선: 인덱싱 진행률과 함께 상세 로그를 누적 표시.
        ts = datetime.now().strftime("%H:%M:%S")
        self.index_log.configure(state="normal")
        self.index_log.insert("end", f"[{ts}] {message}\n")
        self.index_log.see("end")
        self.index_log.configure(state="disabled")

    def _clear_index_log(self) -> None:
        self.index_log.configure(state="normal")
        self.index_log.delete("1.0", "end")
        self.index_log.configure(state="disabled")

    def _choose_source(self) -> None:
        selected = filedialog.askdirectory(title="원본 문서 폴더 선택")
        if selected:
            self.source_dir.set(selected)

    def _choose_index(self) -> None:
        selected = filedialog.askdirectory(title="인덱스 저장 폴더 선택")
        if selected:
            self.index_dir.set(selected)

    def _on_backend_change(self, _event: object | None = None) -> None:
        if self.backend.get() == "ollama":
            self.endpoint.set("http://127.0.0.1:11434")
        else:
            self.endpoint.set("http://127.0.0.1:1234")

    def _start_indexing(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("안내", "이미 작업이 실행 중입니다.")
            return

        source = Path(self.source_dir.get().strip())
        if not source.exists() or not source.is_dir():
            messagebox.showerror("오류", "유효한 원본 문서 폴더를 선택해 주세요.")
            return

        raw_index_dir = self.index_dir.get().strip()
        if not raw_index_dir:
            messagebox.showerror("오류", "인덱스 저장 폴더를 선택해 주세요.")
            return
        index_folder = Path(raw_index_dir)

        config = PipelineConfig(
            source_folder=source,
            index_folder=index_folder,
            index_name=self.index_name.get(),
        )

        self.total_progress.set(0)
        self.item_progress.set(0)
        self.status_text.set("인덱싱 시작 준비 중...")
        self._clear_index_log()
        self._append_index_log(f"인덱싱 시작 | source={source} | index={index_folder}")

        def worker_job() -> None:
            try:
                run_indexing(config, self._enqueue_progress)
                self.event_queue.put(("done", "인덱싱이 완료되었습니다."))
            except Exception as exc:
                self.event_queue.put(("error", str(exc)))

        self.worker = threading.Thread(target=worker_job, daemon=True)
        self.worker.start()

    def _fetch_models(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("안내", "이미 작업이 실행 중입니다.")
            return

        def worker_job() -> None:
            try:
                self.event_queue.put(("qa_status", "모델 목록 조회 중..."))
                models = list_models(self.backend.get(), self.endpoint.get().strip())
                self.event_queue.put(("models", models))
            except Exception as exc:
                self.event_queue.put(("qa_error", str(exc)))

        self.worker = threading.Thread(target=worker_job, daemon=True)
        self.worker.start()

    def _warmup_model(self) -> None:
        model = self.model_name.get().strip()
        if not model:
            messagebox.showerror("오류", "먼저 모델을 선택해 주세요.")
            return

        def worker_job() -> None:
            try:
                self.event_queue.put(("qa_status", "모델 로딩 중..."))
                message = warmup_model(
                    LLMConfig(
                        backend=self.backend.get(),
                        endpoint=self.endpoint.get().strip(),
                        model_name=model,
                    )
                )
                self.event_queue.put(("qa_status", message))
            except Exception as exc:
                self.event_queue.put(("qa_error", str(exc)))

        self.worker = threading.Thread(target=worker_job, daemon=True)
        self.worker.start()

    def _ask_question(self) -> None:
        question = self.question_text.get("1.0", "end").strip()
        if not question:
            messagebox.showerror("오류", "질문을 입력해 주세요.")
            return
        if not self.model_name.get().strip():
            messagebox.showerror("오류", "Local LLM 모델을 먼저 선택/로딩해 주세요.")
            return

        index_path = Path(self.index_dir.get().strip())

        def worker_job() -> None:
            try:
                self.event_queue.put(("qa_status", "인덱스 검색 중..."))
                contexts = retrieve_context(index_path, question, top_k=3)
                self.event_queue.put(("contexts", contexts))

                self.event_queue.put(("qa_status", "LLM 답변 생성 중..."))
                answer, context_text = chat(
                    LLMConfig(
                        backend=self.backend.get(),
                        endpoint=self.endpoint.get().strip(),
                        model_name=self.model_name.get().strip(),
                    ),
                    question,
                    contexts,
                )
                self.event_queue.put(("answer", answer, context_text))
                self.event_queue.put(("qa_status", "질의응답 완료"))
            except Exception as exc:
                self.event_queue.put(("qa_error", str(exc)))

        self.worker = threading.Thread(target=worker_job, daemon=True)
        self.worker.start()

    def _enqueue_progress(self, message: str, total_pct: float, item_pct: float) -> None:
        self.event_queue.put(("progress", message, total_pct, item_pct))

    def _poll_events(self) -> None:
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break

            kind = event[0]
            if kind == "progress":
                _, msg, total_pct, item_pct = event
                self.status_text.set(msg)
                self.total_progress.set(total_pct)
                self.item_progress.set(item_pct)
                self._append_index_log(msg)
            elif kind == "done":
                self.status_text.set(event[1])
                self.total_progress.set(100)
                self.item_progress.set(100)
                self._append_index_log(event[1])
                messagebox.showinfo("완료", event[1])
            elif kind == "error":
                self.status_text.set("오류 발생")
                self._append_index_log(f"오류: {event[1]}")
                messagebox.showerror("오류", event[1])
            elif kind == "qa_status":
                self.qa_status.set(event[1])
            elif kind == "qa_error":
                self.qa_status.set("오류 발생")
                messagebox.showerror("로컬 LLM 오류", event[1])
            elif kind == "models":
                models = event[1]
                self.model_combo["values"] = models
                if models:
                    self.model_name.set(models[0])
                self.qa_status.set(f"모델 {len(models)}개 조회 완료")
            elif kind == "contexts":
                contexts = event[1]
                self.context_text.delete("1.0", "end")
                self.context_text.insert("end", "\n".join(contexts))
            elif kind == "answer":
                _, answer, context_text = event
                self.answer_text.delete("1.0", "end")
                self.answer_text.insert("end", answer)
                if not self.context_text.get("1.0", "end").strip():
                    self.context_text.insert("end", context_text)

        self.after(100, self._poll_events)


if __name__ == "__main__":
    app = RuntimeApp()
    app.mainloop()
