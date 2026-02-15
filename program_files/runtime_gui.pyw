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
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from patch_notes import PATCH_NOTES
from rag_pipeline import PipelineConfig, run_indexing


class RuntimeApp(tk.Tk):
    # 실행 GUI 요구사항: 전체/현재 항목 진행률을 GUI Progress Bar로 이중 표시.
    def __init__(self) -> None:
        super().__init__()
        self.title("ColPali + Byaldi 실행 GUI")
        self.geometry("980x720")

        self.event_queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None

        self.source_dir = tk.StringVar(value="")
        self.index_dir = tk.StringVar(value=str((Path.cwd() / "index_output").resolve()))
        self.index_name = tk.StringVar(value="colpali_byaldi_local_index")

        self.status_text = tk.StringVar(value="대기 중")
        self.total_progress = tk.DoubleVar(value=0)
        self.item_progress = tk.DoubleVar(value=0)

        self._build_layout()
        self.after(100, self._poll_events)

    def _build_layout(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=12, pady=12)

        index_tab = ttk.Frame(notebook)
        patch_tab = ttk.Frame(notebook)
        notebook.add(index_tab, text="인덱싱")
        notebook.add(patch_tab, text="패치/업데이트")

        self._build_index_tab(index_tab)
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

        ttk.Label(frm, text="인덱스 이름").grid(row=4, column=0, pady=(12, 0), sticky="w")
        ttk.Entry(frm, textvariable=self.index_name, width=50).grid(row=5, column=0, sticky="w")

        ttk.Label(frm, text="전체 진행률").grid(row=6, column=0, pady=(20, 0), sticky="w")
        ttk.Progressbar(frm, variable=self.total_progress, maximum=100).grid(row=7, column=0, columnspan=2, sticky="ew")

        ttk.Label(frm, text="현재 항목 진행률").grid(row=8, column=0, pady=(12, 0), sticky="w")
        ttk.Progressbar(frm, variable=self.item_progress, maximum=100).grid(row=9, column=0, columnspan=2, sticky="ew")

        ttk.Label(frm, textvariable=self.status_text).grid(row=10, column=0, columnspan=2, pady=(12, 0), sticky="w")

        ttk.Button(frm, text="인덱싱 시작", command=self._start_indexing).grid(row=11, column=0, pady=(20, 0), sticky="w")

        frm.columnconfigure(0, weight=1)

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

    def _choose_source(self) -> None:
        selected = filedialog.askdirectory(title="원본 문서 폴더 선택")
        if selected:
            self.source_dir.set(selected)

    def _choose_index(self) -> None:
        selected = filedialog.askdirectory(title="인덱스 저장 폴더 선택")
        if selected:
            self.index_dir.set(selected)

    def _start_indexing(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("안내", "이미 인덱싱이 실행 중입니다.")
            return

        source = Path(self.source_dir.get().strip())
        if not source.exists() or not source.is_dir():
            messagebox.showerror("오류", "유효한 원본 문서 폴더를 선택해 주세요.")
            return

        config = PipelineConfig(
            source_folder=source,
            index_folder=Path(self.index_dir.get().strip()),
            index_name=self.index_name.get().strip() or "colpali_byaldi_local_index",
        )

        self.total_progress.set(0)
        self.item_progress.set(0)
        self.status_text.set("인덱싱 시작 준비 중...")

        def worker_job() -> None:
            try:
                run_indexing(config, self._enqueue_progress)
                self.event_queue.put(("done", "인덱싱이 완료되었습니다."))
            except Exception as exc:
                self.event_queue.put(("error", str(exc)))

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
            elif kind == "done":
                self.status_text.set(event[1])
                self.total_progress.set(100)
                self.item_progress.set(100)
                messagebox.showinfo("완료", event[1])
            elif kind == "error":
                self.status_text.set("오류 발생")
                messagebox.showerror("오류", event[1])

        self.after(100, self._poll_events)


if __name__ == "__main__":
    app = RuntimeApp()
    app.mainloop()
