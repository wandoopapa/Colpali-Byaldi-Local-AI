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

import os
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


class InstallerApp(tk.Tk):
    # 설치 GUI 요구사항: 폴더 선택 + 파일 생성 + 라이브러리 설치를 원클릭으로 자동 수행.
    def __init__(self) -> None:
        super().__init__()
        self.title("ColPali + Byaldi 설치 GUI")
        self.geometry("840x520")

        self.target_dir = tk.StringVar(value=str((Path.home() / "ColpaliByaldiLocalAI").resolve()))
        self.status_text = tk.StringVar(value="대기 중")
        self.total_progress = tk.DoubleVar(value=0)
        self.item_progress = tk.DoubleVar(value=0)

        self.worker: threading.Thread | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        frm = ttk.Frame(self, padding=14)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="설치 폴더").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.target_dir, width=95).grid(row=1, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(frm, text="폴더 선택", command=self._select_folder).grid(row=1, column=1)

        ttk.Label(frm, text="전체 진행률").grid(row=2, column=0, pady=(24, 0), sticky="w")
        ttk.Progressbar(frm, variable=self.total_progress, maximum=100).grid(row=3, column=0, columnspan=2, sticky="ew")

        ttk.Label(frm, text="현재 항목 진행률").grid(row=4, column=0, pady=(14, 0), sticky="w")
        ttk.Progressbar(frm, variable=self.item_progress, maximum=100).grid(row=5, column=0, columnspan=2, sticky="ew")

        ttk.Label(frm, textvariable=self.status_text).grid(row=6, column=0, columnspan=2, pady=(12, 0), sticky="w")

        ttk.Button(frm, text="원클릭 설치 시작", command=self._start_install).grid(row=7, column=0, pady=(24, 0), sticky="w")
        frm.columnconfigure(0, weight=1)

    def _select_folder(self) -> None:
        selected = filedialog.askdirectory(title="설치 폴더 선택")
        if selected:
            self.target_dir.set(selected)

    def _start_install(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("안내", "이미 설치가 진행 중입니다.")
            return

        target = Path(self.target_dir.get().strip())
        target.mkdir(parents=True, exist_ok=True)

        self.worker = threading.Thread(target=self._run_install, args=(target,), daemon=True)
        self.worker.start()

    def _set_progress(self, message: str, total: float, item: float) -> None:
        self.after(0, lambda: self.status_text.set(message))
        self.after(0, lambda: self.total_progress.set(total))
        self.after(0, lambda: self.item_progress.set(item))

    def _run_install(self, target: Path) -> None:
        try:
            payload = Path(__file__).resolve().parent / "program_files"
            if not payload.exists():
                raise FileNotFoundError("program_files 폴더를 찾을 수 없습니다.")

            files = [p for p in payload.iterdir() if p.is_file()]
            task_count = len(files) + 1

            self._set_progress("설치 폴더 준비 중...", 5, 30)

            # 설치 GUI 요구사항: 결과 파일을 지정 폴더에 하위폴더 없이 배치.
            for i, src in enumerate(files, start=1):
                dst = target / src.name
                shutil.copy2(src, dst)
                total_pct = 10 + (i / task_count) * 50
                item_pct = (i / len(files)) * 100
                self._set_progress(f"파일 생성/복사 중: {src.name}", total_pct, item_pct)

            self._set_progress("라이브러리 설치 준비 중...", 65, 5)
            req_file = target / "requirements.txt"
            self._install_requirements(req_file)

            self._set_progress("설치 완료", 100, 100)
            self.after(0, lambda: messagebox.showinfo("완료", f"설치가 완료되었습니다.\n실행 파일: {target / 'runtime_gui.pyw'}"))
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("설치 오류", str(exc)))

    def _install_requirements(self, requirements_path: Path) -> None:
        # 설치 GUI 요구사항: CMD 대신 GUI 상태 텍스트/프로그레스만으로 진행 상황 안내.
        commands = [
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
        ]

        for idx, cmd in enumerate(commands, start=1):
            self._set_progress(
                f"라이브러리 설치 중 ({idx}/{len(commands)})...",
                65 + (idx - 1) * 15,
                30,
            )

            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=creationflags,
            )
            output_lines = 0
            while True:
                line = proc.stdout.readline() if proc.stdout else ""
                if not line and proc.poll() is not None:
                    break
                if line:
                    output_lines += 1
                    # 항목 진행률은 설치 로그를 읽을 때마다 완만하게 증가시켜 시각화.
                    item_pct = min(95, 30 + output_lines)
                    total_pct = 65 + (idx - 1) * 15 + min(14, output_lines * 0.2)
                    self._set_progress(line.strip()[:120], total_pct, item_pct)

            if proc.returncode != 0:
                raise RuntimeError(f"라이브러리 설치 실패: {' '.join(cmd)}")

        self._set_progress("라이브러리 설치 완료", 95, 100)


if __name__ == "__main__":
    app = InstallerApp()
    app.mainloop()
