# -*- coding: utf-8 -*-
import sys
import os
import cv2
import time
import json
import traceback
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox, QTableWidget,
    QTableWidgetItem, QMessageBox, QTabWidget, QGroupBox, QDialog, QFormLayout,
    QDialogButtonBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ==========================================
# 全局异常拦截 (防闪退)
# ==========================================
def excepthook(exc_type, exc_value, exc_tb):
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    with open("error_log.txt", "w", encoding="utf-8") as f:
        f.write(error_msg)
    QMessageBox.critical(None, "程序崩溃", f"详情请查看 error_log.txt\n{exc_value}")
sys.excepthook = excepthook

# ==========================================
# 配置与数据管理
# ==========================================
DATA_FILE = "study_data.json"
# 模拟书籍序列号数据库
BOOK_DATABASE = {
    "SN1001": "高等数学（上册）",
    "SN1002": "大学英语综合教程",
    "SN1003": "计算机网络原理",
    "SN1004": "马克思主义基本原理"
}

class DataManager:
    def __init__(self):
        self.data = {"records": []}
        self.load()

    def load(self):
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                pass

    def save(self):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)

    def add_record(self, record):
        self.data["records"].append(record)
        self.save()

    def update_record(self, index, updated_record):
        if 0 <= index < len(self.data["records"]):
            self.data["records"][index] = updated_record
            self.save()

dm = DataManager()

# ==========================================
# 视觉监控与分神检测线程
# ==========================================
class MonitorThread(QThread):
    frame_ready = pyqtSignal(QImage)
    status_update = pyqtSignal(str, float, int) # 状态文本, 专注度, 剩余时间
    alert_triggered = pyqtSignal(str)

    def __init__(self, total_seconds):
        super().__init__()
        self.total_seconds = total_seconds
        self.running = False
        self.is_paused = False
        self.time_elapsed = 0
        self.focus_time = 0
        self._last_alert_time = 0
        self._writer = None
        self.clip_dir = "distractions"
        os.makedirs(self.clip_dir, exist_ok=True)

    def run(self):
        # 尝试打开摄像头，使用 DIRECTSHOW 后端提高 Windows 兼容性
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.alert_triggered.emit("无法打开摄像头，请检查设备连接！")
            return

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.running = True

        while self.running:
            if self.is_paused:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret: break
            
            self.time_elapsed += 0.03
            remain_time = max(0, self.total_seconds - self.time_elapsed)
            
            # 1. 人脸检测判定专注度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            
            if len(faces) > 0:
                self.focus_time += 0.03
                status_text = "专注学习中"
            else:
                status_text = "⚠️ 未检测到人脸，请回到屏幕前"

            focus_score = (self.focus_time / self.time_elapsed) * 100 if self.time_elapsed > 0 else 100.0
            self.status_update.emit(status_text, focus_score, int(remain_time))

            # 2. 触发分神报警与录像 (每 5 秒触发一次)
            if len(faces) == 0 and (time.time() - self._last_alert_time > 5.0):
                self._last_alert_time = time.time()
                self.alert_triggered.emit("检测到分神或离开，已录制片段！")
                self._record_video(frame)

            # 3. 录像写入与结束
            if self._writer is not None:
                self._writer.write(frame)
                if time.time() - self._last_alert_time > 3.0: # 录制3秒
                    self._writer.release()
                    self._writer = None

            # 4. 发送画面给 UI
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_ready.emit(qt_img)

            # 5. 时间到
            if self.time_elapsed >= self.total_seconds:
                break
                
            time.sleep(0.03)

        cap.release()
        if self._writer:
            self._writer.release()
        self.running = False

    def _record_video(self, frame):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.clip_dir, f"distract_{ts}.mp4")
        h, w = frame.shape[:2]
        try:
            self._writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        except Exception as e:
            print("录像初始化失败:", e)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.running = False
        self.wait()

# ==========================================
# 教师编辑对话框
# ==========================================
class TeacherDialog(QDialog):
    def __init__(self, record, parent=None):
        super().__init__(parent)
        self.setWindowTitle("教师审核与评定")
        self.layout = QVBoxLayout(self)
        form = QFormLayout()

        self.pages_input = QSpinBox()
        self.pages_input.setMaximum(1000)
        self.pages_input.setValue(record.get("actual_pages", 0))
        
        self.status_combo = QComboBox()
        self.status_combo.addItems(["未完成", "已完成"])
        self.status_combo.setCurrentText(record.get("status", "未完成"))

        form.addRow("实际完成页数:", self.pages_input)
        form.addRow("完成状态评定:", self.status_combo)
        self.layout.addLayout(form)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        self.layout.addWidget(btn_box)

    def get_data(self):
        return self.pages_input.value(), self.status_combo.currentText()

# ==========================================
# 主界面 UI
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("学生智能学习监控系统")
        self.resize(1100, 700)
        self.worker = None
        self.current_task = {}
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        self.tab_study = QWidget()
        self.tab_history = QWidget()
        self.tabs.addTab(self.tab_study, "学习监控")
        self.tabs.addTab(self.tab_history, "历史数据与报告")
        layout.addWidget(self.tabs)

        self._build_study_tab()
        self._build_history_tab()

    def _build_study_tab(self):
        layout = QVBoxLayout(self.tab_study)
        
        # 任务设置
        setup_group = QGroupBox("学习任务设定")
        form = QGridLayout()
        
        form.addWidget(QLabel("书籍序列号:"), 0, 0)
        self.sn_input = QLineEdit()
        self.sn_input.returnPressed.connect(self._load_book)
        form.addWidget(self.sn_input, 0, 1)
        btn_load = QPushButton("录入书籍")
        btn_load.clicked.connect(self._load_book)
        form.addWidget(btn_load, 0, 2)
        
        form.addWidget(QLabel("当前书名:"), 1, 0)
        self.book_name_label = QLabel("未录入")
        self.book_name_label.setStyleSheet("font-weight: bold; color: blue; font-size: 14px;")
        form.addWidget(self.book_name_label, 1, 1, 1, 2)
        
        form.addWidget(QLabel("任务类型:"), 2, 0)
        self.task_type = QComboBox()
        self.task_type.addItems(["预习", "学习", "复习"])
        form.addWidget(self.task_type, 2, 1)
        
        form.addWidget(QLabel("规定时间(分钟):"), 3, 0)
        self.time_spin = QSpinBox()
        self.time_spin.setRange(1, 180)
        self.time_spin.setValue(30)
        form.addWidget(self.time_spin, 3, 1)
        
        form.addWidget(QLabel("规定页数(页):"), 4, 0)
        self.pages_spin = QSpinBox()
        self.pages_spin.setRange(1, 200)
        self.pages_spin.setValue(10)
        form.addWidget(self.pages_spin, 4, 1)
        
        setup_group.setLayout(form)
        layout.addWidget(setup_group)

        # 监控区
        monitor_group = QGroupBox("实时监控与专注度")
        m_layout = QHBoxLayout(monitor_group)
        
        self.video_label = QLabel("摄像头未开启")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background: #000; color: #fff;")
        self.video_label.setAlignment(Qt.AlignCenter)
        m_layout.addWidget(self.video_label)

        status_layout = QVBoxLayout()
        self.status_label = QLabel("状态: 等待开始")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.focus_label = QLabel("综合专注度: 0.0%")
        self.focus_label.setStyleSheet("font-size: 18px; font-weight: bold; color: green;")
        self.time_label = QLabel("剩余时间: --:--")
        self.time_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #d32f2f;")
        
        self.btn_start = QPushButton("▶ 开始学习")
        self.btn_start.setStyleSheet("background: #4CAF50; color: white; padding: 10px; font-size: 16px;")
        self.btn_start.clicked.connect(self._start_study)
        
        self.btn_pause = QPushButton("⏸ 暂停")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._toggle_pause)
        
        self.btn_stop = QPushButton("⏹ 结束并保存")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background: #f44336; color: white; padding: 10px; font-size: 16px;")
        self.btn_stop.clicked.connect(self._stop_study)

        status_layout.addWidget(self.time_label)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.focus_label)
        status_layout.addStretch()
        status_layout.addWidget(self.btn_start)
        status_layout.addWidget(self.btn_pause)
        status_layout.addWidget(self.btn_stop)
        
        m_layout.addLayout(status_layout)
        layout.addWidget(monitor_group)

    def _build_history_tab(self):
        layout = QVBoxLayout(self.tab_history)
        
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["日期", "书名", "任务", "规定页数", "实际页数", "专注度", "状态"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        btn_refresh = QPushButton("刷新数据")
        btn_refresh.clicked.connect(self._load_history)
        btn_layout.addWidget(btn_refresh)

        btn_edit = QPushButton("教师审核编辑")
        btn_edit.setStyleSheet("background: #2196F3; color: white;")
        btn_edit.clicked.connect(self._teacher_edit)
        btn_layout.addWidget(btn_edit)

        btn_pdf = QPushButton("导出 PDF 报告")
        btn_pdf.clicked.connect(self._export_pdf)
        btn_layout.addWidget(btn_pdf)

        layout.addLayout(btn_layout)
        self._load_history()

    # ==========================================
    # 逻辑功能实现
    # ==========================================
    def _load_book(self):
        sn = self.sn_input.text().strip()
        if sn in BOOK_DATABASE:
            self.book_name_label.setText(BOOK_DATABASE[sn])
        else:
            self.book_name_label.setText("❌ 未知序列号")
            QMessageBox.warning(self, "错误", "未找到该序列号对应的书籍！")

    def _start_study(self):
        if self.book_name_label.text() in ["未录入", "❌ 未知序列号"]:
            QMessageBox.warning(self, "提示", "请先录入有效的书籍序列号！")
            return

        self.current_task = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "book": self.book_name_label.text(),
            "type": self.task_type.currentText(),
            "req_pages": self.pages_spin.value(),
            "actual_pages": 0,
            "focus_score": 0.0,
            "status": "未完成"
        }

        total_sec = self.time_spin.value() * 60
        self.worker = MonitorThread(total_sec)
        self.worker.frame_ready.connect(self._update_video)
        self.worker.status_update.connect(self._update_status)
        self.worker.alert_triggered.connect(lambda msg: self.statusBar().showMessage(msg, 5000))
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)

    def _toggle_pause(self):
        if not self.worker: return
        if self.worker.is_paused:
            self.worker.resume()
            self.btn_pause.setText("⏸ 暂停")
        else:
            self.worker.pause()
            self.btn_pause.setText("▶ 继续")

    def _stop_study(self):
        if self.worker:
            self.worker.stop()
            self.worker = None

        if self.current_task:
            dm.add_record(self.current_task)
            QMessageBox.information(self, "保存成功", "本次学习数据已保存，请在历史记录中进行教师审核。")
            self.current_task = {}

        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸ 暂停")
        self.btn_stop.setEnabled(False)
        self.video_label.clear()
        self._load_history()

    def _on_worker_finished(self):
        if self.btn_stop.isEnabled():
            self._stop_study() # 时间到自动触发保存

    def _update_video(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(self.video_label.size(), Qt.KeepAspectRatio))

    def _update_status(self, text, score, remain_sec):
        self.status_label.setText(f"状态: {text}")
        self.focus_label.setText(f"综合专注度: {score:.1f}%")
        self.focus_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {'green' if score >= 60 else 'red'};")
        
        m, s = divmod(remain_sec, 60)
        self.time_label.setText(f"剩余时间: {m:02d}:{s:02d}")
        
        if self.current_task:
            self.current_task["focus_score"] = score

    def _load_history(self):
        self.table.setRowCount(0)
        for record in dm.data["records"]:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(record["date"]))
            self.table.setItem(row, 1, QTableWidgetItem(record["book"]))
            self.table.setItem(row, 2, QTableWidgetItem(record["type"]))
            self.table.setItem(row, 3, QTableWidgetItem(str(record["req_pages"])))
            self.table.setItem(row, 4, QTableWidgetItem(str(record["actual_pages"])))
            self.table.setItem(row, 5, QTableWidgetItem(f"{record['focus_score']:.1f}%"))
            
            status_item = QTableWidgetItem(record["status"])
            if record["status"] == "已完成":
                status_item.setForeground(Qt.green)
            else:
                status_item.setForeground(Qt.red)
            self.table.setItem(row, 6, status_item)

    def _teacher_edit(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "提示", "请先选中一条记录！")
            return
        
        record = dm.data["records"][row]
        dialog = TeacherDialog(record, self)
        if dialog.exec_() == QDialog.Accepted:
            new_pages, new_status = dialog.get_data()
            record["actual_pages"] = new_pages
            record["status"] = new_status
            dm.update_record(row, record)
            self._load_history()
            QMessageBox.information(self, "成功", "修改已保存！")

    def _export_pdf(self):
        row = self.table.currentRow()
        if row < 0: return
        record = dm.data["records"][row]
        
        filename = f"Report_{record['book']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # 尝试加载中文字体
        font_path = "C:\\Windows\\Fonts\\simhei.ttf"
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont("SimHei", font_path))
            title_style = ParagraphStyle("T", parent=styles["Heading1"], fontName="SimHei")
            body_style = ParagraphStyle("B", parent=styles["Normal"], fontName="SimHei")
        else:
            title_style = styles["Heading1"]
            body_style = styles["Normal"]

        diff_pages = record['req_pages'] - record['actual_pages']
        diff_text = f"还差 {diff_pages} 页未完成" if diff_pages > 0 else "已超额完成"

        story = [
            Paragraph("智能学习监控系统 - 学习报告", title_style),
            Spacer(1, 15),
            Paragraph(f"日期: {record['date']}", body_style),
            Paragraph(f"书籍: {record['book']}", body_style),
            Paragraph(f"任务: {record['type']}", body_style),
            Spacer(1, 10),
            Paragraph(f"<b>学习进度:</b>", body_style),
            Paragraph(f"规定页数: {record['req_pages']} 页", body_style),
            Paragraph(f"实际完成: {record['actual_pages']} 页 ({diff_text})", body_style),
            Paragraph(f"评定状态: {record['status']}", body_style),
            Spacer(1, 10),
            Paragraph(f"<b>专注度分析:</b>", body_style),
            Paragraph(f"综合专注度: {record['focus_score']:.1f}%", body_style),
        ]
        
        doc.build(story)
        QMessageBox.information(self, "成功", f"PDF已生成: {os.path.abspath(filename)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())