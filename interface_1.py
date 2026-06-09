#!/usr/bin/env python3
# qt6_video_grid.py
# Запуск: python qt6_video_grid.py video1.mp4 video2.mp4 ... (до 9 файлов)

import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QSize
from PySide6.QtGui import QImage, QOpenGLContext, QSurfaceFormat
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout,
                               QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                               QFrame, QSizePolicy)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import QOpenGLTexture, QOpenGLFunctions

# ------------------------------------------------------------
# Поток для декодирования видео (работает в отдельном потоке)
# ------------------------------------------------------------
class VideoDecoder(QObject):
    frame_ready = Signal(int, QImage, float)  # cell_index, qimage, current_fps
    finished = Signal(int)

    def __init__(self, cell_index, video_path):
        super().__init__()
        self.cell_index = cell_index
        self.video_path = video_path
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        if not self.video_path:
            self.finished.emit(self.cell_index)
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit(self.cell_index)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1.0 / fps if fps > 0 else 0.033
        prev_time = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            current_fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now

            # Конвертация OpenCV BGR -> RGB -> QImage
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # Копируем данные, т.к. оригинальный буфер будет перезаписан
            qimage = qimage.copy()

            self.frame_ready.emit(self.cell_index, qimage, current_fps)

            # Сон для поддержания реальной частоты кадров
            time.sleep(frame_time)

        cap.release()
        self.finished.emit(self.cell_index)


# ------------------------------------------------------------
# OpenGL виджет для отображения одного видео
# ------------------------------------------------------------
class VideoGLWidget(QOpenGLWidget):
    def __init__(self, cell_index, parent=None):
        super().__init__(parent)
        self.cell_index = cell_index
        self.video_path = None
        self.current_image = None
        self.texture = None
        self.info_text = "Нет видео"
        self.setMinimumSize(160, 90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Для отображения FPS
        self.fps = 0.0

    def set_video_path(self, path):
        self.video_path = path
        if path:
            name = Path(path).name
            self.info_text = f"{name}\n{self.fps:.1f} fps" if self.fps else f"{name}\n0 fps"
        else:
            self.info_text = "Нет видео"
        self.update()

    def update_frame(self, qimage, fps):
        self.current_image = qimage
        self.fps = fps
        if self.video_path:
            name = Path(self.video_path).name
            self.info_text = f"{name}\n{fps:.1f} fps"
        else:
            self.info_text = f"{fps:.1f} fps"
        self.update()  # вызывает paintGL

    def set_finished(self):
        self.info_text += " [конец]"
        self.update()

    def initializeGL(self):
        # Инициализация OpenGL
        gl = self.context().functions()
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glEnable(gl.GL_TEXTURE_2D)

    def paintGL(self):
        gl = self.context().functions()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        if self.current_image is not None:
            # Создаём или обновляем текстуру
            if self.texture is None:
                self.texture = QOpenGLTexture(QOpenGLTexture.Target2D)
                self.texture.setMinificationFilter(QOpenGLTexture.Linear)
                self.texture.setMagnificationFilter(QOpenGLTexture.Linear)

            # Преобразуем QImage в формат RGBA для текстуры
            img = self.current_image.convertToFormat(QImage.Format_RGBA8888)
            self.texture.setData(img, QOpenGLTexture.RGBA, QOpenGLTexture.Uint8)

            # Рисуем текстуру на весь виджет
            w = self.width()
            h = self.height()
            tex_w = self.texture.width()
            tex_h = self.texture.height()

            # Сохраняем пропорции
            scale = min(w / tex_w, h / tex_h)
            draw_w = int(tex_w * scale)
            draw_h = int(tex_h * scale)
            x = (w - draw_w) // 2
            y = (h - draw_h) // 2

            gl.glViewport(x, y, draw_w, draw_h)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glOrtho(0, draw_w, draw_h, 0, -1, 1)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()

            self.texture.bind()
            gl.glBegin(gl.GL_QUADS)
            gl.glTexCoord2f(0, 0); gl.glVertex2f(0, 0)
            gl.glTexCoord2f(1, 0); gl.glVertex2f(draw_w, 0)
            gl.glTexCoord2f(1, 1); gl.glVertex2f(draw_w, draw_h)
            gl.glTexCoord2f(0, 1); gl.glVertex2f(0, draw_h)
            gl.glEnd()
            self.texture.release()

            # Восстанавливаем область для текста
            gl.glViewport(0, 0, w, h)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glOrtho(0, w, h, 0, -1, 1)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()

        # Рисуем текст (простой вариант, без шрифтов OpenGL, можно через QPainter)
        # Для простоты используем QPainter поверх OpenGL
        painter = self.painter_for_text()
        if painter:
            painter.setPen(Qt.white)
            painter.setFont(self.font())
            painter.drawText(5, self.height() - 5, self.info_text)
            painter.end()

    def painter_for_text(self):
        """Возвращает QPainter для рисования текста поверх OpenGL"""
        try:
            from PySide6.QtGui import QPainter
            painter = QPainter(self)
            painter.begin(self)
            return painter
        except:
            return None

    def resizeGL(self, w, h):
        gl = self.context().functions()
        gl.glViewport(0, 0, w, h)


# ------------------------------------------------------------
# Главное окно с сеткой
# ------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, video_paths):
        super().__init__()
        self.video_paths = video_paths[:9]  # максимум 9
        self.num_cells = 9
        self.threads = [None] * self.num_cells
        self.decoders = [None] * self.num_cells
        self.cells = [None] * self.num_cells  # список VideoGLWidget

        self.setWindowTitle("Видеосетка Qt6 + OpenGL")
        self.setMinimumSize(800, 600)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Сетка 3x3
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)

        # Создаём 9 виджетов
        for i in range(self.num_cells):
            widget = VideoGLWidget(i)
            self.cells[i] = widget
            row = i // 3
            col = i % 3
            grid_layout.addWidget(widget, row, col, 1, 1)

        # Делаем центральный (индекс 4) больше
        # В QGridLayout можно задать растяжение, но проще установить размер
        self.cells[4].setMinimumSize(320, 180)
        # Устанавливаем одинаковые пропорции для всех
        for i in range(self.num_cells):
            grid_layout.setRowStretch(i // 3, 1)
            grid_layout.setColumnStretch(i % 3, 1)

        main_layout.addLayout(grid_layout)

        # Панель кнопок
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Запустить все")
        self.stop_btn = QPushButton("Остановить все")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.on_start_all)
        self.stop_btn.clicked.connect(self.on_stop_all)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        main_layout.addLayout(btn_layout)

        # Статусбар
        self.status_label = QLabel("Готов")
        main_layout.addWidget(self.status_label)

        # Назначаем видео
        self.assign_videos()

        # Подключаем обработчики кликов (через eventFilter или через собственный сигнал)
        for i, w in enumerate(self.cells):
            if i != 4:
                w.mousePressEvent = lambda event, idx=i: self.on_cell_clicked(idx)

    def assign_videos(self):
        """Назначает файлы виджетам"""
        for i in range(self.num_cells):
            if i < len(self.video_paths):
                self.cells[i].set_video_path(self.video_paths[i])
            else:
                self.cells[i].set_video_path(None)

    def on_cell_clicked(self, idx):
        """Обмен видео между ячейкой idx и центральной (4)"""
        if idx == 4:
            return
        # Останавливаем оба потока
        self.stop_cell(4)
        self.stop_cell(idx)

        # Меняем пути
        center_path = self.cells[4].video_path
        small_path = self.cells[idx].video_path
        self.cells[4].set_video_path(small_path)
        self.cells[idx].set_video_path(center_path)

        # Если воспроизведение было запущено (кнопка Stop активна) – перезапускаем
        if self.stop_btn.isEnabled():
            self.start_cell(4)
            self.start_cell(idx)

    def start_cell(self, cell_index):
        """Запускает декодирование для одной ячейки"""
        widget = self.cells[cell_index]
        if widget.video_path is None:
            return

        # Останавливаем старый поток
        self.stop_cell(cell_index)

        # Создаём объект в отдельном потоке
        decoder = VideoDecoder(cell_index, widget.video_path)
        self.decoders[cell_index] = decoder

        thread = threading.Thread(target=decoder.run, daemon=True)
        self.threads[cell_index] = thread

        # Подключаем сигналы
        decoder.frame_ready.connect(self.on_frame_ready)
        decoder.finished.connect(self.on_video_finished)

        thread.start()

    def stop_cell(self, cell_index):
        """Останавливает поток для ячейки"""
        if self.decoders[cell_index] is not None:
            self.decoders[cell_index].stop()
        if self.threads[cell_index] is not None:
            self.threads[cell_index].join(timeout=0.5)
            self.threads[cell_index] = None
        self.decoders[cell_index] = None

    def on_frame_ready(self, cell_index, qimage, fps):
        """Сигнал из потока: новый кадр"""
        widget = self.cells[cell_index]
        if widget:
            widget.update_frame(qimage, fps)

    def on_video_finished(self, cell_index):
        """Видео закончилось"""
        widget = self.cells[cell_index]
        if widget:
            widget.set_finished()
        self.threads[cell_index] = None
        self.decoders[cell_index] = None

    def on_start_all(self):
        for i in range(self.num_cells):
            self.start_cell(i)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Воспроизведение...")

    def on_stop_all(self):
        for i in range(self.num_cells):
            self.stop_cell(i)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Остановлено")

    def closeEvent(self, event):
        self.on_stop_all()
        event.accept()


# ------------------------------------------------------------
# Запуск приложения
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python qt6_video_grid.py video1.mp4 video2.mp4 ... (до 9 файлов)")
        sys.exit(1)

    app = QApplication(sys.argv)

    # Запрашиваем OpenGL 2.1 (достаточно для текстур)
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    video_files = sys.argv[1:10]
    window = MainWindow(video_files)
    window.show()
    sys.exit(app.exec())
