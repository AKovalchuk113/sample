import sys
import cv2
import time
import threading
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, cairo

# ------------------------------------------------------------
# Конвертация OpenCV (BGR) в GdkPixbuf (RGB)
# ------------------------------------------------------------
def cv2_to_pixbuf(frame):
    if frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    # Используем new_from_bytes для безопасного копирования данных
    data = GLib.Bytes.new(rgb.tobytes())
    pixbuf = Gdk.pixbuf_new_from_bytes(
        data,
        Gdk.Colorspace.RGB,
        False,
        8,
        w, h,
        w * ch
    )
    return pixbuf

# ------------------------------------------------------------
# Поток для чтения видео
# ------------------------------------------------------------
class VideoThread(threading.Thread):
    def __init__(self, cell_index, video_path, callback_new_frame, callback_finished):
        super().__init__()
        self.cell_index = cell_index
        self.video_path = video_path
        self.callback_new_frame = callback_new_frame   # будет вызывать GLib.idle_add
        self.callback_finished = callback_finished
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        if not self.video_path:
            GLib.idle_add(self.callback_finished, self.cell_index)
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            GLib.idle_add(self.callback_finished, self.cell_index)
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

            pixbuf = cv2_to_pixbuf(frame)
            if pixbuf is not None:
                GLib.idle_add(self.callback_new_frame, self.cell_index, pixbuf, current_fps)

            time.sleep(frame_time)

        cap.release()
        GLib.idle_add(self.callback_finished, self.cell_index)

# ------------------------------------------------------------
# Виджет отображения видео (DrawingArea)
# ------------------------------------------------------------
class VideoWidget(Gtk.DrawingArea):
    def __init__(self, video_path=None):
        super().__init__()
        self.video_path = video_path
        self.pixbuf = None
        self.info_text = "Нет видео"
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_size_request(180, 101)   # размер для маленьких окон
        self.connect("draw", self.on_draw)

    def on_draw(self, widget, cr):
        cr.set_source_rgb(0, 0, 0)
        cr.paint()
        if self.pixbuf is not None:
            w = self.get_allocated_width()
            h = self.get_allocated_height()
            pix_w = self.pixbuf.get_width()
            pix_h = self.pixbuf.get_height()
            scale = min(w / pix_w, h / pix_h)
            draw_w = int(pix_w * scale)
            draw_h = int(pix_h * scale)
            x = (w - draw_w) // 2
            y = (h - draw_h) // 2
            Gdk.cairo_set_source_pixbuf(cr, self.pixbuf, x, y)
            cr.rectangle(x, y, draw_w, draw_h)
            cr.fill()
        cr.set_source_rgb(1, 1, 1)
        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(12)
        cr.move_to(5, self.get_allocated_height() - 5)
        cr.show_text(self.info_text)

    def set_frame(self, pixbuf, fps=None):
        self.pixbuf = pixbuf
        if fps is not None:
            name = self.video_path.split('/')[-1] if self.video_path else "None"
            self.info_text = f"{name}\n{round(fps, 1)} fps"
        self.queue_draw()

    def set_finished(self):
        if not self.info_text.endswith(" [конец]"):
            self.info_text += " [конец]"
        self.queue_draw()

# ------------------------------------------------------------
# Главное окно
# ------------------------------------------------------------
class MainWindow(Gtk.Window):
    def __init__(self, video_paths):
        super().__init__(title="Мультиплеер видео (GTK)")
        self.video_paths = video_paths[:9]   # максимум 9 видео
        self.num_cells = 9
        self.threads = [None] * self.num_cells
        self.cells = [None] * self.num_cells   # список VideoWidget или EventBox

        self.set_default_size(1280, 720)
        self.set_position(Gtk.WindowPosition.CENTER)

        # Корневой вертикальный контейнер
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add(vbox)

        # Основная сетка: три колонки
        grid = Gtk.Grid()
        grid.set_row_spacing(10)
        grid.set_column_spacing(10)
        vbox.pack_start(grid, True, True, 0)

        # Левая колонка (4 виджета)
        left_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        for i in range(4):
            vid = VideoWidget()
            self.cells[i] = vid
            left_vbox.pack_start(vid, True, True, 0)
        grid.attach(left_vbox, 0, 0, 1, 1)

        # Центральный виджет (большой)
        self.center_cell = VideoWidget()
        self.center_cell.set_size_request(600, 338)  # 16:9
        self.cells[4] = self.center_cell
        grid.attach(self.center_cell, 1, 0, 1, 1)

        # Правая колонка (4 виджета)
        right_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        for i in range(5, 9):
            vid = VideoWidget()
            self.cells[i] = vid
            right_vbox.pack_start(vid, True, True, 0)
        grid.attach(right_vbox, 2, 0, 1, 1)

        # Панель кнопок
        btn_box = Gtk.Box(spacing=10)
        self.start_btn = Gtk.Button(label="Запустить все")
        self.start_btn.connect("clicked", self.on_start_clicked)
        self.stop_btn = Gtk.Button(label="Остановить все")
        self.stop_btn.connect("clicked", self.on_stop_clicked)
        self.stop_btn.set_sensitive(False)
        btn_box.pack_start(self.start_btn, False, False, 0)
        btn_box.pack_start(self.stop_btn, False, False, 0)
        vbox.pack_start(btn_box, False, False, 0)

        # Статусная строка
        self.statusbar = Gtk.Statusbar()
        vbox.pack_start(self.statusbar, False, False, 0)

        # Назначаем видео виджетам
        self.assign_videos()

        # При клике по маленькому виджету меняем его с центральным
        for idx, cell in enumerate(self.cells):
            if idx != 4 and cell.video_path is not None:
                evbox = Gtk.EventBox()
                evbox.add(cell)
                evbox.connect("button-press-event", self.on_small_click, idx)
                parent = cell.get_parent()
                if parent:
                    parent.remove(cell)
                    evbox.show_all()
                    parent.pack_start(evbox, True, True, 0)
                self.cells[idx] = evbox

    def assign_videos(self):
        """Назначает видео из списка первым N ячейкам (N = len(video_paths))"""
        for i in range(self.num_cells):
            # Получаем VideoWidget (может быть внутри EventBox)
            if isinstance(self.cells[i], Gtk.EventBox):
                widget = self.cells[i].get_child()
            else:
                widget = self.cells[i]

            if i < len(self.video_paths):
                path = self.video_paths[i]
                widget.video_path = path
                widget.info_text = path.split('/')[-1] + "\nОстановлено"
            else:
                widget.video_path = None
                widget.info_text = "Нет видео"
            widget.queue_draw()

    def on_small_click(self, eventbox, event, idx):
        """Обмен видео между маленьким виджетом и центральным"""
        small_widget = eventbox.get_child()
        center_widget = self.center_cell

        # Останавливаем потоки, если они запущены
        if self.threads[4] and self.threads[4].is_alive():
            self.threads[4].stop()
            self.threads[4].join()
            self.threads[4] = None
        if self.threads[idx] and self.threads[idx].is_alive():
            self.threads[idx].stop()
            self.threads[idx].join()
            self.threads[idx] = None

        # Меняем пути
        small_widget.video_path, center_widget.video_path = center_widget.video_path, small_widget.video_path

        # Обновляем тексты
        small_widget.info_text = (small_widget.video_path.split('/')[-1] if small_widget.video_path else "Нет видео") + "\nОстановлено"
        center_widget.info_text = (center_widget.video_path.split('/')[-1] if center_widget.video_path else "Нет видео") + "\nОстановлено"
        small_widget.queue_draw()
        center_widget.queue_draw()

        # Если плееры были запущены – перезапускаем потоки
        if not self.start_btn.get_sensitive():  # кнопка "Запустить" неактивна → всё запущено
            self.start_cell_thread(idx)
            self.start_cell_thread(4)

    def start_cell_thread(self, cell_index):
        """Запуск потока для конкретной ячейки"""
        cell = self.cells[cell_index]
        if isinstance(cell, Gtk.EventBox):
            widget = cell.get_child()
        else:
            widget = cell

        if widget.video_path is None:
            return

        # Останавливаем старый поток
        if self.threads[cell_index] and self.threads[cell_index].is_alive():
            self.threads[cell_index].stop()
            self.threads[cell_index].join()

        thread = VideoThread(cell_index, widget.video_path,
                             self.on_new_frame, self.on_video_finished)
        self.threads[cell_index] = thread
        thread.start()

    def on_new_frame(self, cell_index, pixbuf, fps):
        """Вызывается из потока через GLib.idle_add"""
        cell = self.cells[cell_index]
        if isinstance(cell, Gtk.EventBox):
            widget = cell.get_child()
        else:
            widget = cell
        widget.set_frame(pixbuf, fps)

    def on_video_finished(self, cell_index):
        """Видео закончилось"""
        cell = self.cells[cell_index]
        if isinstance(cell, Gtk.EventBox):
            widget = cell.get_child()
        else:
            widget = cell
        widget.set_finished()
        self.threads[cell_index] = None

    def on_start_clicked(self, button):
        """Запустить все потоки"""
        self.start_btn.set_sensitive(False)
        self.stop_btn.set_sensitive(True)
        for i in range(self.num_cells):
            self.start_cell_thread(i)

    def on_stop_clicked(self, button):
        """Остановить все потоки"""
        for i, th in enumerate(self.threads):
            if th and th.is_alive():
                th.stop()
                th.join()
                self.threads[i] = None
        self.start_btn.set_sensitive(True)
        self.stop_btn.set_sensitive(False)

    def on_destroy(self, widget):
        Gtk.main_quit()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Использование: python gtk_video_player.py video1.mp4 video2.mp4 ... (до 9 файлов)")
        sys.exit(1)

    video_files = sys.argv[1:10]  # берём максимум 9
    app = MainWindow(video_files)
    app.connect("destroy", app.on_destroy)
    app.show_all()
    Gtk.main()
