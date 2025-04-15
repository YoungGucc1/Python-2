import sys
import os
import cv2
import numpy as np
import albumentations as A
import qdarkstyle # Импорт для темной темы
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QSpinBox, QMessageBox, QStatusBar,
    QMainWindow
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon # Можно добавить иконку

# --- Поток для выполнения аугментации в фоне ---
class AugmentationWorker(QThread):
    progress_updated = pyqtSignal(int)  # Сигнал для обновления прогресса (процент)
    status_updated = pyqtSignal(str)  # Сигнал для обновления статуса
    finished = pyqtSignal(bool, str)   # Сигнал завершения (успех?, сообщение)

    def __init__(self, input_path, output_dir, num_variations, augmentations):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.num_variations = num_variations
        self.augmentations = augmentations
        self._is_running = True

    def run(self):
        try:
            self.status_updated.emit(f"Чтение исходного изображения: {os.path.basename(self.input_path)}")
            image = cv2.imread(self.input_path)
            if image is None:
                raise ValueError(f"Не удалось прочитать изображение: {self.input_path}")

            # Конвертируем в RGB, так как albumentations часто ожидает RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            base_filename = os.path.splitext(os.path.basename(self.input_path))[0]
            extension = os.path.splitext(self.input_path)[1] # Сохраняем исходное расширение

            # Создаем папку вывода, если ее нет
            os.makedirs(self.output_dir, exist_ok=True)
            self.status_updated.emit(f"Сохранение в: {self.output_dir}")

            for i in range(self.num_variations):
                if not self._is_running:
                    self.status_updated.emit("Процесс прерван.")
                    self.finished.emit(False, "Аугментация прервана пользователем.")
                    return

                # Применяем аугментации
                augmented = self.augmentations(image=image)
                augmented_image = augmented['image']

                # Конвертируем обратно в BGR для сохранения через OpenCV
                augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

                # Генерируем имя файла
                output_filename = f"{base_filename}_aug_{i+1:04d}{extension}"
                output_path = os.path.join(self.output_dir, output_filename)

                # Сохраняем изображение
                success = cv2.imwrite(output_path, augmented_image_bgr)
                if not success:
                     self.status_updated.emit(f"Ошибка сохранения: {output_filename}")
                     # Можно продолжить или прервать - пока продолжаем
                # else:
                #     self.status_updated.emit(f"Сохранено: {output_filename}") # Слишком много сообщений

                # Обновляем прогресс
                progress = int(((i + 1) / self.num_variations) * 100)
                self.progress_updated.emit(progress)
                # Небольшая задержка, чтобы GUI успевал обновляться (опционально)
                # self.msleep(10)

            if self._is_running:
                self.status_updated.emit(f"Завершено. Создано {self.num_variations} вариаций.")
                self.finished.emit(True, f"Успешно создано {self.num_variations} вариаций.")

        except Exception as e:
            error_message = f"Ошибка в процессе аугментации: {e}"
            self.status_updated.emit(error_message)
            self.finished.emit(False, error_message)

    def stop(self):
        self._is_running = False
        self.status_updated.emit("Остановка процесса...")

# --- Основное окно приложения ---
class AugmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генератор Аугментаций для YOLO")
        self.setGeometry(100, 100, 600, 250) # x, y, width, height

        # --- Переменные состояния ---
        self.input_image_path = ""
        self.output_directory = ""
        self.worker = None # Для фонового потока

        # --- Определение пайплайна аугментаций ---
        # Настройте этот пайплайн под ваши нужды!
        # Это примеры, выберите то, что подходит для ваших товаров
        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            # A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2), # Иногда может быть полезен
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.6, border_mode=cv2.BORDER_CONSTANT, value=0), # Черные границы
            # A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.3), # Может сильно искажать, аккуратно
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.2), # Имитация перекрытий
             A.OneOf([ # Применить один из эффектов размытия
                 A.MotionBlur(blur_limit=7, p=1.0),
                 A.MedianBlur(blur_limit=5, p=1.0),
                 A.Blur(blur_limit=5, p=1.0),
             ], p=0.3),
             A.OneOf([ # Применить один из эффектов качества
                 A.ImageCompression(quality_lower=75, quality_upper=95, p=1.0),
                 # A.Downscale(scale_min=0.75, scale_max=0.95, p=1.0), # Может уменьшать размер
             ], p=0.2)
        ])

        # --- Создание виджетов ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Выбор входного файла ---
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Входное фото:")
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Выберите файл изображения...")
        self.input_path_edit.setReadOnly(True)
        self.input_button = QPushButton("Обзор...")
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_path_edit)
        input_layout.addWidget(self.input_button)
        main_layout.addLayout(input_layout)

        # --- Выбор папки назначения ---
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Папка вывода:")
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Выберите папку для сохранения...")
        self.output_path_edit.setReadOnly(True)
        self.output_button = QPushButton("Обзор...")
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(self.output_button)
        main_layout.addLayout(output_layout)

        # --- Количество вариаций ---
        variations_layout = QHBoxLayout()
        self.variations_label = QLabel("Количество вариаций:")
        self.variations_spinbox = QSpinBox()
        self.variations_spinbox.setRange(1, 10000) # От 1 до 10000 вариаций
        self.variations_spinbox.setValue(50)       # Значение по умолчанию
        self.variations_spinbox.setFixedWidth(100)
        variations_layout.addWidget(self.variations_label)
        variations_layout.addWidget(self.variations_spinbox)
        variations_layout.addStretch() # Заполняет пустое пространство справа
        main_layout.addLayout(variations_layout)

        # --- Кнопка Старт ---
        self.start_button = QPushButton("🚀 Начать Аугментацию")
        self.start_button.setFixedHeight(40) # Сделать кнопку повыше
        main_layout.addWidget(self.start_button)

        # --- Статус бар ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов.")

        # --- Подключение сигналов к слотам ---
        self.input_button.clicked.connect(self.select_input_image)
        self.output_button.clicked.connect(self.select_output_dir)
        self.start_button.clicked.connect(self.start_augmentation_process)

        # --- Установка иконки (опционально) ---
        # self.setWindowIcon(QIcon("path/to/your/icon.png"))

    # --- Слоты (обработчики событий) ---
    def select_input_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите исходное изображение",
            "", # Начальная директория
            "Изображения (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if file_path:
            self.input_image_path = file_path
            self.input_path_edit.setText(file_path)
            self.status_bar.showMessage(f"Выбрано изображение: {os.path.basename(file_path)}")

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку для сохранения результатов",
            "" # Начальная директория
        )
        if dir_path:
            self.output_directory = dir_path
            self.output_path_edit.setText(dir_path)
            self.status_bar.showMessage(f"Выбрана папка: {dir_path}")

    def start_augmentation_process(self):
        # Проверка входных данных
        if not self.input_image_path:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, выберите входное изображение.")
            return
        if not os.path.exists(self.input_image_path):
             QMessageBox.warning(self, "Ошибка", f"Файл не найден: {self.input_image_path}")
             return
        if not self.output_directory:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, выберите папку для сохранения результатов.")
            return

        num_variations = self.variations_spinbox.value()

        # Отключаем кнопку старта во время работы
        self.start_button.setEnabled(False)
        self.start_button.setText("В процессе...")

        # Создаем и запускаем рабочий поток
        self.worker = AugmentationWorker(
            self.input_image_path,
            self.output_directory,
            num_variations,
            self.augmentation_pipeline
        )
        # Подключаем сигналы потока к слотам GUI
        self.worker.status_updated.connect(self.update_status)
        self.worker.progress_updated.connect(self.update_progress) # Пока просто выводим в статус
        self.worker.finished.connect(self.on_augmentation_finished)
        self.worker.start()

    def update_status(self, message):
        self.status_bar.showMessage(message)

    def update_progress(self, percentage):
         self.status_bar.showMessage(f"Выполнение: {percentage}%")
         # В будущем можно добавить QProgressBar

    def on_augmentation_finished(self, success, message):
        # Включаем кнопку обратно
        self.start_button.setEnabled(True)
        self.start_button.setText("🚀 Начать Аугментацию")
        self.worker = None # Очищаем ссылку на поток

        if success:
            QMessageBox.information(self, "Завершено", message)
        else:
            QMessageBox.critical(self, "Ошибка", message)
        self.status_bar.showMessage("Готов." if success else f"Ошибка: {message}")

    # Переопределяем метод закрытия окна, чтобы остановить поток, если он работает
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, 'Подтверждение выхода',
                                         "Процесс аугментации еще выполняется. Вы уверены, что хотите выйти?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop() # Посылаем сигнал остановки потоку
                self.worker.wait(500) # Даем потоку немного времени на завершение
                event.accept() # Закрываем окно
            else:
                event.ignore() # Отменяем закрытие
        else:
            event.accept() # Закрываем окно


# --- Запуск приложения ---
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Применяем темный стиль
    # Убедитесь, что qdarkstyle установлен: pip install qdarkstyle pyqt6
    try:
        # Указываем API явно, т.к. qdarkstyle поддерживает несколько
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))
    except ImportError:
        print("Предупреждение: qdarkstyle не найден. Будет использована стандартная тема.")
        # Можно добавить базовый темный стиль вручную, если qdarkstyle нет
        # app.setStyleSheet("QWidget { background-color: #333; color: #EEE; } ...")
    except Exception as e:
         print(f"Не удалось применить темный стиль: {e}")


    window = AugmentationApp()
    window.show()
    sys.exit(app.exec())