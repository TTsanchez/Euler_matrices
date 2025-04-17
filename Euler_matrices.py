import numpy as np
import galois
from sympy import isprime, primitive_root
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QPushButton, QHBoxLayout,
                             QTextEdit, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QImage
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class EulerMatrixApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Визуализатор матриц Эйлера")
        self.setFixedSize(800, 900)

        # Главный виджет
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        # Основной layout
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        # Стилизация
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                font-size: 14px;
            }
            QLineEdit, QTextEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        # Создаем элементы интерфейса
        self.create_input_panel()
        self.create_matrix_display()
        self.create_info_panel()

        # Начальное сообщение
        self.show_initial_message()

    def create_input_panel(self):
        """Панель ввода параметров"""
        input_panel = QWidget()
        input_layout = QHBoxLayout()
        input_panel.setLayout(input_layout)

        self.input_label = QLabel("Размер матрицы (4t-2), где t =")
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Введите целое число ≥1")
        self.input_field.setFixedWidth(150)

        self.generate_btn = QPushButton("Сгенерировать")
        self.generate_btn.clicked.connect(self.generate_matrix)

        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.generate_btn)
        input_layout.addStretch()

        self.layout.addWidget(input_panel)

    def create_matrix_display(self):
        """Область отображения матрицы"""
        self.figure = Figure(figsize=(6, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('equal', adjustable='box')

    def create_info_panel(self):
        """Панель информации и векторов"""
        info_panel = QWidget()
        info_layout = QVBoxLayout()
        info_panel.setLayout(info_layout)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)

        self.vectors_text = QTextEdit()
        self.vectors_text.setReadOnly(True)
        self.vectors_text.setFixedHeight(100)
        self.vectors_text.setFont(QFont("Monospace"))

        info_layout.addWidget(self.info_label)
        info_layout.addWidget(QLabel("Векторы:"))
        info_layout.addWidget(self.vectors_text)

        self.layout.addWidget(info_panel)

    def show_initial_message(self):
        """Показывает начальное сообщение"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Введите t (целое ≥1)\nМатрица будет размером 4t-2",
                     ha='center', va='center', fontsize=12)
        self.ax.axis('off')
        self.canvas.draw()

        self.info_label.setText("Ожидание ввода параметров...")
        self.vectors_text.clear()

    def generate_matrix(self):
        """Генерация и отображение матрицы"""
        try:
            t = int(self.input_field.text())
            self.draw_matrix(t)
        except ValueError as e:
            self.show_error(str(e))

    def draw_matrix(self, t):
        """Рисует матрицу Эйлера, включая особые случаи вроде n=18 и n=30"""
        try:
            n = 4 * t - 2
            q = n // 2
            q2 = n + 1  # n+1 для метода GF

            # Стандартный метод (если q простое)
            if isprime(q):
                a = np.zeros(q, dtype=int)
                for i in range(1, q):
                    a[i] = self.legendre_symbol(i, q)
                a[0] = 1
                b = -a if q % 4 == 3 else a
                b[0] = 1

            # Метод GF(n+1) когда n+1 простое (для n=18,30,66 и др.)
            elif isprime(q2):
                p = q2  # n+1
                chi = np.zeros(p, dtype=int)
                g = primitive_root(p)  # Примитивный элемент GF(p)

                # Заполняем символы Лежандра для GF(p)
                for i in range(1, p):
                    x = pow(g, i, p)
                    chi[x] = 1 if pow(x, (p - 1) // 2, p) == 1 else -1

                # Строим векторы a и b
                a = np.array([chi[(i + 1) % p] for i in range(q)])
                b = np.array([chi[(q + i + 1) % p] for i in range(q)])
                a[0] = 1
                b[0] = 1


            # Универсальный метод: если q = p^m, p — простое
            elif any(isprime(p) and p**m == q for p in range(2, q) for m in range(2, int(np.log2(q)) + 2)):
                for p in range(2, q):
                    if not isprime(p):
                        continue
                    m = 1
                    while p ** m < q:
                        m += 1
                    if p ** m == q:
                        GF = galois.GF(p ** m)
                        elements = GF.elements
                        chi = {}
                        for x in elements[1:]:
                            chi[int(x)] = 1 if x ** ((GF.order - 1) // 2) == 1 else -1
                        chi[0] = 0
                        a = np.array([chi[int(elements[i % len(elements)])] for i in range(q)])
                        b = np.array([chi[int(elements[(i + q) % len(elements)])] for i in range(q)])
                        a[0] = 1
                        b[0] = 1
                        n3p = p

            # Метод сжатия через орбиты циклической группы (универсальный fallback)
            elif q > 2:
                # Универсальный метод: орбиты циклической группы Z_q
                G = np.arange(q)
                used = set()
                orbits = []

                # Используем генератор g=2 (или другой), создаем орбиты
                g = 2
                for i in range(q):
                    if i in used:
                        continue
                    orbit = []
                    j = i
                    while j not in orbit:
                        orbit.append(j)
                        used.add(j)
                        j = (j * g) % q
                    orbits.append(orbit)

                a = np.zeros(q, dtype=int)
                sign = 1
                for orbit in orbits:
                    for idx in orbit:
                        a[idx] = sign
                    sign *= -1

                # b — циклический сдвиг a
                b = np.roll(a, q // 2)

                a[0] = 1
                b[0] = 1
                print(f"Метод орбит циклической группы (g={g}, Z_{q})")

                method = f"Метод орбит (Z_{q} циклическая группа)"

            else:
                raise ValueError(f"Для n={n} (t={t}) нет известного метода построения")

            # Далее идёт стандартная сборка матрицы и отображение...
            A = np.zeros((q, q), dtype=int)
            B = np.zeros((q, q), dtype=int)
            for i in range(q):
                A[i] = np.roll(a, i)
                B[i] = np.roll(b, i)

            H_top = np.hstack((A, B))
            H_bottom = np.hstack((-B.T, A.T))
            matrix = np.vstack((H_top, H_bottom))

            # Отрисовка
            self.ax.clear()
            self.ax.imshow(matrix, cmap='viridis', aspect='equal')
            self.ax.set_title(f'Матрица Эйлера {n}x{n}', pad=20)
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            if n <= 20:
                for i in range(n):
                    for j in range(n):
                        self.ax.text(j, i, str(matrix[i, j]),
                                     ha='center', va='center',
                                     color='white' if abs(matrix[i, j]) == 1 else 'black')

            self.canvas.draw()

            method = (
                'Символы Лежандра' if isprime(q) else
                f'Поле Галуа GF({q2})' if isprime(q2) else
                f'Специальная конструкция  (GF({n3p}^{m}))' if any(isprime(p) and p**m == q for p in range(2, q)
                                                                   for m in range(2, int(np.log2(q)) + 2)) else
                f"Метод орбит циклической группы (g={g}, Z_{q})"
            )

            info_text = f"""
                                <b>Параметры:</b><br>
                                t = {t}, n = 4t-2 = {n}, q = n/2 = {q}, q2 = n+1 = {q2}<br>
                                <b>Метод:</b> {method}
                                """
            self.info_label.setText(info_text)

            vectors_text = f"a=[{','.join(map(str, a))}];\nb=[{','.join(map(str, b))}];"
            self.vectors_text.setText(vectors_text)

        except Exception as e:
            self.show_error(str(e))

    def show_error(self, message):
        """Показывает сообщение об ошибке"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, message,
                     ha='center', va='center',
                     bbox=dict(facecolor='red', alpha=0.5))
        self.ax.axis('off')
        self.canvas.draw()

        self.info_label.setText(f"<font color='red'>Ошибка: {message}</font>")
        self.vectors_text.clear()

    def legendre_symbol(self, a, p):
        """Вычисляет символ Лежандра (a/p)"""
        if a % p == 0:
            return 0
        ls = pow(a, (p - 1) // 2, p)
        return -1 if ls == p - 1 else ls


if __name__ == "__main__":
    app = QApplication([])
    app.setStyle('Fusion')  # Современный стиль

    window = EulerMatrixApp()
    window.show()
    app.exec_()
