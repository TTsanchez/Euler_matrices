import numpy as np
import galois
from sympy import isprime, primitive_root
import os

# Функция для вычисления символа Лежандра
def legendre_symbol(a, p):
    if a % p == 0:
        return 0
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls

# Функция для генерации матриц Эйлера для полей Галуа произвольной степени
def generate_euler_matrix(n):
    q = n // 2
    q2 = n + 1  # n+1 для метода GF

    # Стандартный метод (если q простое)
    if isprime(q):
        a = np.zeros(q, dtype=int)
        for i in range(1, q):
            a[i] = legendre_symbol(i, q)
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
    elif any(isprime(p) and p ** m == q for p in range(2, q) for m in range(2, int(np.log2(q)) + 2)):
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
    else:
        raise ValueError(f"Для n={n} (t={n//4}) нет известного метода построения")

    return a, b

# Функция для сохранения данных в файл
def save_to_file():
    with open("euler_matrices.txt", "w") as f:
        for n in range(6, 379, 4):  # Только чётные n
            try:
                a, b = generate_euler_matrix(n)
                f.write(f"if (n=={n}) {{\n")
                f.write(f"a=[{','.join(map(str, a))}];\n")  # Используем квадратные скобки
                f.write(f"b=[{','.join(map(str, b))}];\n")  # Используем квадратные скобки
                f.write(f"}}\n\n")
            except Exception as e:
                print(f"Ошибка для n={n}: {e}")

if __name__ == "__main__":
    save_to_file()
    print("Данные сохранены в файл 'euler_matrices.txt'")
