"""Formulario: errores, raices de ecuaciones, sucesiones y series.

Cada funcion tiene arriba, en comentarios, las formulas que usa.

    python3 formulas.py    -> corre un ejemplo de cada tema, con conclusiones

Para un ejercicio nuevo, las conclusiones salen de (ver seccion 11):
    analizar(f, df, a, b, tol)              -> tabla de metodos y conclusiones
    conclusion_tolerancias(f, df, a, b)     -> que cambia al pedir mas precision
    conclusion_sucesion(a)                  -> a(n) converge o diverge
    conclusion_serie(a)                     -> la serie de a(k) converge o diverge
    conclusion_aproximacion(real, aprox, tol, terminos)

Indice (buscar con Ctrl+F):
    1. ERRORES
    2. BISECCION
    3. REGLA FALSA
    4. SECANTE
    5. NEWTON
    6. COMPARACION DE METODOS
    7. SUCESIONES
    8. SERIES
    9. CRITERIOS DE CONVERGENCIA
   10. SERIES DE POTENCIAS Y TAYLOR
   11. CONCLUSIONES
"""

import math
import textwrap

MAX_IT = 200


# =============================================================================
# 1. ERRORES
# =============================================================================
# Con el valor real conocido:
#   Error absoluto:    Ea = |real - aprox|
#   Error relativo:    Er = |real - aprox| / |real|         (real != 0)
#   Error porcentual:  Ep = Er * 100 %
#
# Sin el valor real (metodos iterativos), se compara con la aproximacion anterior:
#   Error absoluto aproximado:    |p_n - p_(n-1)|
#   Error relativo aproximado:    |p_n - p_(n-1)| / |p_n|   (p_n != 0)
#   Error porcentual aproximado:  relativo * 100 %
#   Criterio de parada:           error < tolerancia
#
# Equivalencias: 1% = 10^-2 | 0.01% = 10^-4 | 0.001% = 10^-5 (relativo)
#
# Cifras significativas (Scarborough): para asegurar al menos n cifras correctas
#   tolerancia porcentual:  Es = 0.5 * 10^(2 - n) %
#
# Tipos de error:
#   Redondeo:    la computadora guarda una cantidad finita de digitos.
#   Truncamiento: se corta un proceso infinito (ej: usar pocos terminos de una serie).
#   Error total = redondeo + truncamiento


def error_absoluto(real, aprox):
    return abs(real - aprox)


def error_relativo(real, aprox):
    return abs((real - aprox) / real)


def error_porcentual(real, aprox):
    return error_relativo(real, aprox) * 100


def error_aproximado(nuevo, viejo):
    # Relativo entre iteraciones; si nuevo = 0 se usa el absoluto para no dividir por 0.
    return abs((nuevo - viejo) / nuevo) if nuevo != 0 else abs(nuevo - viejo)


def tolerancia_cifras(n):
    # Es = 0.5 * 10^(2 - n) %
    return 0.5 * 10 ** (2 - n)


# =============================================================================
# 2. BISECCION (metodo cerrado)
# =============================================================================
# Requisito (Bolzano): f continua en [a, b] y f(a) * f(b) < 0
#
# Formula:  p = (a + b) / 2
#
# Eleccion del subintervalo:
#   f(a) * f(p) < 0  ->  la raiz esta en [a, p]  ->  b = p
#   f(a) * f(p) > 0  ->  la raiz esta en [p, b]  ->  a = p
#   f(p) = 0         ->  p es la raiz
# Si f(a) = 0 o f(b) = 0, ese extremo ya es la raiz (0 iteraciones).
#
# Cota del error:  |p_n - r| <= (b - a) / 2^n
# Iteraciones para error absoluto < E:  n >= log2((b - a) / E)
# Convergencia: lineal (el intervalo se divide por 2 en cada paso).


def biseccion(f, a, b, tol=1e-4, historial=None):
    if f(a) * f(b) > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos")
    if f(a) == 0 or f(b) == 0:
        return (a if f(a) == 0 else b), 0
    p_ant = None
    for n in range(1, MAX_IT + 1):
        p = (a + b) / 2
        if historial is not None:
            historial.append((a, b, p))
        if f(p) == 0 or (p_ant is not None and error_aproximado(p, p_ant) < tol):
            return p, n
        if f(a) * f(p) < 0:
            b = p
        else:
            a = p
        p_ant = p
    raise RuntimeError("no converge")


def iteraciones_biseccion(a, b, error):
    # n >= log2((b - a) / E)
    return math.ceil(math.log2((b - a) / error))


# =============================================================================
# 3. REGLA FALSA o POSICION FALSA (metodo cerrado)
# =============================================================================
# Requisito: igual que biseccion, f(a) * f(b) < 0
#
# p es donde la recta que une (a, f(a)) y (b, f(b)) corta al eje x:
#   p = b - f(b) * (b - a) / (f(b) - f(a))
#   p = (a * f(b) - b * f(a)) / (f(b) - f(a))     (forma equivalente)
#
# Eleccion del subintervalo: igual que biseccion.
# Convergencia: lineal; un extremo puede quedar fijo si f es muy curva.


def regla_falsa(f, a, b, tol=1e-4, historial=None):
    if f(a) * f(b) > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos")
    p_ant = None
    for n in range(1, MAX_IT + 1):
        p = b - f(b) * (b - a) / (f(b) - f(a))
        if historial is not None:
            historial.append((a, b, p))
        if f(p) == 0 or (p_ant is not None and error_aproximado(p, p_ant) < tol):
            return p, n
        if f(a) * f(p) < 0:
            b = p
        else:
            a = p
        p_ant = p
    raise RuntimeError("no converge")


# =============================================================================
# 4. SECANTE (metodo abierto)
# =============================================================================
# Necesita dos puntos iniciales p0 y p1. No exige cambio de signo ni derivada.
#
# Formula:
#   p_(n+1) = p_n - f(p_n) * (p_n - p_(n-1)) / (f(p_n) - f(p_(n-1)))
#
# Sale de Newton, cambiando la derivada por la pendiente de la secante:
#   f'(p_n) ~ (f(p_n) - f(p_(n-1))) / (p_n - p_(n-1))
#
# Falla si f(p_n) = f(p_(n-1)). Puede escaparse del intervalo.
# Convergencia: superlineal, orden ~ 1.618
#   e_(n+1) ~ |f''(r) / (2 * f'(r))| * e_n * e_(n-1)


def secante(f, p0, p1, tol=1e-4, historial=None):
    if historial is not None:
        historial.extend([p0, p1])
    for n in range(1, MAX_IT + 1):
        if f(p1) == f(p0):
            raise ZeroDivisionError("f(p1) = f(p0)")
        p = p1 - f(p1) * (p1 - p0) / (f(p1) - f(p0))
        if historial is not None:
            historial.append(p)
        if f(p) == 0 or error_aproximado(p, p1) < tol:
            return p, n
        p0, p1 = p1, p
    raise RuntimeError("no converge")


# =============================================================================
# 5. NEWTON-RAPHSON (metodo abierto)
# =============================================================================
# Necesita un punto inicial p0 y la derivada f'(x).
#
# Formula (donde la tangente en p_n corta al eje x):
#   p_(n+1) = p_n - f(p_n) / f'(p_n)
#
# Recta tangente:  y = f(p_n) + f'(p_n) * (x - p_n)
#
# Falla o se aleja si f'(p_n) ~ 0. Depende de elegir bien p0.
# Convergencia: cuadratica si la raiz es simple (f'(r) != 0)
#   e_(n+1) ~ |f''(r) / (2 * f'(r))| * e_n^2
# Si la raiz tiene multiplicidad m, es lineal. Se recupera con:
#   p_(n+1) = p_n - m * f(p_n) / f'(p_n)
#
# Raices de un numero:
#   raiz n-esima de A  ->  f(x) = x^n - A,  f'(x) = n * x^(n-1)
#   raiz cuadrada de A ->  p_(n+1) = (p_n + A / p_n) / 2


def newton(f, df, p0, tol=1e-4, historial=None):
    if historial is not None:
        historial.append(p0)
    for n in range(1, MAX_IT + 1):
        if df(p0) == 0:
            raise ZeroDivisionError("f'(p0) = 0")
        p = p0 - f(p0) / df(p0)
        if historial is not None:
            historial.append(p)
        if f(p) == 0 or error_aproximado(p, p0) < tol:
            return p, n
        p0 = p
    raise RuntimeError("no converge")


# =============================================================================
# 6. COMPARACION DE METODOS
# =============================================================================
# Metodo     | Tipo    | Necesita          | Converge siempre | Orden
# -----------+---------+-------------------+------------------+-------
# Biseccion  | Cerrado | a, b con cambio   | Si               | 1
# Regla falsa| Cerrado | a, b con cambio   | Si               | 1
# Secante    | Abierto | p0, p1            | No               | ~1.618
# Newton     | Abierto | p0 y f'           | No               | 2
#
# Orden de convergencia p:  e_(n+1) ~ C * e_n^p


# =============================================================================
# 7. SUCESIONES
# =============================================================================
# Sucesion: lista ordenada de numeros a_1, a_2, a_3, ...
#
# Aritmetica (diferencia d constante):
#   a_n = a_1 + (n - 1) * d
#   Suma de n terminos:  S_n = n * (a_1 + a_n) / 2
#
# Geometrica (razon r constante):
#   a_n = a_1 * r^(n - 1)
#   Suma de n terminos:  S_n = a_1 * (1 - r^n) / (1 - r)     (r != 1)
#
# Convergencia: a_n -> L si lim (n -> inf) a_n = L finito. Si no, diverge.
#   Monotona y acotada  ->  converge
#   Algebra de limites:  lim (a_n + b_n) = lim a_n + lim b_n
#                        lim (a_n * b_n) = lim a_n * lim b_n
#                        lim (a_n / b_n) = lim a_n / lim b_n   (lim b_n != 0)
#   Sandwich: si a_n <= b_n <= c_n y a_n, c_n -> L, entonces b_n -> L
#
# Limites notables:
#   r^n -> 0 si |r| < 1;  -> 1 si r = 1;  diverge si r > 1 o r <= -1
#   1 / n^p -> 0 (p > 0)
#   n^(1/n) -> 1
#   a^(1/n) -> 1 (a > 0)
#   (1 + 1/n)^n -> e
#   (1 + x/n)^n -> e^x
#   x^n / n! -> 0
#   ln(n) / n -> 0


def aritmetica_termino(a1, d, n):
    return a1 + (n - 1) * d


def aritmetica_suma(a1, d, n):
    return n * (a1 + aritmetica_termino(a1, d, n)) / 2


def geometrica_termino(a1, r, n):
    return a1 * r ** (n - 1)


def geometrica_suma(a1, r, n):
    return a1 * n if r == 1 else a1 * (1 - r**n) / (1 - r)


def terminos(a, n, inicio=1):
    # Primeros n terminos de la sucesion a(k).
    return [a(k) for k in range(inicio, inicio + n)]


# =============================================================================
# 8. SERIES
# =============================================================================
# Serie: suma infinita de los terminos de una sucesion.
#   S_n = a_1 + a_2 + ... + a_n     (suma parcial)
#   Serie = lim (n -> inf) S_n      (converge si el limite es finito)
#
# Propiedad: si la serie converge, entonces a_n -> 0.
# La reciproca es FALSA: la armonica tiene a_n -> 0 y diverge.
#
# Geometrica:  sum (k=0..inf) a * r^k
#   Converge a a / (1 - r) si |r| < 1.  Diverge si |r| >= 1 (a != 0).
#   Error al usar n terminos, k=0..n-1:  |a * r^n / (1 - r)|
#
# Serie p:  sum (k=1..inf) 1 / k^p
#   Converge si p > 1.  Diverge si p <= 1.
#   Armonica (p = 1):  1 + 1/2 + 1/3 + ...  diverge.
#
# Telescopica:  sum (k=1..n) (b_k - b_(k+1)) = b_1 - b_(n+1)
#   Ejemplo:  1 / (k * (k+1)) = 1/k - 1/(k+1)  ->  la suma infinita da 1
#
# Alternante:  sum (k=1..inf) (-1)^(k+1) * b_k,  con b_k > 0
#   Leibniz: si b_k decrece y b_k -> 0, converge.
#   Cota del error:  |S - S_n| <= b_(n+1)
#
# Convergencia absoluta: si sum |a_k| converge, entonces sum a_k converge.
# Convergencia condicional: sum a_k converge pero sum |a_k| diverge.


def suma_parcial(a, n, inicio=1):
    # S_n = a(inicio) + ... + a(inicio + n - 1)
    return math.fsum(a(k) for k in range(inicio, inicio + n))


def geometrica_infinita(a, r):
    # S = a / (1 - r), con |r| < 1
    if abs(r) >= 1:
        raise ValueError("diverge: hace falta |r| < 1")
    return a / (1 - r)


def serie_alternante(b, tol=1e-4, inicio=1):
    # Suma (-1)^(k-inicio) * b(k) hasta que la cota de Leibniz b_(n+1) sea < tol.
    # Solo vale si b(k) es positiva, decreciente y tiende a 0.
    suma = 0.0
    signo = 1
    for n in range(1, 1_000_000):
        k = inicio + n - 1
        suma += signo * b(k)
        signo = -signo
        if b(k + 1) < tol:
            return suma, n
    raise RuntimeError("no alcanzo la tolerancia")


# =============================================================================
# 9. CRITERIOS DE CONVERGENCIA
# =============================================================================
# Divergencia:  si a_n no tiende a 0, la serie diverge.
#               Si a_n -> 0, este criterio no decide nada.
#
# Cociente (D'Alembert):   L = lim |a_(n+1) / a_n|
#   L < 1 converge absolutamente | L > 1 diverge | L = 1 no decide
#
# Raiz (Cauchy):           L = lim |a_n|^(1/n)
#   L < 1 converge absolutamente | L > 1 diverge | L = 1 no decide
#
# Comparacion directa (0 <= a_n <= b_n):
#   sum b_n converge  ->  sum a_n converge
#   sum a_n diverge   ->  sum b_n diverge
#
# Comparacion en el limite (a_n, b_n > 0):  L = lim a_n / b_n
#   0 < L < inf  ->  las dos series hacen lo mismo
#
# Integral (f positiva, continua y decreciente, f(n) = a_n):
#   sum a_n converge  <->  integral de 1 a inf de f(x) dx converge
#   Resto:  int (n+1..inf) f <= S - S_n <= int (n..inf) f
#
# Cuidado: las funciones de abajo solo muestran valores para distintos n.
# Sirven para ver hacia donde va; lo que decide es el limite, no un valor suelto.


def cociente(a, ns=(10, 50, 100)):
    # |a_(n+1) / a_n| para varios n
    return {n: abs(a(n + 1) / a(n)) for n in ns}


def raiz_enesima(a, ns=(10, 50, 100)):
    # |a_n|^(1/n) para varios n
    return {n: abs(a(n)) ** (1 / n) for n in ns}


# =============================================================================
# 10. SERIES DE POTENCIAS Y TAYLOR
# =============================================================================
# Serie de potencias:  sum (k=0..inf) c_k * (x - x0)^k
#   Radio de convergencia:  R = lim |c_n / c_(n+1)|  o  R = 1 / lim |c_n|^(1/n)
#   Converge si |x - x0| < R, diverge si |x - x0| > R.
#   En |x - x0| = R hay que probar cada extremo por separado.
#
# Taylor alrededor de x0 (Maclaurin si x0 = 0):
#   f(x) = sum (k=0..inf) f^(k)(x0) / k! * (x - x0)^k
#
# Resto de Lagrange, con xi entre x0 y x:
#   R_n(x) = f^(n+1)(xi) / (n+1)! * (x - x0)^(n+1)
#
# Maclaurin mas usadas:
#   e^x      = sum x^k / k!                              para todo x
#   sen x    = sum (-1)^k * x^(2k+1) / (2k+1)!           para todo x
#   cos x    = sum (-1)^k * x^(2k) / (2k)!               para todo x
#   1/(1-x)  = sum x^k                                   |x| < 1
#   ln(1+x)  = sum (k=1..inf) (-1)^(k+1) * x^k / k       -1 < x <= 1
#   atan x   = sum (-1)^k * x^(2k+1) / (2k+1)            |x| <= 1
#   (1+x)^m  = sum C(m, k) * x^k                         |x| < 1


def taylor(termino, tol=1e-4, inicio=0):
    # Suma termino(k) hasta que el error relativo entre sumas sea < tol.
    # Es un error aproximado: compara con la suma anterior, no con el valor real.
    suma = termino(inicio)
    for n in range(2, MAX_IT + 1):
        anterior = suma
        suma += termino(inicio + n - 1)
        if error_aproximado(suma, anterior) < tol:
            return suma, n
    raise RuntimeError("no converge")


def exp_taylor(x, tol=1e-4):
    return taylor(lambda k: x**k / math.factorial(k), tol)


def sen_taylor(x, tol=1e-4):
    return taylor(lambda k: (-1) ** k * x ** (2 * k + 1) / math.factorial(2 * k + 1), tol)


def cos_taylor(x, tol=1e-4):
    return taylor(lambda k: (-1) ** k * x ** (2 * k) / math.factorial(2 * k), tol)


def ln1mas_taylor(x, tol=1e-4):
    # Solo para -1 < x <= 1
    return taylor(lambda k: (-1) ** (k + 1) * x**k / k, tol, inicio=1)


# =============================================================================
# 11. CONCLUSIONES
# =============================================================================
# Las conclusiones se arman con lo que paso en cada corrida. Son evidencia numerica:
# orientan y sirven para justificar, pero no reemplazan una demostracion.
#
# Orden estimado, con d_n = |p_n - p_(n-1)|:
#   q ~ ln(d_(n+1) / d_n) / ln(d_n / d_(n-1))
#
# Multiplicidad estimada en Newton, si converge linealmente con razon C = d_(n+1) / d_n:
#   m ~ 1 / (1 - C)
#
# Iteraciones de biseccion al cortar por error relativo < tol, con raiz r != 0:
#   d_n = (b - a) / 2^n < tol * |r|   ->   n ~ log2((b - a) / (tol * |r|))
#
# Cifras significativas alcanzadas: la mayor n con Ep < 0.5 * 10^(2 - n) %


def _imprimir(conclusiones):
    print("Conclusiones:")
    for c in conclusiones:
        print(textwrap.fill(c, 92, initial_indent="- ", subsequent_indent="  "))


def _sin_cero(valor):
    return f"{valor:.8f}".rstrip("0").rstrip(".")


def _diferencias(ps):
    # Saca las diferencias que ya son ruido de redondeo.
    d = [abs(y - x) for x, y in zip(ps, ps[1:])]
    return [v for v in d if v > 1e-13 * max(abs(ps[-1]), 1)]


def orden_estimado(ps):
    d = _diferencias(ps)
    if len(d) < 3 or d[-1] == d[-2] or d[-2] == d[-3]:
        return None
    try:
        return math.log(d[-1] / d[-2]) / math.log(d[-2] / d[-3])
    except (ValueError, ZeroDivisionError):
        return None


def _raiz_refinada(f, p):
    # Mejor estimacion de la raiz, para medir el error real de cada metodo.
    try:
        return secante(f, p, p + 1e-6 * max(abs(p), 1), 1e-13)[0]
    except (ZeroDivisionError, RuntimeError, OverflowError, ValueError):
        return p


def _cambios_de_signo(f, a, b, muestras=2000):
    cambios, anterior = 0, None
    for i in range(muestras + 1):
        try:
            y = f(a + (b - a) * i / muestras)
        except (ValueError, ZeroDivisionError, OverflowError):
            anterior = None
            continue
        if anterior is not None and anterior * y < 0:
            cambios += 1
        if y != 0:
            anterior = y
    return cambios


def _correr(f, df, a, b, tol, x0):
    x0 = (a + b) / 2 if x0 is None else x0
    llamadas = {
        "Biseccion": lambda h: biseccion(f, a, b, tol, h),
        "Regla falsa": lambda h: regla_falsa(f, a, b, tol, h),
        "Secante": lambda h: secante(f, a, b, tol, h),
        "Newton": lambda h: newton(f, df, x0, tol, h),
    }
    corridas = {}
    for nombre, llamar in llamadas.items():
        hist = []
        try:
            raiz, n = llamar(hist)
            corridas[nombre] = {"raiz": raiz, "n": n, "hist": hist}
        except (ValueError, ZeroDivisionError, RuntimeError, OverflowError) as ex:
            corridas[nombre] = {"error": str(ex), "hist": hist}
    return corridas


def _puntos(nombre, hist):
    return [h[2] for h in hist] if nombre in ("Biseccion", "Regla falsa") else hist


def analizar(f, df, a, b, tol=1e-4, x0=None):
    """Corre los 4 metodos en [a, b] e imprime la comparacion y sus conclusiones."""
    x0 = (a + b) / 2 if x0 is None else x0
    corridas = _correr(f, df, a, b, tol, x0)
    lo, hi = min(a, b), max(a, b)

    print(f"{'Metodo':12} {'Raiz':>13} {'Iter.':>5} {'|f(raiz)|':>10} {'Er real':>10} {'Orden':>6}  Obs.")
    for nombre, c in corridas.items():
        if "error" in c:
            print(f"{nombre:12} {'-':>13} {'-':>5} {'-':>10} {'-':>10} {'-':>6}  {c['error']}")
            continue
        c["q"] = orden_estimado(_puntos(nombre, c["hist"]))
        c["ref"] = _raiz_refinada(f, c["raiz"])
        c["er"] = error_aproximado(c["ref"], c["raiz"])
        c["fuera"] = not lo <= c["raiz"] <= hi
        q = f"{c['q']:.2f}" if c["q"] is not None else "-"
        obs = "FUERA DEL INTERVALO" if c["fuera"] else ""
        print(f"{nombre:12} {c['raiz']:13.8f} {c['n']:5d} {abs(f(c['raiz'])):10.2e} {c['er']:10.2e} {q:>6}  {obs}")

    ok = {n: c for n, c in corridas.items() if "error" not in c}
    adentro = {n: c for n, c in ok.items() if not c["fuera"]}
    salida = []

    # Existencia de la raiz
    cambios = _cambios_de_signo(f, a, b)
    try:
        producto = f(a) * f(b)
    except (ValueError, ZeroDivisionError, OverflowError):
        producto = None
    if producto is None:
        salida.append("f no esta definida en algun extremo del intervalo: no se puede aplicar Bolzano en [a, b].")
    elif producto == 0:
        salida.append("Un extremo del intervalo ya es raiz: f(a) * f(b) = 0.")
    elif producto < 0:
        texto = (f"Bolzano: f(a) = {f(a):.4g} y f(b) = {f(b):.4g} tienen signos opuestos; si f es "
                 "continua en [a, b], hay al menos una raiz.")
        if cambios > 1:
            texto += f" El muestreo detecta {cambios} cambios de signo: puede haber varias raices."
        salida.append(texto)
    else:
        salida.append("Sin cambio de signo en los extremos: biseccion y regla falsa no se pueden aplicar. "
                      "Esto no prueba que no haya raices: puede haber una cantidad par o una raiz doble.")

    # Velocidad
    if len(adentro) > 1:
        minimo = min(c["n"] for c in adentro.values())
        maximo = max(c["n"] for c in adentro.values())
        rapidos = [n for n, c in adentro.items() if c["n"] == minimo]
        lentos = [n for n, c in adentro.items() if c["n"] == maximo]
        if minimo == maximo:
            salida.append(f"Todos los metodos que quedaron en el intervalo usaron {minimo} iteraciones.")
        else:
            salida.append(f"Menos iteraciones: {' y '.join(rapidos)} ({minimo}). Mas iteraciones: "
                          f"{' y '.join(lentos)} ({maximo})" + (f", unas {maximo / minimo:.1f} veces mas." if minimo else "."))
    abiertos = [c["n"] for n, c in adentro.items() if n in ("Secante", "Newton")]
    cerrados = [c["n"] for n, c in adentro.items() if n in ("Biseccion", "Regla falsa")]
    if abiertos and cerrados:
        if max(abiertos) < min(cerrados):
            salida.append("Coincide con la teoria: los metodos abiertos convergen mas rapido, pero los "
                          "cerrados dan la garantia de no perder la raiz.")
        else:
            salida.append("Aca no hay una separacion clara entre metodos abiertos y cerrados: el "
                          "orden de convergencia se nota mas cuando se pide mayor precision.")

    # Biseccion contra su prediccion
    bis = ok.get("Biseccion")
    if bis and bis["ref"] != 0 and bis["n"] > 1:
        predicho = math.ceil(math.log2(abs(b - a) / (tol * abs(bis["ref"]))))
        salida.append(f"Biseccion: n ~ log2((b - a) / (tol * |r|)) predice {predicho} iteraciones y uso "
                      f"{bis['n']}. Depende del ancho del intervalo, de la tolerancia y del tamano de la "
                      "raiz, pero no de la forma de f: por eso es tan predecible.")

    # Regla falsa: extremo fijo
    rf = ok.get("Regla falsa")
    if rf and rf["n"] >= 3:
        for letra, i in (("a", 0), ("b", 1)):
            fijo = rf["hist"][-1][i]
            veces = sum(h[i] == fijo for h in rf["hist"])
            if veces >= 0.7 * rf["n"]:
                salida.append(f"Regla falsa: el extremo {letra} = {_sin_cero(fijo)} quedo fijo en {veces} de "
                              f"{rf['n']} iteraciones. Es lo tipico si f no cambia de concavidad en el "
                              "intervalo, y por eso converge linealmente.")
                break
    if rf and bis:
        if rf["n"] > bis["n"]:
            salida.append("Regla falsa fue mas lenta que biseccion: usar los valores de f no siempre "
                          "ayuda, sobre todo si f es muy curva en el intervalo.")
        elif rf["n"] < bis["n"]:
            salida.append("Regla falsa mejoro a biseccion porque aprovecha los valores de f, no solo su signo.")

    # Orden de convergencia
    q = ok.get("Newton", {}).get("q")
    if q is not None:
        if q >= 1.5:
            salida.append(f"Newton: orden estimado {q:.2f}, compatible con convergencia cuadratica (raiz simple).")
        elif q < 1.3:
            d = _diferencias(_puntos("Newton", ok["Newton"]["hist"]))
            razon = d[-1] / d[-2]
            texto = f"Newton: orden estimado {q:.2f}, convergencia lineal con razon {razon:.2f}. "
            if razon < 0.95:
                texto += (f"Indica una raiz multiple de multiplicidad m ~ {1 / (1 - razon):.1f}, donde "
                          "f'(r) = 0. Se recupera el orden 2 con p_(n+1) = p_n - m * f(p_n) / f'(p_n).")
            salida.append(texto)
        else:
            salida.append(f"Newton: orden estimado {q:.2f}, todavia lejos de 2. Con pocas iteraciones "
                          "la estimacion no es confiable; se ve mejor con una tolerancia mas chica.")
    q = ok.get("Secante", {}).get("q")
    if q is not None:
        compatible = "compatible con el orden teorico ~1.618" if q >= 1.3 else "por debajo del orden teorico ~1.618 (la estimacion usa solo los ultimos pasos, no es definitiva)"
        salida.append(f"Secante: orden estimado {q:.2f}, {compatible}.")

    # Escapes y fallas
    for nombre, c in ok.items():
        if c["fuera"]:
            inicio = f"p0 = {_sin_cero(x0)}, f'(p0) = {df(x0):.4g}" if nombre == "Newton" else f"p0 = {_sin_cero(a)}, p1 = {_sin_cero(b)}"
            salida.append(f"{nombre} convergio a {_sin_cero(c['raiz'])}, FUERA de [{_sin_cero(a)}, {_sin_cero(b)}] "
                          f"(arranco con {inicio}). Los metodos abiertos no garantizan quedarse en el "
                          "intervalo: hay que elegir puntos iniciales mas cerca de la raiz buscada.")
    if bis and any(c["fuera"] for c in ok.values()):
        salida.append(f"Una forma de arreglarlo: arrancar cerca de la raiz de biseccion, {_sin_cero(bis['raiz'])}.")
    for nombre, c in corridas.items():
        if "error" in c and nombre in ("Secante", "Newton"):
            salida.append(f"{nombre} fallo ({c['error']}): no tiene convergencia garantizada.")

    # Distintas raices
    distintas = sorted({round(c["ref"], 6) for c in ok.values()})
    if len(distintas) > 1:
        salida.append("Los metodos llegaron a raices distintas (" + ", ".join(map(str, distintas)) +
                      "): la ecuacion tiene mas de una solucion y cada metodo encuentra una segun su arranque.")

    # Error real
    malos = [n for n, c in ok.items() if c["er"] >= tol]
    if malos:
        salida.append(f"{' y '.join(malos)}: el error aproximado cumplio la tolerancia pero el error real no. "
                      "El criterio mide el cambio entre iteraciones, no la distancia a la raiz.")
    elif ok:
        salida.append(f"Los resultados obtenidos cumplen la tolerancia {tol:g} tambien en error real, estimado "
                      "contra una raiz refinada.")
    _imprimir(salida)
    return corridas


def conclusion_tolerancias(f, df, a, b, tols=(1e-4, 1e-6, 1e-8), x0=None):
    """Compara las iteraciones de cada metodo al pedir mas precision."""
    tabla = {}
    escapados = set()
    print(f"{'Metodo':12}" + "".join(f"{t:>9g}" for t in tols))
    for t in tols:
        for nombre, c in _correr(f, df, a, b, t, x0).items():
            tabla.setdefault(nombre, []).append(c.get("n"))
            if "raiz" in c and not min(a, b) <= c["raiz"] <= max(a, b):
                escapados.add(nombre)
    for nombre, fila in tabla.items():
        print(f"{nombre:12}" + "".join(f"{'-' if n is None else n:>9}" for n in fila))

    salida = []
    factor = math.log2(tols[0] / tols[-1])
    sumas = {n: fila[-1] - fila[0] for n, fila in tabla.items() if None not in fila}
    if "Biseccion" in sumas:
        salida.append(f"Biseccion sumo {sumas['Biseccion']} iteraciones. La teoria predice log2({tols[0]:g} / "
                      f"{tols[-1]:g}) = {factor:.1f}: siempre el mismo costo por cada digito (~3.3 iteraciones).")
    if "Regla falsa" in sumas:
        salida.append(f"Regla falsa sumo {sumas['Regla falsa']} iteraciones: al ser lineal, tambien crece "
                      "de forma proporcional a los digitos pedidos.")
    rapidos = [f"{n} sumo {sumas[n]}" for n in ("Secante", "Newton") if n in sumas]
    if rapidos:
        salida.append(", ".join(rapidos) + ". Cerca de la raiz los digitos correctos se multiplican en "
                      "cada paso, asi que pedir mucha mas precision cuesta muy pocas iteraciones.")
    if escapados:
        salida.append(f"Ojo: {' y '.join(sorted(escapados))} convergieron fuera de [a, b], asi que sus "
                      "iteraciones corresponden a otra raiz.")
    _imprimir(salida)


def conclusion_aproximacion(real, aprox, tol, terminos=None):
    """Compara una aproximacion con el valor real, por ejemplo una suma de Taylor."""
    er = error_relativo(real, aprox)
    salida = [f"Valor real {real:.10g}, aproximacion {aprox:.10g}, error relativo real {er:.2e}."]
    if er == 0:
        salida.append("La aproximacion coincide con el valor real en la precision de la maquina.")
    else:
        cifras = max(0, math.floor(2 - math.log10(er * 100 / 0.5)))
        salida.append(f"Por Scarborough, el resultado tiene al menos {cifras} cifras significativas correctas.")
    salida.append("Cumple la tolerancia pedida tambien en error real." if er < tol else
                  "NO cumple la tolerancia en error real: el criterio de parada no alcanzo a controlarlo.")
    if terminos is not None:
        salida.append(f"Con {terminos} terminos, este error es de truncamiento: se corta una suma infinita. "
                      "Mas terminos lo achican hasta que domina el redondeo.")
    _imprimir(salida)


def conclusion_sucesion(a):
    """Estimacion numerica del comportamiento de a(n), mirando n grandes."""
    salida = []
    try:
        v1, v2, v3 = a(100), a(1000), a(10000)
        d1, d2 = abs(v2 - v1), abs(v3 - v2)
        salto = abs(a(10001) - v3)
        razon = d2 / d1 if d1 else (0 if d2 == 0 else float("inf"))
    except OverflowError:
        salida.append("Los valores crecen hasta desbordar: la sucesion parece no estar acotada, asi que diverge.")
        _imprimir(salida)
        return
    if d1 == 0 and d2 == 0 and salto == 0:
        salida.append(f"Los valores no cambian: parece constante, y una sucesion constante converge a {v3:.6g}.")
    elif salto > 1e-9 * max(abs(v3), 1) and salto > 0.5 * d2:
        salida.append(f"Dos terminos seguidos, a(10000) y a(10001), difieren en {salto:.4g}: no se estabiliza, "
                      "parece oscilar y por lo tanto diverge.")
    elif razon >= 0.7:
        salida.append(f"a(100) = {v1:.6g}, a(1000) = {v2:.6g}, a(10000) = {v3:.6g}: las diferencias no se achican. "
                      "Parece diverger, aunque puede ser muy lenta: hay que probarlo con el limite.")
    else:
        salida.append(f"a(100) = {v1:.6g}, a(1000) = {v2:.6g}, a(10000) = {v3:.6g}: las diferencias se achican. "
                      f"Parece converger a un valor cercano a {v3:.6g}; el limite exacto hay que calcularlo.")
        # Terminos consecutivos: con n de a saltos no se nota si alterna de signo.
        if (a(10001) - v3) * (a(10002) - a(10001)) > 0:
            salida.append("Los terminos consecutivos no cambian de direccion, parece monotona. "
                          "Una sucesion monotona y acotada siempre converge.")
        else:
            salida.append("Los terminos consecutivos oscilan alrededor del limite, pero cada vez menos.")
    _imprimir(salida)


def conclusion_serie(a):
    """Aplica los criterios de forma numerica a la serie de a(k), con k >= 1."""
    salida = []

    # Divergencia: si a_n no tiende a 0, la serie diverge
    try:
        a100, a1000 = abs(a(100)), abs(a(1000))
        no_tiende = a1000 > 1e-6 and a1000 >= 0.9 * a100
    except OverflowError:
        a100, a1000, no_tiende = None, None, True
    if no_tiende:
        salida.append("Los terminos no parecen tender a 0: por el criterio de divergencia, la serie diverge.")
        _imprimir(salida)
        return

    # Cociente, extrapolando: si v(n) ~ L + c/n, entonces L ~ 2 * v(2n) - v(n)
    L = None
    try:
        v = cociente(a, (50, 100))
        L = 2 * v[100] - v[50]
    except (ZeroDivisionError, OverflowError):
        pass
    if L is not None and L < 0.99:
        salida.append(f"Criterio del cociente: L ~ {max(L, 0):.4f} < 1, parece converger absolutamente.")
        _imprimir(salida)
        return
    if L is not None and L > 1.01:
        salida.append(f"Criterio del cociente: L ~ {L:.4f} > 1, la serie parece diverger.")
        _imprimir(salida)
        return
    if L is not None:
        salida.append(f"Criterio del cociente: L ~ {L:.4f}, muy cerca de 1: el criterio no decide.")

    # Comparacion con una serie p: si a_n ~ C / n^p, entonces p ~ log2(a(n) / a(2n))
    p = None
    try:
        p = math.log2(a1000 / abs(a(2000)))
    except (ZeroDivisionError, ValueError, OverflowError):
        pass
    if p is not None:
        salida.append(f"Los terminos decrecen como 1 / k^p con p ~ {p:.2f} (orientativo).")

    # Alternante
    alterna = all(a(k) * a(k + 1) < 0 for k in range(1, 30))
    if alterna:
        decrece = all(abs(a(k + 1)) < abs(a(k)) for k in range(1, 200))
        if decrece:
            salida.append("Es alternante y |a_k| decrece a 0: converge por Leibniz, con error "
                          "|S - S_n| <= |a_(n+1)|.")
            if p is not None and p > 1.2:
                salida.append("Como la serie de |a_k| se comporta como una serie p con p > 1, la convergencia es absoluta.")
            elif p is not None and p < 1.2:
                salida.append("La serie de |a_k| no converge o queda en duda (p <= 1 da divergencia): "
                              "si diverge, la convergencia es condicional.")
        else:
            salida.append("Es alternante, pero |a_k| no decrece en los primeros terminos: Leibniz no se aplica directo.")
    elif p is not None:
        if p > 1.2:
            salida.append("Por comparacion en el limite con una serie p con p > 1, parece converger.")
        elif p < 0.8:
            salida.append("Por comparacion en el limite con una serie p con p < 1, parece diverger.")
        else:
            salida.append("Con p tan cerca de 1 no alcanza la estimacion: usar criterio de la integral o "
                          "comparacion. Ojo: 1 / (k * ln k) diverge aunque se parezca a p > 1.")
    salida.append("Es evidencia numerica: en la entrega hay que justificar el limite o el criterio.")
    _imprimir(salida)


# =============================================================================
# EJEMPLOS
# =============================================================================
if __name__ == "__main__":
    print("--- Errores: aproximar pi con 3.14 ---")
    print("absoluto:", error_absoluto(math.pi, 3.14))
    print("relativo:", error_relativo(math.pi, 3.14))
    print("porcentual:", error_porcentual(math.pi, 3.14))
    print("tolerancia para 4 cifras significativas (%):", tolerancia_cifras(4))

    # Ej. 1 del TP en [0, 1]; la raiz exacta es 2 - sqrt(2)
    f = lambda x: x**3 - 7 * x**2 + 14 * x - 6
    df = lambda x: 3 * x**2 - 14 * x + 14
    print("\n--- Raices: x^3 - 7x^2 + 14x - 6 = 0 en [0, 1], tol = 1e-4 ---")
    print("biseccion:  ", biseccion(f, 0, 1))
    print("regla falsa:", regla_falsa(f, 0, 1))
    print("secante:    ", secante(f, 0, 1))
    print("newton:     ", newton(f, df, 0.5))
    print("iteraciones de biseccion para error absoluto < 1e-4:", iteraciones_biseccion(0, 1, 1e-4))

    print("\n--- Conclusiones: Ej. 1 en [0, 1] ---")
    analizar(f, df, 0, 1)

    print("\n--- Conclusiones: que cambia con mas precision ---")
    conclusion_tolerancias(f, df, 0, 1)

    print("\n--- Conclusiones: Ej. 5c en [0, 1], donde los metodos abiertos se escapan ---")
    analizar(lambda x: 230 * x**4 + 18 * x**3 + 9 * x**2 - 221 * x - 9,
             lambda x: 920 * x**3 + 54 * x**2 + 18 * x - 221, 0, 1, 1e-5)

    print("\n--- Conclusiones: (x - 1)^2 = 0, raiz doble ---")
    analizar(lambda x: (x - 1) ** 2, lambda x: 2 * (x - 1), 0, 3, 1e-6)

    print("\n--- Sucesiones ---")
    print("aritmetica a1=2, d=3: a_10 =", aritmetica_termino(2, 3, 10), " S_10 =", aritmetica_suma(2, 3, 10))
    print("geometrica a1=2, r=3: a_5 =", geometrica_termino(2, 3, 5), " S_5 =", geometrica_suma(2, 3, 5))
    print("(1 + 1/n)^n -> e:", [round((1 + 1 / n) ** n, 6) for n in (1, 10, 100, 10000)])

    for nombre, a in [("(1 + 1/n)^n", lambda n: (1 + 1 / n) ** n), ("ln n", math.log), ("(-1)^n", lambda n: (-1) ** n),
                      ("1.1^n", lambda n: 1.1**n)]:
        print(f"\nSucesion {nombre}")
        conclusion_sucesion(a)

    print("\n--- Series ---")
    print("geometrica 1 + 1/2 + 1/4 + ... =", geometrica_infinita(1, 0.5))
    print("1/k^2 con 1000 terminos =", suma_parcial(lambda k: 1 / k**2, 1000), " (limite: pi^2/6 =", math.pi**2 / 6, ")")
    print("armonica con 1000 terminos =", suma_parcial(lambda k: 1 / k, 1000), " (diverge, crece muy lento)")
    print("alternante 1 - 1/2 + 1/3 - ... =", serie_alternante(lambda k: 1 / k), " (limite: ln 2 =", math.log(2), ")")

    print("\n--- Criterios ---")
    print("cociente de 1/k!  (tiende a 0, converge):", cociente(lambda k: 1 / math.factorial(k)))
    print("cociente de 1/k   (tiende a 1, no decide):", cociente(lambda k: 1 / k))
    print("raiz de (1/2)^k   (vale 1/2, converge):", raiz_enesima(lambda k: 0.5**k))

    for nombre, a in [("1/k!", lambda k: 1 / math.factorial(k)), ("2^k/k!", lambda k: 2**k / math.factorial(k)),
                      ("k!/2^k", lambda k: math.factorial(k) / 2**k), ("1/k^2", lambda k: 1 / k**2), ("1/k", lambda k: 1 / k),
                      ("1/(k ln k)", lambda k: 1 / (k * math.log(k + 1))), ("(-1)^k/k", lambda k: (-1) ** k / k),
                      ("(-1)^k/k^2", lambda k: (-1) ** k / k**2), ("k/(k+1)", lambda k: k / (k + 1)), ("(-1)^k", lambda k: (-1) ** k)]:
        print(f"\nSerie de {nombre}")
        conclusion_serie(a)

    print("\n--- Taylor con x = 1, tol = 1e-4: (valor, terminos) ---")
    print("e^1:   ", exp_taylor(1), " real:", math.e)
    print("sen 1: ", sen_taylor(1), " real:", math.sin(1))
    print("cos 1: ", cos_taylor(1), " real:", math.cos(1))
    print("ln 1.5:", ln1mas_taylor(0.5), " real:", math.log(1.5))

    print("\n--- Conclusion: Taylor de e^1 ---")
    valor, n = exp_taylor(1)
    conclusion_aproximacion(math.e, valor, 1e-4, n)
