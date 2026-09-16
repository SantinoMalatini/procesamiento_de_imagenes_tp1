import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Clave de respuestas correctas (pregunta 1 a 10)
RESPUESTAS_CORRECTAS = ['C', 'B', 'A', 'D', 'B', 'B', 'A', 'B', 'D', 'D']

# Umbrales generales
TH_LINEAS = 150          # umbral de gris para detectar líneas de la grilla y subrayados
TH_TRAZOS = 200          # umbral más permisivo para letras con antialiasing
TH_AREA = 4              # área mínima de una componente válida (descarta restos de líneas)
LARGO_MIN_SUBRAYADO = 20 # largo mínimo (en pixels) de un subrayado
ALTO_RENGLON = 14        # alto de la franja a analizar por encima de un subrayado
ESCALA_LETRA = 4         # factor de escalado de la letra antes de binarizar
TH_RATIO_A_D = 0.30      # ratio área_agujero/área_total que separa la A de la D
MIN_APROBACION = 6       # cantidad mínima de respuestas correctas para aprobar
# Hueco mínimo entre palabras relativo al alto de los caracteres (~12 px).
# Entre palabras hay 6-7 px, pero entre dígitos angostos como "11" hay 5 px.
FACTOR_ESPACIO = 0.45

# --- 1. Funciones Auxiliares --------------------------------------------------

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    if title is not None:
        plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


def detectar_lineas(proyeccion, umbral):
    """
    Recibe una proyección (suma de pixels por fila o por columna) y devuelve
    una lista de tramos (inicio, fin) donde la proyección supera el umbral.
    Como las líneas pueden tener más de un pixel de ancho, se agrupan los
    índices consecutivos en un único tramo.
    """
    mascara = (proyeccion > umbral).astype(int)
    # Los cambios 0->1 marcan inicios y los cambios 1->0 marcan finales
    cambios = np.diff(np.concatenate([[0], mascara, [0]]))
    inicios = np.where(cambios == 1)[0]
    finales = np.where(cambios == -1)[0] - 1
    return list(zip(inicios, finales))


def tramos_horizontales(fila_binaria, largo_min):
    """
    Devuelve los tramos (x0, x1) de pixels oscuros consecutivos de una fila
    binaria cuyo largo sea al menos largo_min (candidatos a subrayado).
    """
    return [(x0, x1) for x0, x1 in detectar_lineas(fila_binaria, 0) if x1 - x0 + 1 >= largo_min]


def componentes_validas(recorte, umbral_gris=TH_TRAZOS, th_area=TH_AREA):
    """
    Umbraliza el recorte y obtiene sus componentes conectadas, descartando
    las de área muy chica. Devuelve los stats (x, y, w, h, area) ordenados
    de izquierda a derecha.
    """
    recorte_th = (recorte < umbral_gris).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(recorte_th, 8, cv2.CV_32S)
    stats = stats[1:]                     # descartamos el fondo
    ix_area = stats[:, -1] > th_area
    stats = stats[ix_area, :]
    return stats[np.argsort(stats[:, 0])]

# --- 2. Detección de la estructura del examen ---------------------------------

def detectar_grilla(img):
    """
    Detecta las 10 celdas de preguntas mediante proyección de pixels.
    Devuelve la lista de celdas (y0, y1, x0, x1) ordenadas de la pregunta 1
    a la 10 (primero columna izquierda, luego derecha) y la fila donde
    comienza la tabla.
    """
    img_th_ones = (img < TH_LINEAS).astype(np.uint8)

    # Líneas verticales: columnas con muchos pixels oscuros
    img_cols = np.sum(img_th_ones, 0)
    lineas_v = detectar_lineas(img_cols, 0.5 * img_cols.max())

    # La tabla empieza donde arranca el tramo oscuro más largo de la primera vertical
    col_ref = lineas_v[0][1]
    tramos_col = detectar_lineas(img_th_ones[:, col_ref], 0)
    y_tabla = max(tramos_col, key=lambda t: t[1] - t[0])[0]

    # Líneas horizontales: se proyecta sólo la zona de la tabla (sin el encabezado)
    img_rows = np.sum(img_th_ones[y_tabla:, :], 1)
    lineas_h = [(y0 + y_tabla, y1 + y_tabla) for y0, y1 in detectar_lineas(img_rows, 0.5 * img_rows.max())]

    celdas = []
    for c in range(2):
        # Interior de la columna: entre el final de su línea izquierda y el inicio de la derecha
        x0 = lineas_v[2 * c][1] + 1
        x1 = lineas_v[2 * c + 1][0]
        for f in range(5):
            y0 = lineas_h[f][1] + 1
            y1 = lineas_h[f + 1][0]
            celdas.append((y0, y1, x0, x1))
    return celdas, y_tabla


def detectar_campos_encabezado(img, y_tabla):
    """
    Detecta los campos Name, Date y Class del encabezado. En la zona por
    encima de la tabla, la fila con mayor proyección es la de los subrayados,
    y cada tramo largo de esa fila corresponde a un campo.
    Devuelve la fila del subrayado y un diccionario campo -> (x0, x1).
    """
    encabezado_th = (img[:y_tabla, :] < TH_LINEAS).astype(np.uint8)
    img_rows = np.sum(encabezado_th, 1)
    y_linea = int(np.argmax(img_rows))

    tramos = tramos_horizontales(encabezado_th[y_linea], LARGO_MIN_SUBRAYADO)
    campos = dict(zip(['Name', 'Date', 'Class'], tramos))
    return y_linea, campos


def recortar_sobre_subrayado(img, y_linea, x0, x1, alto=ALTO_RENGLON):
    """
    Recorta la franja de alto 'alto' ubicada justo encima de un subrayado,
    sin incluir la línea del subrayado.
    """
    return img[max(0, y_linea - alto):y_linea - 1, x0:x1 + 1]

# --- 3. Corrección de respuestas (punto a) ------------------------------------

def detectar_respuesta(celda):
    """
    Busca el espacio en blanco (subrayado) de la celda y analiza las
    componentes conectadas escritas sobre él.
    Devuelve None si no hay respuesta, 'MULTIPLE' si hay más de una marca,
    o el recorte de la letra si hay exactamente una.
    """
    celda_th = (celda < TH_LINEAS).astype(np.uint8)

    # El primer tramo horizontal largo de la celda es el subrayado del enunciado
    subrayado = None
    for y in range(celda_th.shape[0]):
        tramos = tramos_horizontales(celda_th[y], LARGO_MIN_SUBRAYADO)
        if tramos:
            subrayado = (y, tramos[0][0], tramos[0][1])
            break
    if subrayado is None:
        return None

    y, x0, x1 = subrayado
    renglon = recortar_sobre_subrayado(celda, y, x0, x1)
    stats = componentes_validas(renglon)

    if len(stats) == 0:
        return None
    if len(stats) > 1:
        return 'MULTIPLE'

    # Recorte de la única letra con un pequeño margen
    x, yy, w, h, _ = stats[0]
    margen = 2
    return renglon[max(0, yy - margen):yy + h + margen, max(0, x - margen):x + w + margen]


def clasificar_letra(recorte):
    """
    Identifica la letra (A, B, C o D) según la cantidad de agujeros:
    C = 0, B = 2, A y D = 1 (se distinguen por el ratio área_agujero/área_total,
    que es mayor en la D).
    Como las letras son muy chicas y con antialiasing, primero se escala el
    recorte y se binariza con un umbral permisivo.
    """
    recorte_grande = cv2.resize(recorte, None, fx=ESCALA_LETRA, fy=ESCALA_LETRA,
                                interpolation=cv2.INTER_CUBIC)
    letra_bin = (recorte_grande < TH_TRAZOS).astype(np.uint8) * 255

    contornos, jerarquia = cv2.findContours(letra_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    jerarquia = jerarquia[0]

    # Contorno externo principal (sin padre) y sus agujeros (hijos)
    externos = [i for i in range(len(contornos)) if jerarquia[i][3] == -1]
    principal = max(externos, key=lambda i: cv2.contourArea(contornos[i]))
    agujeros = [i for i in range(len(contornos)) if jerarquia[i][3] == principal]

    if len(agujeros) == 0:
        return 'C'
    if len(agujeros) >= 2:
        return 'B'

    area_agujero = cv2.contourArea(contornos[agujeros[0]])
    area_total = cv2.contourArea(contornos[principal])
    ratio = area_agujero / area_total
    return 'D' if ratio > TH_RATIO_A_D else 'A'


def corregir_examen(img):
    """
    Corrige las 10 preguntas del examen. Devuelve una lista de booleanos
    (True = correcta) y la lista de respuestas detectadas.
    """
    celdas, _ = detectar_grilla(img)
    resultados = []
    respuestas = []
    for i, (y0, y1, x0, x1) in enumerate(celdas):
        respuesta = detectar_respuesta(img[y0:y1, x0:x1])
        if respuesta is None:
            letra = '-'             # sin respuesta
        elif isinstance(respuesta, str):
            letra = respuesta       # múltiple marcada
        else:
            letra = clasificar_letra(respuesta)
        respuestas.append(letra)
        resultados.append(letra == RESPUESTAS_CORRECTAS[i])
    return resultados, respuestas

# --- 4. Validación del encabezado (punto b) -----------------------------------

def contar_caracteres_y_palabras(recorte):
    """
    Cuenta los caracteres de un campo como componentes conectadas, y las
    palabras como grupos de caracteres separados por un espacio (un hueco
    horizontal mayor a FACTOR_ESPACIO veces el alto típico de los caracteres).
    """
    stats = componentes_validas(recorte)
    n_caracteres = len(stats)
    if n_caracteres == 0:
        return 0, 0

    alto_tipico = np.median(stats[:, 3])
    fin_anterior = stats[:-1, 0] + stats[:-1, 2]
    huecos = stats[1:, 0] - fin_anterior
    n_palabras = 1 + int(np.sum(huecos > FACTOR_ESPACIO * alto_tipico))
    return n_caracteres, n_palabras


def validar_encabezado(img):
    """
    Valida los campos del encabezado:
      Name: al menos dos palabras y no más de 25 caracteres.
      Date: 8 caracteres formando una sola palabra.
      Class: un único caracter.
    Devuelve un diccionario campo -> bool y el recorte del campo Name.
    """
    _, y_tabla = detectar_grilla(img)
    y_linea, campos = detectar_campos_encabezado(img, y_tabla)

    estados = {}
    for campo, (x0, x1) in campos.items():
        recorte = recortar_sobre_subrayado(img, y_linea, x0, x1, alto=y_linea)
        n_car, n_pal = contar_caracteres_y_palabras(recorte)
        if campo == 'Name':
            estados[campo] = n_pal >= 2 and n_car <= 25
        elif campo == 'Date':
            estados[campo] = n_car == 8 and n_pal == 1
        else:
            estados[campo] = n_car == 1

    x0, x1 = campos['Name']
    recorte_name = img[0:y_linea + 3, x0:x1 + 1]
    return estados, recorte_name

# --- 5. Informe de aprobados (punto d) ----------------------------------------

def generar_informe(recortes_name, aprobados, ruta_salida):
    """
    Genera una única imagen con los recortes del campo Name de todos los
    exámenes, con borde verde y leyenda APROBADO para los aprobados, y borde
    rojo y leyenda DESAPROBADO para los desaprobados.
    """
    escala = 2
    ancho_leyenda = 260
    filas = []
    for recorte, aprobado in zip(recortes_name, aprobados):
        color = (0, 170, 0) if aprobado else (0, 0, 220)   # BGR
        leyenda = 'APROBADO' if aprobado else 'DESAPROBADO'

        name = cv2.resize(recorte, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
        name = cv2.cvtColor(name, cv2.COLOR_GRAY2BGR)

        # Espacio a la derecha para la leyenda
        name = cv2.copyMakeBorder(name, 0, 0, 0, ancho_leyenda, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(name, leyenda, (name.shape[1] - ancho_leyenda + 15, name.shape[0] // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # Borde de color que identifica el estado y separación blanca entre filas
        name = cv2.copyMakeBorder(name, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=color)
        name = cv2.copyMakeBorder(name, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        filas.append(name)

    # Igualamos anchos antes de apilar verticalmente
    ancho_max = max(f.shape[1] for f in filas)
    filas = [cv2.copyMakeBorder(f, 0, 0, 0, ancho_max - f.shape[1], cv2.BORDER_CONSTANT,
                                value=(255, 255, 255)) for f in filas]
    informe = np.vstack(filas)

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    cv2.imwrite(ruta_salida, informe)
    return informe

# --- 6. Ejecución y Resultados (punto c) --------------------------------------

if __name__ == '__main__':
    # Se ejecuta desde la raíz del repositorio
    rutas_examenes = [f'./Imagenes/Iniciales/examen_{i}.png' for i in range(1, 6)]
    ruta_informe = './Imagenes/Resultados/Problema_2/informe_aprobados.png'

    recortes_name = []
    aprobados = []

    for ruta in rutas_examenes:
        # Cargamos el examen en escala de grises (única entrada del algoritmo)
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        print(f"\n--- {os.path.basename(ruta)} ---")

        # 6.1 Corrección de las preguntas
        resultados, respuestas = corregir_examen(img)
        for i, correcta in enumerate(resultados):
            print(f"Pregunta {i + 1}: {'OK' if correcta else 'MAL'}")

        # 6.2 Validación del encabezado
        estados, recorte_name = validar_encabezado(img)
        for campo, valido in estados.items():
            print(f"{campo}: {'OK' if valido else 'MAL'}")

        n_correctas = sum(resultados)
        aprobado = n_correctas >= MIN_APROBACION
        print(f"Respuestas detectadas: {' '.join(respuestas)}")
        print(f"Correctas: {n_correctas}/10 -> {'APROBADO' if aprobado else 'DESAPROBADO'}")

        recortes_name.append(recorte_name)
        aprobados.append(aprobado)

    # 6.3 Imagen de salida con los Name de aprobados y desaprobados
    informe = generar_informe(recortes_name, aprobados, ruta_informe)
    imshow(cv2.cvtColor(informe, cv2.COLOR_BGR2RGB), title="Informe de aprobados",
           color_img=True, colorbar=False, blocking=True)
