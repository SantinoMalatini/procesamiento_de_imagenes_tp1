import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- Respuestas correctas -------------------------------------------------
RESPUESTAS_CORRECTAS = ['C', 'B', 'A', 'D', 'B', 'B', 'A', 'B', 'D', 'D']


# --- 1. Funciones Auxiliares ----------------------------------------------

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


def tramos(v):
    """Pares (inicio, fin) de cada tramo True de un vector booleano."""
    d = np.diff(np.concatenate([[0], v.astype(int), [0]]))
    return np.column_stack((np.where(d == 1)[0], np.where(d == -1)[0] - 1))


# --- 2. Detección de la estructura ---------------------------------------

def detectar_estructura(img):
    """Devuelve la binarización, proyecciones, líneas, celdas y campos."""
    binaria = (img < 150).astype(np.uint8)

    # Proyección sobre columnas: picos = líneas verticales
    proy_col = binaria.sum(axis=0)
    lineas_v = tramos(proy_col > 0.5 * proy_col.max())

    # La tabla arranca después de la primera línea vertical
    x_ref = lineas_v[0][1]
    y_tabla = max(tramos(binaria[:, x_ref]), key=lambda t: t[1] - t[0])[0]

    # Proyección sobre filas desde la tabla: picos = líneas horizontales
    proy_fil = binaria[y_tabla:, :].sum(axis=1)
    lineas_h = tramos(proy_fil > 0.5 * proy_fil.max()) + y_tabla

    # Armo las 10 celdas (2 columnas x 5 filas)
    celdas = []
    for c in range(2):
        x0, x1 = lineas_v[2 * c][1] + 1, lineas_v[2 * c + 1][0]
        for f in range(5):
            y0, y1 = lineas_h[f][1] + 1, lineas_h[f + 1][0]
            celdas.append((y0, y1, x0, x1))

    # Encabezado: la línea horizontal más larga en la zona superior
    enc = binaria[:y_tabla, :]
    y_linea = int(np.argmax(enc.sum(axis=1)))
    campos_v = tramos(enc[y_linea, :] > 0)
    campos = dict(zip(['Name', 'Date', 'Class'],
                      [t for t in campos_v if (t[1] - t[0]) > 20]))

    return binaria, proy_col, lineas_v, proy_fil, lineas_h, celdas, y_linea, campos


# --- 3. Corrección de una celda ------------------------------------------

def corregir_celda(celda):
    """Devuelve (letra, renglon, letra_recortada, letra_binaria)."""
    binaria = (celda < 150).astype(np.uint8)

    # Busco el subrayado: tramo horizontal más largo
    sub = None
    for y in range(binaria.shape[0]):
        largos = [t for t in tramos(binaria[y]) if (t[1] - t[0]) >= 20]
        if largos:
            sub = (y, largos[0][0], largos[0][1]); break
    if sub is None:
        return '-', None, None, None
    y_sub, x0, x1 = sub

    # Recorto la zona justo encima del subrayado
    renglon = celda[max(0, y_sub - 14):y_sub - 1, x0:x1 + 1]

    # Componentes conectadas: descarto las de área pequeña
    _, _, stats, _ = cv2.connectedComponentsWithStats((renglon < 200).astype(np.uint8), 8, cv2.CV_32S)
    validas = [stats[i] for i in range(1, len(stats)) if stats[i, cv2.CC_STAT_AREA] > 4]

    if len(validas) == 0: return '-', renglon, None, None
    if len(validas) > 1:  return 'MULTIPLE', renglon, None, None

    x, y, w, h, _ = validas[0]
    letra = renglon[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2]
    if letra.size == 0: return '-', renglon, None, None

    # Clasificación por cantidad y tamaño de agujeros internos
    letra_bin = (cv2.resize(letra, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC) < 200).astype(np.uint8) * 255
    contornos, jer = cv2.findContours(letra_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if jer is None: return '-', renglon, letra, letra_bin
    jer = jer[0]

    externos = [i for i in range(len(contornos)) if jer[i][3] == -1]
    if not externos: return '-', renglon, letra, letra_bin
    principal = max(externos, key=lambda i: cv2.contourArea(contornos[i]))
    agujeros = [i for i in range(len(contornos)) if jer[i][3] == principal]

    if len(agujeros) == 0:      r = 'C'        # sin agujeros
    elif len(agujeros) >= 2:    r = 'B'        # dos agujeros
    else:
        ratio = cv2.contourArea(contornos[agujeros[0]]) / cv2.contourArea(contornos[principal])
        r = 'D' if ratio > 0.3 else 'A'        # agujero grande / chico

    return r, renglon, letra, letra_bin


# --- 4. Validación del encabezado ----------------------------------------

def validar_campos(img, y_linea, campos):
    """Valida Name, Date y Class según las restricciones del enunciado."""
    estados, recortes, recorte_name = {}, {}, None

    for campo, (x0, x1) in campos.items():
        renglon = img[0:y_linea - 1, x0:x1 + 1]
        recortes[campo] = renglon

        _, _, stats, _ = cv2.connectedComponentsWithStats((renglon < 200).astype(np.uint8), 8, cv2.CV_32S)
        letras = [stats[i] for i in range(1, len(stats)) if stats[i, cv2.CC_STAT_AREA] > 4]
        letras.sort(key=lambda s: s[0])

        n = len(letras)
        if n == 0:
            estados[campo] = False
            continue

        # Cuento palabras por los huecos grandes entre letras consecutivas
        alto = np.median([s[3] for s in letras])
        palabras = 1 + sum(1 for i in range(1, n)
                           if (letras[i][0] - (letras[i - 1][0] + letras[i - 1][2])) > 0.45 * alto)

        if campo == 'Name':
            estados[campo] = (palabras >= 2) and (n <= 25)
            recorte_name = img[0:y_linea + 3, x0:x1 + 1]
        elif campo == 'Date':
            estados[campo] = (n == 8) and (palabras == 1)
        elif campo == 'Class':
            estados[campo] = (n == 1)

    return estados, recortes, recorte_name


# --- 5. Informe final -----------------------------------------------------

def generar_informe(recortes, aprobados):
    """Apila los crops del campo Name con su estado."""
    filas = []
    for recorte, apro in zip(recortes, aprobados):
        color = (0, 170, 0) if apro else (0, 0, 220)
        txt = 'APROBADO' if apro else 'DESAPROBADO'

        fila = cv2.cvtColor(cv2.resize(recorte, None, fx=2, fy=2), cv2.COLOR_GRAY2BGR)
        fila = cv2.copyMakeBorder(fila, 0, 0, 0, 260, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(fila, txt, (fila.shape[1] - 245, fila.shape[0] // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        fila = cv2.copyMakeBorder(fila, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=color)
        fila = cv2.copyMakeBorder(fila, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        filas.append(fila)

    w = max(f.shape[1] for f in filas)
    filas = [cv2.copyMakeBorder(f, 0, 0, 0, w - f.shape[1], cv2.BORDER_CONSTANT,
                                value=(255, 255, 255)) for f in filas]
    return np.vstack(filas)


# --- 6. Ejecución Principal ----------------------------------------------

if __name__ == '__main__':
    rutas = [f'./Imagenes/Iniciales/examen_{i}.png' for i in range(1, 6)]

    recortes, aprobados = [], []

    for idx, ruta in enumerate(rutas):
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        print(f"\n===== examen_{idx+1}.png =====")

        # --- PASO 1: estructura de la tabla -------------------------
        binaria, proy_col, lineas_v, proy_fil, lineas_h, celdas, y_linea, campos = detectar_estructura(img)

        plt.figure(figsize=(11, 7))
        plt.suptitle(f'examen_{idx+1} — Paso 1: Detección de la estructura', fontsize=12)

        plt.subplot(2, 2, 1)
        imshow(img, new_fig=False, title='1) Imagen original')

        plt.subplot(2, 2, 2)
        imshow(binaria * 255, new_fig=False, title='2) Imagen binarizada (img < 150)')

        plt.subplot(2, 2, 3)
        plt.plot(proy_col); plt.title('3) Proyección sobre columnas')
        plt.xticks([]); plt.yticks([])

        plt.subplot(2, 2, 4)
        plt.plot(proy_fil); plt.title('4) Proyección sobre filas')
        plt.xticks([]); plt.yticks([])

        plt.tight_layout()
        plt.show(block=True)

        # --- PASO 2: líneas y celdas --------------------------------
        plt.figure(figsize=(15, 6))
        plt.suptitle(f'examen_{idx+1} — Paso 2: Líneas y celdas detectadas', fontsize=12)

        ax1 = plt.subplot(1, 3, 1)
        imshow(img, new_fig=False, title='1) Líneas verticales')
        for v in lineas_v:
            plt.axvline(v[0], color='r', lw=1)
            plt.axvline(v[1], color='r', lw=1)

        plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
        imshow(img, new_fig=False, title='2) Líneas horizontales')
        for h in lineas_h:
            plt.axhline(h[0], color='g', lw=1)
            plt.axhline(h[1], color='g', lw=1)

        plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
        imshow(img, new_fig=False, title='3) Celdas (amarillo) y campos (azul)')
        ax = plt.gca()
        for (y0, y1, x0, x1) in celdas:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='y', lw=1.5))
        for (x0, x1) in campos.values():
            ax.add_patch(Rectangle((x0, y_linea - 15), x1 - x0, 15, fill=False, edgecolor='b', lw=1.5))

        plt.tight_layout()
        plt.show(block=True)

        # --- PASO 3: validación del encabezado ----------------------
        estados, recortes_campos, recorte_name = validar_campos(img, y_linea, campos)

        plt.figure(figsize=(15, 4))
        plt.suptitle(f'examen_{idx+1} — Paso 3: Validación del encabezado', fontsize=12)

        ax1 = plt.subplot(1, 4, 1)
        imshow(img[:y_linea + 3, :], new_fig=False, title='Encabezado completo')
        for (x0, x1) in campos.values():
            ax1.add_patch(Rectangle((x0, 0), x1 - x0, y_linea + 3, fill=False, edgecolor='r', lw=1.5))

        for i, (campo, rec) in enumerate(recortes_campos.items()):
            estado = 'OK' if estados[campo] else 'MAL'
            plt.subplot(1, 4, i + 2)
            imshow(rec, new_fig=False, title=f'{campo}: {estado}')

        plt.tight_layout()
        plt.show(block=True)

        # --- PASO 4: clasificación de las 10 celdas -----------------
        letras_detectadas, detalles_primera = [], None
        for i, (y0, y1, x0, x1) in enumerate(celdas):
            letra, renglon, letra_r, letra_b = corregir_celda(img[y0:y1, x0:x1])
            letras_detectadas.append(letra)
            if i == 0:
                detalles_primera = (img[y0:y1, x0:x1], renglon, letra_r, letra_b, letra)

        plt.figure(figsize=(15, 6))
        plt.suptitle(f'examen_{idx+1} — Paso 4: Clasificación de las 10 celdas', fontsize=12)
        for i, (y0, y1, x0, x1) in enumerate(celdas):
            r = letras_detectadas[i]
            esperada = RESPUESTAS_CORRECTAS[i]
            color = 'green' if r == esperada else 'red'
            ax = plt.subplot(2, 5, i + 1)
            imshow(img[y0:y1, x0:x1], new_fig=False, title=f'P{i+1}: {r} (esp. {esperada})')
            ax.title.set_color(color)
        plt.tight_layout()
        plt.show(block=True)

        # --- PASO 5: proceso interno de una celda -------------------
        celda0, renglon0, letra0, letra_bin0, r0 = detalles_primera
        plt.figure(figsize=(15, 4))
        plt.suptitle(f'examen_{idx+1} — Paso 5: Proceso interno de clasificación (celda 1)', fontsize=12)

        plt.subplot(1, 4, 1)
        imshow(celda0, new_fig=False, title='1) Celda recortada')

        plt.subplot(1, 4, 2)
        if renglon0 is not None:
            imshow(renglon0, new_fig=False, title='2) Renglón (zona sobre el subrayado)')

        plt.subplot(1, 4, 3)
        if letra0 is not None:
            imshow(letra0, new_fig=False, title='3) Letra aislada')

        plt.subplot(1, 4, 4)
        if letra_bin0 is not None:
            imshow(letra_bin0, new_fig=False, title=f'4) Clasificación: {r0}')

        plt.tight_layout()
        plt.show(block=True)

        # --- PASO 6: resultado del examen --------------------------
        resultados = [letras_detectadas[i] == RESPUESTAS_CORRECTAS[i] for i in range(10)]
        aprobado = sum(resultados) >= 6

        for i, ok in enumerate(resultados):
            print(f"Pregunta {i+1:02d}: {'OK' if ok else 'MAL'} (detectado: {letras_detectadas[i]})")
        for campo, valido in estados.items():
            print(f"{campo}: {'OK' if valido else 'MAL'}")
        print(f"Correctas: {sum(resultados)}/10 -> {'APROBADO' if aprobado else 'DESAPROBADO'}")

        plt.figure(figsize=(9, 11))
        estado_txt = 'APROBADO' if aprobado else 'DESAPROBADO'
        color_txt = 'green' if aprobado else 'red'
        plt.suptitle(f'examen_{idx+1} — Resultado: {estado_txt}  ({sum(resultados)}/10)', color=color_txt, fontsize=13)
        ax = plt.gca()
        imshow(img, new_fig=False, title=None)
        for i, (y0, y1, x0, x1) in enumerate(celdas):
            ok = resultados[i]
            color = 'lime' if ok else 'red'
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, lw=2))
            ax.text(x1 + 4, (y0 + y1) / 2, f'{i+1}: {letras_detectadas[i]} {"OK" if ok else "X"}',
                    color=color, fontsize=9, va='center')
        plt.tight_layout()
        plt.show(block=True)

        # Guardo datos para el informe final
        recortes.append(recorte_name)
        aprobados.append(aprobado)

    # --- Informe final -----------------------------------------------
    if recortes:
        informe = generar_informe(recortes, aprobados)
        plt.figure(figsize=(9, 2 * len(recortes)))
        plt.imshow(cv2.cvtColor(informe, cv2.COLOR_BGR2RGB))
        plt.title('Informe final — aprobados y desaprobados')
        plt.axis('off')
        plt.tight_layout()
        plt.show(block=True)