import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# --- 2. Algoritmo Principal ---------------------------------------------------

def ecualizacion_local_histograma(img, M, N, border_type=cv2.BORDER_REPLICATE):
    """
    Implementa la ecualización local del histograma moviendo una ventana 
    de tamaño MxN pixel a pixel.
    M: Alto de la ventana (filas).
    N: Ancho de la ventana (columnas).
    border_type: Técnica de llenado de bordes.
    """
    # Inicializamos la imagen de salida con ceros
    img_out = np.zeros_like(img)
    
    # Calculamos la cantidad de pixels a agregar en los bordes
    pad_top = M // 2
    pad_bottom = M // 2
    pad_left = N // 2
    pad_right = N // 2
    
    # Agregamos bordes con la técnica seleccionada
    img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, border_type)
    
    filas, columnas = img.shape
    
    # Desplazamos la ventana pixel a pixel por toda la imagen
    for i in range(filas):
        for j in range(columnas):
            # Extraemos el vecindario local
            ventana = img_padded[i : i + M, j : j + N]
            
            # Calculamos la transformación local ecualizando el histograma de la ventana
            ventana_eq = cv2.equalizeHist(ventana)
            
            # Mapeamos únicamente el nivel de intensidad del pixel centrado
            img_out[i, j] = ventana_eq[pad_top, pad_left]
            
    return img_out

# --- 3. Ejecución y Exploración de Resultados ---------------------------------

if __name__ == '__main__':
    # Cargamos la imagen original en escala de grises
    img = cv2.imread('./imagenes/iniciales/Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)
    
    # 3.1 Visualización de la imagen original y su histograma
    imshow(img, title="Imagen Original", blocking=False)
    plt.figure()
    plt.hist(img.flatten(), 256, [0, 256])
    plt.title('Histograma Original')
    plt.show(block=False)
    
    # 3.2 Ecualización Global (para comparación)
    # Como menciona el TP, una ecualización global pierde la localidad del análisis
    img_eq_global = cv2.equalizeHist(img)
    imshow(img_eq_global, title="Ecualización Global", blocking=True)
    
    # Definimos las técnicas de bordes a evaluar
    tecnicas_bordes = {
        "REPLICATE": cv2.BORDER_REPLICATE,
        "REFLECT": cv2.BORDER_REFLECT,
        "CONSTANT": cv2.BORDER_CONSTANT
    }
    
    ventanas_cuadradas = [(3, 3), (5, 5), (7, 7), (11, 11), (15, 15), (31, 31)]
    ventanas_rectangulares = [(3, 5), (5, 3), (3, 15), (15, 3), (5, 31), (31, 5)]

    # Exploramos cada técnica de borde
    for nombre_borde, tipo_borde in tecnicas_bordes.items():
        print(f"\n--- Evaluando técnica de borde: {nombre_borde} ---")
        
        # 3.3 Ecualización Local - Exploración de vecindarios cuadrados
        plt.figure(figsize=(14, 8))
        plt.suptitle(f"Vecindarios Cuadrados - Borde: {nombre_borde}", fontsize=16)
        
        ax1_cuad = None
        for i, (M, N) in enumerate(ventanas_cuadradas):
            print(f"Procesando vecindario cuadrado de {M}x{N}...")
            img_eq_local = ecualizacion_local_histograma(img, M, N, border_type=tipo_borde)
            
            # Acomodamos en una grilla de 2 filas por 3 columnas
            if i == 0:
                ax1_cuad = plt.subplot(2, 3, i + 1)
            else:
                plt.subplot(2, 3, i + 1, sharex=ax1_cuad, sharey=ax1_cuad)
                
            imshow(img_eq_local, new_fig=False, title=f"Cuadrado ({M}x{N})", blocking=False)
        
        plt.tight_layout()
        plt.show(block=True)
            
        # 3.4 Ecualización Local - Exploración de vecindarios rectangulares
        plt.figure(figsize=(14, 8))
        plt.suptitle(f"Vecindarios Rectangulares - Borde: {nombre_borde}", fontsize=16)
        
        ax1_rect = None
        for i, (M_rect, N_rect) in enumerate(ventanas_rectangulares):
            print(f"Procesando vecindario rectangular de {M_rect}x{N_rect}...")
            img_eq_rect = ecualizacion_local_histograma(img, M_rect, N_rect, border_type=tipo_borde)
            
            # Acomodamos en una grilla de 2 filas por 3 columnas
            if i == 0:
                ax1_rect = plt.subplot(2, 3, i + 1)
            else:
                plt.subplot(2, 3, i + 1, sharex=ax1_rect, sharey=ax1_rect)
                
            imshow(img_eq_rect, new_fig=False, title=f"Rectangular ({M_rect}x{N_rect})", blocking=False)
        
        plt.tight_layout()
        plt.show(block=True)
        
    # Bloquea la ejecución al final para mantener todas las figuras abiertas
    plt.show()