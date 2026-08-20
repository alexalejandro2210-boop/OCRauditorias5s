%%writefile pdf_reader.py
"""
pdf_reader.py
=============

Puerta de entrada del pipeline: convierte un archivo PDF escaneado en
una secuencia de imagenes (arrays de NumPy, convencion de color BGR de
OpenCV), una por cada pagina, listas para que `image_processing.py` las
procese.

Este modulo NO sabe nada sobre tablas, celdas ni evaluaciones. Su unica
responsabilidad es la conversion PDF -> imagenes. Esto permite que, si
en el futuro cambiara la fuente de entrada (por ejemplo, fotos sueltas
en vez de un PDF escaneado), solo este modulo necesite reescribirse.

DEPENDENCIA DE SISTEMA
-----------------------
`pdf2image` es una envoltura de Python sobre el binario `poppler-utils`
(especificamente `pdftoppm`), que NO es un paquete de Python: es un
programa del sistema operativo. En Google Colab debe instalarse con:

    !apt-get install -y poppler-utils
    !pip install pdf2image

antes de poder usar este modulo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

from config import APLICAR_ROTACION_FIJA, RESOLUCION, ROTACION_FIJA_GRADOS
from utils import ErrorLecturaPDF, medir_tiempo

logger = logging.getLogger(__name__)


def _convertir_pil_a_bgr(imagen_pil: Image.Image) -> np.ndarray:
    """
    Convierte una imagen PIL (formato RGB, tal como la entrega
    `pdf2image`) a un array de NumPy en formato BGR (convencion que usa
    OpenCV en todo el resto del pipeline).

    Este es el unico lugar del proyecto donde ocurre esta conversion:
    resolverla aqui, en la frontera de entrada del sistema, evita que
    cada modulo posterior tenga que acordarse de hacerlo por su cuenta.

    Args:
        imagen_pil: Imagen en formato PIL, modo RGB.

    Returns:
        Array de NumPy de forma (alto, ancho, 3) en orden de canales BGR.
    """
    if imagen_pil.mode != "RGB":
        imagen_pil = imagen_pil.convert("RGB")

    arreglo_rgb = np.array(imagen_pil)
    arreglo_bgr = arreglo_rgb[:, :, ::-1]
    return arreglo_bgr


def _corregir_orientacion_escaneo(imagen_bgr: np.ndarray) -> np.ndarray:
    """
    Aplica la rotacion fija de escaneo definida en `config.py`.

    A diferencia de la correccion de pequena rotacion que hace
    `image_processing.py` (+/-5, para compensar una hoja mal alineada
    al escanear), esta funcion corrige una rotacion GRANDE y FIJA
    (90), producto de como el escaner/alimentador capturo fisicamente
    las hojas -- igual en todas las paginas del documento.

    Args:
        imagen_bgr: Imagen recien convertida a BGR, en la orientacion
            cruda tal como la entrego el escaner.

    Returns:
        Imagen rotada a la orientacion correcta si
        `config.APLICAR_ROTACION_FIJA` es True; si es False, devuelve
        la imagen sin modificar.
    """
    if not APLICAR_ROTACION_FIJA:
        return imagen_bgr

    mapa_rotaciones = {
        90: cv2.ROTATE_90_CLOCKWISE,
        -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
    }
    codigo_rotacion = mapa_rotaciones.get(ROTACION_FIJA_GRADOS)
    if codigo_rotacion is None:
        raise ErrorLecturaPDF(
            f"ROTACION_FIJA_GRADOS={ROTACION_FIJA_GRADOS} no es un valor "
            f"soportado. Use 90, -90 o 180."
        )
    return cv2.rotate(imagen_bgr, codigo_rotacion)


def obtener_numero_paginas(ruta_pdf: str) -> int:
    """
    Obtiene el numero total de paginas de un PDF sin cargar sus
    imagenes en memoria.

    Args:
        ruta_pdf: Ruta al archivo PDF.

    Returns:
        Numero total de paginas del documento.

    Raises:
        ErrorLecturaPDF: Si el archivo no existe o no es un PDF valido.
    """
    ruta = Path(ruta_pdf)
    if not ruta.is_file():
        raise ErrorLecturaPDF(f"El archivo PDF no existe: {ruta_pdf}")

    try:
        from pdf2image.pdf2image import pdfinfo_from_path

        info = pdfinfo_from_path(str(ruta))
        return int(info["Pages"])
    except Exception as error:
        raise ErrorLecturaPDF(
            f"No se pudo leer la informacion del PDF '{ruta_pdf}': {error}"
        ) from error


@medir_tiempo
def leer_paginas_pdf(
    ruta_pdf: str, dpi: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Generador que entrega, una a la vez, cada pagina del PDF convertida
    a imagen en formato NumPy/BGR.

    Args:
        ruta_pdf: Ruta al archivo PDF a procesar.
        dpi: Resolucion a la que se rasteriza cada pagina. Si es None,
            se usa `config.RESOLUCION.dpi_esperado` (300 por defecto).

    Yields:
        Tuplas (numero_pagina, imagen).

    Raises:
        ErrorLecturaPDF: Si el archivo no existe, esta corrupto, o
            `pdf2image`/`poppler` fallan al convertir alguna pagina.
    """
    ruta = Path(ruta_pdf)
    if not ruta.is_file():
        raise ErrorLecturaPDF(f"El archivo PDF no existe: {ruta_pdf}")

    dpi_a_usar = dpi if dpi is not None else RESOLUCION.dpi_esperado
    logger.info("Iniciando lectura de '%s' a %d dpi", ruta_pdf, dpi_a_usar)

    total_paginas = obtener_numero_paginas(str(ruta))
    if total_paginas == 0:
        raise ErrorLecturaPDF(f"El PDF '{ruta_pdf}' no contiene paginas.")

    logger.info("El PDF contiene %d pagina(s)", total_paginas)

    for numero_pagina in range(1, total_paginas + 1):
        try:
            paginas_convertidas = convert_from_path(
                str(ruta),
                dpi=dpi_a_usar,
                first_page=numero_pagina,
                last_page=numero_pagina,
            )
        except Exception as error:
            raise ErrorLecturaPDF(
                f"Fallo al convertir la pagina {numero_pagina} de "
                f"'{ruta_pdf}': {error}"
            ) from error

        if not paginas_convertidas:
            raise ErrorLecturaPDF(
                f"La pagina {numero_pagina} no produjo ninguna imagen."
            )

        imagen_bgr = _convertir_pil_a_bgr(paginas_convertidas[0])
        imagen_bgr = _corregir_orientacion_escaneo(imagen_bgr)
        logger.debug(
            "Pagina %d/%d convertida: %dx%d px",
            numero_pagina,
            total_paginas,
            imagen_bgr.shape[1],
            imagen_bgr.shape[0],
        )
        yield numero_pagina, imagen_bgr


if __name__ == "__main__":
    import sys

    from config import configurar_logging

    configurar_logging(logging.INFO)

    ruta_prueba = "test_muestra.pdf"
    if not Path(ruta_prueba).is_file():
        print(
            f"AVISO: no se encontro '{ruta_prueba}' en este directorio. "
            "Coloca un PDF de prueba con ese nombre para ejecutar esta "
            "prueba manual."
        )
        sys.exit(0)

    print(f"\nNumero de paginas detectadas: {obtener_numero_paginas(ruta_prueba)}")

    print("\n--- Leyendo paginas ---")
    for numero, imagen in leer_paginas_pdf(ruta_prueba):
        print(
            f"Pagina {numero}: forma={imagen.shape}, "
            f"dtype={imagen.dtype}, "
            f"valor min/max={imagen.min()}/{imagen.max()}"
        )

    print("\npdf_reader.py se ejecuto sin errores.")
