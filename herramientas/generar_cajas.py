"""
generar_cajas.py  -  Crea las CAJAS DE PRUEBA (ficticias) para practicar el OCR.

Para que sirve
--------------
Para probar la lectura OCR necesitas "una caja de celular" que puedas
fotografiar. Este programa dibuja la cara frontal de una caja por cada modelo
del catalogo y la guarda como imagen (PNG) y como PDF para imprimir.

Que dice cada caja (y por que)
------------------------------
  * Marca y modelo grandes     -> el OCR las usa para saber QUE celular es.
  * "CANTIDAD EN CAJA: N"      -> el OCR la usa para saber CUANTOS vienen.
  * SKU y codigo de barras     -> adorno realista (el codigo de barras NO se
                                  lee, es solo dibujo y su numero es inventado).
  * "CAJA DE PRUEBA - FICTICIA"-> aclara que no es un producto real.

Las cajas no tienen logos ni disenos de las marcas reales: solo texto.

Como usarlo
-----------
    python herramientas/generar_cajas.py
Crea la carpeta  cajas_de_prueba/  con un PNG por modelo y un PDF con las tres.

    python herramientas/generar_cajas.py --extra
Agrega 5 cajas extra (mismos modelos, otras cantidades) y cajas_de_prueba_extra.pdf,
sin tocar las 3 cajas originales.
"""
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- rutas
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "cajas_de_prueba")
IMAGENES = os.path.join(RAIZ, "static", "imagenes")
EJEMPLOS = os.path.join(RAIZ, "static", "ejemplos")


def _primera_existente(*rutas):
    """Devuelve la primera ruta de fuente que exista (Linux o Windows)."""
    return next((r for r in rutas if os.path.exists(r)), rutas[0])


FUENTE_NEGRITA = _primera_existente("/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
                                    "C:/Windows/Fonts/calibrib.ttf", "C:/Windows/Fonts/arialbd.ttf")
FUENTE_NORMAL = _primera_existente("/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
                                   "C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/arial.ttf")
# En Linux usa Carlito y en Windows Calibri (o Arial). Si no tienes ninguna,
# cambia las rutas anteriores por una fuente instalada en tu equipo.

# ---------------------------------------------------------------- datos de las cajas
# "cantidad" es lo que dice la etiqueta "CANTIDAD EN CAJA". Es el dato que el
# sistema suma al stock cuando la caja se ingresa.
CAJAS = [
    dict(archivo="caja_Galaxy_A15", marca="SAMSUNG", modelo="Galaxy A15",
         specs="128 GB  |  4 GB RAM  |  6,5 pulgadas", color="Color: Negro",
         cantidad=6, sku="SAM-A15-128-NEG", barras="0 780123 456781",
         acento=(20, 40, 120), imagen="Galaxy_A15_frente.png"),
    dict(archivo="caja_iPhone_13", marca="APPLE", modelo="iPhone 13",
         specs="128 GB  |  4 GB RAM  |  6,1 pulgadas", color="Color: Rosa",
         cantidad=3, sku="APL-IP13-128-ROS", barras="0 780123 456798",
         acento=(150, 40, 80), imagen="iPhone_13_frente.png"),
    dict(archivo="caja_Redmi_Note_13", marca="XIAOMI", modelo="Redmi Note 13",
         specs="128 GB  |  6 GB RAM  |  6,67 pulgadas", color="Color: Verde azulado",
         cantidad=5, sku="XIA-RN13-128-VAZ", barras="0 780123 456804",
         acento=(0, 110, 100), imagen="Redmi_Note_13_frente.png"),
]

# Cajas EXTRA de practica (opcion --extra): los mismos 3 modelos del catalogo pero con
# OTRAS cantidades y colores, para probar calculos distintos. Como el modelo esta en
# el catalogo, el OCR los reconoce. Cada una es una imagen distinta, asi que el
# sistema no las toma por duplicadas de las 3 cajas originales.
_BASE = {c["modelo"]: c for c in CAJAS}


def _variante(archivo, modelo, cantidad, color, sku, barras, acento):
    c = dict(_BASE[modelo])
    c.update(archivo=archivo, cantidad=cantidad, color=f"Color: {color}", sku=sku,
             barras=barras, acento=acento)
    return c


CAJAS_EXTRA = [
    _variante("caja_extra_1_Galaxy_A15_x10", "Galaxy A15", 10, "Azul claro",
              "SAM-A15-128-AZC", "0 780123 456811", (30, 90, 160)),
    _variante("caja_extra_2_iPhone_13_x2", "iPhone 13", 2, "Azul",
              "APL-IP13-128-AZU", "0 780123 456828", (20, 80, 150)),
    _variante("caja_extra_3_Redmi_Note_13_x8", "Redmi Note 13", 8, "Negro medianoche",
              "XIA-RN13-128-NEG", "0 780123 456835", (40, 40, 40)),
    _variante("caja_extra_4_Galaxy_A15_x12", "Galaxy A15", 12, "Blanco",
              "SAM-A15-128-BLA", "0 780123 456842", (110, 60, 140)),
    _variante("caja_extra_5_iPhone_13_x1", "iPhone 13", 1, "Negro",
              "APL-IP13-128-NEG", "0 780123 456859", (90, 90, 100)),
]

ANCHO, ALTO = 1200, 1700


def fuente(nombre, tam):
    return ImageFont.truetype(nombre, tam)


def centrar(dib, y, texto, f, color=(0, 0, 0)):
    """Escribe 'texto' centrado horizontalmente en la altura y."""
    ancho = dib.textlength(texto, font=f)
    dib.text(((ANCHO - ancho) / 2, y), texto, font=f, fill=color)


def codigo_de_barras(dib, x, y, ancho, alto, semilla):
    """Dibuja barras negras de grosor aleatorio (solo decoracion)."""
    azar = random.Random(semilla)
    pos = x
    while pos < x + ancho:
        grosor = azar.choice([3, 3, 6, 9])
        dib.rectangle((pos, y, min(pos + grosor, x + ancho), y + alto), fill=(0, 0, 0))
        pos += grosor + azar.choice([3, 6, 9])


def dibujar_caja(c):
    """Devuelve la imagen (PIL) de una caja."""
    img = Image.new("RGB", (ANCHO, ALTO), (246, 243, 236))      # carton claro
    d = ImageDraw.Draw(img)
    acento = c["acento"]

    # marco y franja superior de color
    d.rectangle((30, 30, ANCHO - 30, ALTO - 30), outline=acento, width=10)
    d.rectangle((40, 40, ANCHO - 40, 170), fill=acento)
    centrar(d, 70, "CAJA DE PRUEBA  -  FICTICIA", fuente(FUENTE_NEGRITA, 64), (255, 255, 255))

    # marca y modelo (texto grande y negro: lo mas facil de leer para el OCR)
    centrar(d, 230, c["marca"], fuente(FUENTE_NEGRITA, 120))
    centrar(d, 380, c["modelo"], fuente(FUENTE_NEGRITA, 170))

    # dibujo esquematico del celular (adorno)
    ruta = os.path.join(IMAGENES, c["imagen"])
    if os.path.exists(ruta):
        tel = Image.open(ruta).convert("RGBA")
        tel = tel.resize((int(tel.width * 560 / tel.height), 560))
        img.paste(tel, ((ANCHO - tel.width) // 2, 620), tel)

    # especificaciones
    centrar(d, 1210, c["specs"], fuente(FUENTE_NORMAL, 64))
    centrar(d, 1290, c["color"], fuente(FUENTE_NORMAL, 64))

    # etiqueta blanca con la cantidad (el dato que se calcula)
    d.rectangle((90, 1390, ANCHO - 90, 1500), fill=(255, 255, 255), outline=(0, 0, 0), width=4)
    unidad = "UNIDAD" if c["cantidad"] == 1 else "UNIDADES"
    centrar(d, 1419, f"CANTIDAD EN CAJA: {c['cantidad']} {unidad}", fuente(FUENTE_NEGRITA, 62))

    # SKU y codigo de barras
    d.text((100, 1540), f"SKU: {c['sku']}", font=fuente(FUENTE_NORMAL, 48), fill=(0, 0, 0))
    codigo_de_barras(d, 640, 1535, 440, 75, semilla=c["sku"])
    d.text((640, 1612), c["barras"], font=fuente(FUENTE_NORMAL, 32), fill=(0, 0, 0))
    return img


def main_extra():
    """Crea las 5 cajas extra y un PDF aparte. NO toca las 3 cajas originales ni su PDF."""
    os.makedirs(SALIDA, exist_ok=True)
    paginas = []
    for c in CAJAS_EXTRA:
        img = dibujar_caja(c)
        img.save(os.path.join(SALIDA, c["archivo"] + ".png"))
        paginas.append(img)
    paginas[0].save(os.path.join(SALIDA, "cajas_de_prueba_extra.pdf"), save_all=True,
                    append_images=paginas[1:], resolution=150.0)
    print("Cajas extra creadas en", SALIDA)
    for c in CAJAS_EXTRA:
        print(f"  {c['archivo']}.png  ->  {c['marca']} {c['modelo']} x {c['cantidad']}")


def main():
    if "--extra" in sys.argv[1:]:
        return main_extra()
    os.makedirs(SALIDA, exist_ok=True)
    os.makedirs(EJEMPLOS, exist_ok=True)
    paginas = []
    for c in CAJAS:
        img = dibujar_caja(c)
        img.save(os.path.join(SALIDA, c["archivo"] + ".png"))
        # copia para el boton "Usar caja de ejemplo" del panel web
        img.resize((ANCHO // 2, ALTO // 2), Image.LANCZOS).save(
            os.path.join(EJEMPLOS, c["archivo"] + ".png"))
        paginas.append(img)
    # PDF: una caja por pagina, lista para imprimir
    paginas[0].save(os.path.join(SALIDA, "cajas_de_prueba.pdf"), save_all=True,
                    append_images=paginas[1:], resolution=150.0)
    print("Cajas creadas en", SALIDA)


if __name__ == "__main__":
    main()
