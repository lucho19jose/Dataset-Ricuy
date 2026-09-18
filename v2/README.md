# RICUY v2: anotaciones, partición por grupos y programas (septiembre de 2026)

Versión que acompaña al artículo corregido. Contiene 384 imágenes anotadas (solo anotaciones) con
3567 instancias en 12 clases. La investigación usó además 28 capturas de Google
Street View, que **no se publican**: por eso los conteos de esta carpeta son menores que los del artículo.

## Fuentes
- `A_...`: Andahuaylas, corpus original, 24 imágenes.
- `L_...`: Lima, corpus original (junio de 2026, cada 5 s), 229 imágenes.
- `M_...`: recolección dirigida, recorrido de 47 min, 83 imágenes.
- `O_...`: recolección dirigida, recorrido del 12/09/2026, 32 imágenes.
- `P_...`: recolección dirigida, fotogramas del recorrido de 47 min, 16 imágenes.

## Partición
Las imágenes se agruparon antes de partir: dos fotogramas de un mismo video pertenecen al mismo grupo si los
separan 30 s o menos, y se unieron además los casi duplicados (hash perceptual de 64 bits, distancia de Hamming
de 10 bits o menos). Cada video de Andahuaylas forma un grupo. Los grupos se repartieron 70/10/20 con una
búsqueda de semilla fija (2026); la asignación está congelada en `split_manifest.csv` (columnas `grupo` y
`split`) y en `yolo/splits/*.txt`. La partición de prueba es la misma para todos los modelos del artículo.

Instancias por clase y partición (entre paréntesis, grupos que contienen la clase):

| ID | Clase | Entrenamiento | Validación | Prueba |
|---:|---|--:|--:|--:|
| 0 | Reductor de velocidad | 21 (4) | 5 (1) | 4 (3) |
| 1 | Paso peatonal | 91 (10) | 9 (5) | 27 (11) |
| 2 | Línea recta | 89 (10) | 14 (6) | 26 (6) |
| 3 | Línea recta y derecha | 41 (7) | 6 (3) | 14 (8) |
| 4 | Persona | 538 (16) | 42 (7) | 128 (17) |
| 5 | Motocicleta | 63 (11) | 13 (6) | 20 (9) |
| 6 | Automóvil particular | 1192 (18) | 159 (9) | 360 (18) |
| 7 | Camión | 46 (8) | 12 (2) | 22 (8) |
| 8 | Bus de transporte | 212 (10) | 35 (7) | 48 (8) |
| 9 | Semáforo en verde | 69 (6) | 4 (3) | 12 (5) |
| 10 | Semáforo en rojo | 72 (7) | 7 (3) | 21 (5) |
| 11 | Mototaxi | 104 (10) | 2 (2) | 39 (6) |

## Cambios respecto de la versión 1
- **Semáforos:** un polígono por cabeza de semáforo. Los pórticos completos se reemplazaron por las cabezas
  que propuso el modelo base o se rellenaron de gris si no había cabeza identificable
  (`registros/semaforos_armonizacion.json`, `registros/changelog_etiquetas.csv`).
- **Clases 2 y 3:** flechas pintadas en la calzada, no líneas de carril.
- **Recolección dirigida:** fotogramas elegidos por contener flechas, pasos peatonales, reductores o semáforos,
  con todas las clases visibles anotadas.
- **Correcciones en prueba:** candidatos revisados y polígonos mal asignados (`registros/candidatos_revision_test.json`,
  `registros/correcciones_test.json`).
- **Advertencia sobre el paso peatonal:** 23 polígonos de clase 1 del corpus original de entrenamiento encierran la
  señal vertical de cruce peatonal y no la cebra (`registros/senales_paso_peatonal.csv`). Se conservan tal como se
  usaron en los experimentos principales; `scripts/r1_senales_paso.py` los enmascara para el análisis de
  sensibilidad del artículo. Si entrenas un modelo nuevo, conviene excluirlos.

## Reproducir
1. Solicitar las imágenes (ver el README de la raíz) y colocarlas en `yolo/images/` con los nombres de
   `split_manifest.csv`.
2. Instalar `requirements.txt`. El ajuste parte del punto de control de QHAWAY (MD5
   `2def426c0820750fae575bab517464f2`), que se solicita a sus autores.
3. `scripts/train_r1.py --exp E2 --seed 0` entrena con la configuración del artículo (80 épocas, lote 2,
   AdamW 0.000625, sin volteo horizontal) y `scripts/r1_results.py` recalcula las tablas y los intervalos.
   `scripts/build_dataset_v2.py` documenta la unificación desde las carpetas crudas, que no se publican; las
   anotaciones de `labelme/` son el punto de partida reproducible.

`registros/resumen_resultados.json` contiene todas las métricas del artículo (prueba principal sin Street View,
por fuente, por clase, con intervalos de confianza por *bootstrap* de grupos).
`protocolo_campo/` contiene el protocolo para validar la distancia con cinta métrica desde una mototaxi.
