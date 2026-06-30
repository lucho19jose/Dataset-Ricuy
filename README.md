# Dataset RICUY — Anotaciones de tráfico urbano (Lima)

Anotaciones de **segmentación de instancias** de actores y señalización de
tráfico en vías urbanas de Lima, Perú. Este conjunto es el subconjunto curado
empleado en el estudio de transferencia Ayacucho→Lima del sistema **RICUY**
(ADAS basado en YOLOv8L-seg).

> **⚠️ Este repositorio contiene SOLO las anotaciones, no las imágenes.**
> Las imágenes provienen de grabaciones de cámara de tablero en vías públicas y
> contienen rostros de peatones y placas vehiculares. Por motivos de
> **protección de datos**, no se publican en abierto. Consulta
> [«Cómo solicitar las imágenes»](#cómo-solicitar-las-imágenes).

---

## Contenido

```
Dataset-Ricuy/
├── labelme/                # Anotaciones originales LabelMe (.json)
│   ├── train/  (204 archivos)
│   └── val/    ( 49 archivos)
├── yolo/                    # Anotaciones exportadas a formato YOLOv8-seg
│   ├── data.yaml           # esquema de 12 clases + rutas (relativas)
│   └── labels/
│       ├── train/ (204 .txt)
│       └── val/   ( 49 .txt)
├── CLASSES.md               # Esquema oficial de clases y sinónimos LabelMe
├── LICENSE                  # CC BY 4.0 (solo anotaciones)
└── README.md
```

- **253 imágenes anotadas** (split determinista 80/20, *seed* = 42):
  **204 entrenamiento / 49 validación**.
- **2230 instancias** anotadas (2508 polígonos y 31 rectángulos en el origen
  LabelMe; predominan los polígonos de segmentación).
- **12 clases** (3 de ellas con muy pocos o cero ejemplos; ver tabla).
- Los `.json` de LabelMe se publican con el campo `imageData` vaciado
  (`null`): **no incrustan la imagen**.

## Esquema de clases e instancias (subconjunto de 253)

| ID | Clase (oficial) | Sinónimo LabelMe | Instancias |
|---:|------------------|------------------|-----------:|
| 6 | Automóvil particular | Automóvil particular | 1184 |
| 4 | Persona | Persona | 454 |
| 8 | Bus de transporte | Autobús | 236 |
| 11 | Mototaxi | Mototaxi | 90 |
| 10 | Semáforo en rojo | Semáforo en rojo | 69 |
| 7 | Camión | Camión | 58 |
| 5 | Motocicleta | Motocicleta | 51 |
| 9 | Semáforo en verde | Semáforo en verde | 48 |
| 1 | Paso peatonal | Cruce peatonal | 33 |
| 2 | Línea recta | Línea recta | 4 |
| 0 | Reductor de velocidad | Tope de velocidad | 3 |
| 3 | Línea recta y derecha | Línea recta a la derecha | 0 |

El orden de IDs sigue el esquema oficial del modelo base (no alterar). Detalle
completo en [`CLASSES.md`](CLASSES.md).

## Cómo reproducir el split YOLO

1. Solicita las imágenes (ver abajo) y colócalas en:
   `yolo/images/train/` y `yolo/images/val/` (mismos nombres base que los `.txt`).
2. Entrena con `yolo/data.yaml` (Ultralytics ≥ 8.x), por ejemplo:
   ```bash
   yolo segment train data=yolo/data.yaml model=yolov8l-seg.pt imgsz=640
   ```

## Cómo solicitar las imágenes

Las imágenes se comparten **bajo solicitud, para uso académico/de investigación**,
previo compromiso de tratamiento responsable de datos personales (no
redistribución, anonimización en publicaciones).

- **Abre un *Issue*** en este repositorio describiendo tu afiliación y el uso
  previsto, **o**
- Escribe al autor de correspondencia: **barbozagonzalesjose@gmail.com**.

## Cita

Si utilizas estas anotaciones, por favor cita el artículo asociado:

> Barboza, Ruiz, Tenorio, Yufra, Salas. *RICUY: Sistema Inteligente de
> Asistencia al Conductor en vías urbanas de Lima, Andahuaylas y Ayacucho*.
> Universidad Nacional de Ingeniería (UNI), 2026. *(completar datos de
> publicación / DOI cuando estén disponibles).*

## Licencia

Las **anotaciones** de este repositorio se distribuyen bajo
[Creative Commons Attribution 4.0 (CC BY 4.0)](LICENSE).
Las **imágenes** no están cubiertas por esta licencia y se rigen por el proceso
de solicitud descrito arriba.
