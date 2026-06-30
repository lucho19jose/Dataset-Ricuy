# Esquema de clases

El orden de los identificadores (0–11) corresponde al esquema oficial del modelo
base YOLOv8L-seg y **no debe alterarse**. Durante la anotación en LabelMe se
emplearon sinónimos en español que se mapean a la clase oficial según la
siguiente tabla.

| ID | Clase oficial (YOLO) | Etiqueta usada en LabelMe |
|---:|----------------------|---------------------------|
| 0  | Reductor de velocidad   | Tope de velocidad           |
| 1  | Paso peatonal           | Cruce peatonal              |
| 2  | Línea recta             | Línea recta                 |
| 3  | Línea recta y derecha   | Línea recta a la derecha    |
| 4  | Persona                 | Persona                     |
| 5  | Motocicleta             | Motocicleta                 |
| 6  | Automóvil particular    | Automóvil particular        |
| 7  | Camión                  | Camión                      |
| 8  | Bus de transporte       | Autobús                     |
| 9  | Semáforo en verde       | Semáforo en verde           |
| 10 | Semáforo en rojo        | Semáforo en rojo            |
| 11 | Mototaxi                | Mototaxi                    |

## Notas

- Las clases **Reductor de velocidad** (3), **Línea recta** (4) y
  **Línea recta y derecha** (0) tienen muy pocos o ningún ejemplo en este
  subconjunto; no aprenden señal útil y se reportan por completitud del esquema.
- Las anotaciones son mayoritariamente **polígonos** (segmentación de
  instancias); existe un número reducido de rectángulos en el origen LabelMe.
- En el formato YOLO (`yolo/labels/*.txt`) cada línea es:
  `class_id  x1 y1 x2 y2 ... xn yn` con coordenadas de polígono normalizadas
  a `[0, 1]`.
