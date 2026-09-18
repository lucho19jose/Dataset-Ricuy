# Dataset RICUY: anotaciones de tráfico urbano (Lima y Andahuaylas)

Anotaciones de **segmentación de instancias** de actores y señalización de tráfico en vías urbanas de Lima y
Andahuaylas (Perú), usadas en el sistema de asistencia al conductor **RICUY** (YOLOv8l-seg, 12 clases).

> **Este repositorio contiene solo anotaciones, no imágenes.** Las imágenes provienen de grabaciones en vías
> públicas y contienen rostros y placas; se comparten bajo solicitud para uso académico
> (ver [Cómo solicitar las imágenes](#cómo-solicitar-las-imágenes)). Las capturas de Google Street View usadas en
> la investigación no se publican en ninguna forma.

## Versiones

| Carpeta | Versión | Uso |
|---|---|---|
| [`v2/`](v2/) | **Versión corregida (septiembre de 2026)** | Artículo corregido tras la revisión por pares. Partición por grupos, recolección dirigida, programas y resultados. **Usar esta.** |
| `labelme/`, `yolo/` (raíz) | Versión 1 (junio de 2026) | Envío original: 253 imágenes con partición aleatoria 80/20 (semilla 42). Se conserva solo como registro. |

La versión 1 tenía dos problemas que la versión 2 corrige: la partición aleatoria dejaba fotogramas vecinos de un
mismo video en entrenamiento y validación, y los semáforos estaban anotados con dos convenciones (cabeza o pórtico
completo). Los detalles están en [`v2/README.md`](v2/README.md).

## Esquema de clases

El orden de identificadores es el del modelo base (QHAWAY) y no debe alterarse. Las clases 2 y 3 son las
**flechas direccionales pintadas** en la calzada, no las líneas que separan carriles.

| ID | Clase | ID | Clase |
|---:|---|---:|---|
| 0 | Reductor de velocidad | 6 | Automóvil particular |
| 1 | Paso peatonal | 7 | Camión |
| 2 | Línea recta (flecha recta) | 8 | Bus de transporte |
| 3 | Línea recta y derecha (flecha recta y derecha) | 9 | Semáforo en verde |
| 4 | Persona | 10 | Semáforo en rojo |
| 5 | Motocicleta | 11 | Mototaxi |

Sinónimos de LabelMe aceptados en [`CLASSES.md`](CLASSES.md).

## Cómo solicitar las imágenes

Las imágenes se comparten **bajo solicitud, para uso académico o de investigación**, con el compromiso de no
redistribuirlas y de anonimizar rostros y placas en cualquier publicación. Abre un *issue* en este repositorio
describiendo tu afiliación y el uso previsto, o escribe al autor de correspondencia:
**barbozagonzalesjose@gmail.com**.

El punto de control del modelo base (QHAWAY, YOLOv8l-seg) debe solicitarse a sus autores (De la Cruz et al.,
*Sensors* 2026, 26, 2569, doi: 10.3390/s26082569). Su huella MD5 es `2def426c0820750fae575bab517464f2`.

## Cita

> Barboza Gonzales, J. L.; Tenorio Huarancca, D. O.; Ruiz Crisostomo, P. V.; Yufra Chambilla, M.;
> Salas Vizcarra, O. *RICUY: Sistema Inteligente de Asistencia al Conductor en Vías Urbanas de Lima y
> Andahuaylas*. Journal of Artificial Intelligence for Engineering Solutions, Escuela de Posgrado UNI, 2026
> (en revisión).

## Licencia

Las **anotaciones** se distribuyen bajo [Creative Commons Attribution 4.0 (CC BY 4.0)](LICENSE). Las imágenes no
están cubiertas por esta licencia.
