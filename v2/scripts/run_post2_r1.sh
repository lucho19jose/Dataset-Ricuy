#!/usr/bin/env bash
# Segunda etapa (18/09): corrección del test, reevaluación, evaluaciones extra, control sin el recorrido del 12/09
# y semillas extra de E2-SV (al final, por ser lo menos urgente).
cd "$(dirname "$0")"
PY=./venv/Scripts/python.exe
L=runs/r1/logs
EV="$PY r1_eval.py --split test"
until grep -q "POST COMPLETO" $L/cola.txt 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') POST2: reconstruyendo dataset con correcciones de test" >> $L/cola.txt
$PY build_dataset_v2.py collect > $L/post2_collect.log 2>&1 && $PY build_dataset_v2.py split > $L/post2_split.log 2>&1 || { echo "$(date '+%F %T') POST2 ERROR en collect/split" >> $L/cola.txt; exit 1; }
# reevaluar todos los modelos en el test corregido
$EV --weights ../best.pt --data dataset_v2/data_E2.yaml --imgsz 640 --name E0_640 > $L/post2_eval.log 2>&1
$EV --weights ../best.pt --data dataset_v2/data_E2.yaml --imgsz 1024 --name E0_1024 >> $L/post2_eval.log 2>&1
for r in E1R_s0 E1R_s1 E1R_s2 E2_s0 E2_s1 E2_s2 E2noSV_s0; do
  $EV --weights runs/r1/$r/weights/last.pt --data dataset_v2/data_E2.yaml --imgsz 640 --name $r >> $L/post2_eval.log 2>&1
done
for s in 0 1 2; do  # resolución de la aplicación
  $EV --weights runs/r1/E2_s$s/weights/last.pt --data dataset_v2/data_E2.yaml --imgsz 1024 --name E2_s${s}_1024 >> $L/post2_eval.log 2>&1
done
$EV --weights runs/r1/E2_s0/weights/last.pt --data dataset_v2/data_test_sinSV.yaml --imgsz 640 --name E2_s0_sinSV >> $L/post2_eval.log 2>&1
$EV --weights ../best.pt --data dataset_v2/data_test_sinSV.yaml --imgsz 640 --name E0_640_sinSV >> $L/post2_eval.log 2>&1
$PY r1_latency.py --weights runs/r1/E2_s0/weights/last.pt --video dataset_v2/splits/test_sinSV.txt --n 400 --imgsz 1024 > $L/latencia_1024.log 2>&1
echo "$(date '+%F %T') POST2: reevaluación completa" >> $L/cola.txt
# control: E2 sin el recorrido del 12/09, evaluado en ese recorrido
echo "$(date '+%F %T') INICIO E2noO semilla 0" >> $L/cola.txt
$PY train_r1.py --exp E2noO --seed 0 > $L/E2noO_s0.log 2>&1
echo "$(date '+%F %T') FIN E2noO semilla 0 (exit $?)" >> $L/cola.txt
$EV --weights ../best.pt --data dataset_v2/data_test_src_O.yaml --imgsz 640 --name E0_640_srcO > $L/post2_srcO.log 2>&1
for r in E1R_s0 E1R_s1 E1R_s2 E2_s0 E2_s1 E2_s2 E2noO_s0; do
  $EV --weights runs/r1/$r/weights/last.pt --data dataset_v2/data_test_src_O.yaml --imgsz 640 --name ${r}_srcO >> $L/post2_srcO.log 2>&1
done
echo "$(date '+%F %T') POST2: control evaluado en el recorrido del 12/09" >> $L/cola.txt
# sensibilidad: señales verticales de cruce anotadas como paso peatonal en D1, enmascaradas (dataset_v2c)
for e in E1Rc E2c; do
  echo "$(date '+%F %T') INICIO $e semilla 0" >> $L/cola.txt
  $PY train_r1.py --exp $e --seed 0 > $L/${e}_s0.log 2>&1
  echo "$(date '+%F %T') FIN $e semilla 0 (exit $?)" >> $L/cola.txt
done
echo "$(date '+%F %T') POST2: sensibilidad de pasos peatonales lista" >> $L/cola.txt
for s in 1 2; do
  echo "$(date '+%F %T') INICIO E2noSV semilla $s" >> $L/cola.txt
  $PY train_r1.py --exp E2noSV --seed $s > $L/E2noSV_s$s.log 2>&1
  echo "$(date '+%F %T') FIN E2noSV semilla $s (exit $?)" >> $L/cola.txt
  $EV --weights runs/r1/E2noSV_s$s/weights/last.pt --data dataset_v2/data_E2.yaml --imgsz 640 --name E2noSV_s$s >> $L/post2_eval.log 2>&1
done
echo "$(date '+%F %T') POST2 COMPLETO" >> $L/cola.txt
