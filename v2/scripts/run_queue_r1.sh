#!/usr/bin/env bash
# Cola R1 (etiquetas de semáforo armonizadas, 18/09): E0 sin ajuste y luego las corridas en orden de prioridad.
# Si se corta, volver a lanzarla reanuda desde last.pt y salta lo ya evaluado en test.
cd "$(dirname "$0")"
PY=./venv/Scripts/python.exe
mkdir -p runs/r1/logs
for sz in 640 1024; do
  [ -f "runs/r1/eval/E0_${sz}_test.json" ] || $PY r1_eval.py --weights ../best.pt --data dataset_v2/data_E2.yaml --split test --imgsz $sz --name E0_$sz > "runs/r1/logs/E0_$sz.log" 2>&1
done
echo "$(date '+%F %T') E0 evaluado (640 y 1024)" >> runs/r1/logs/cola.txt
for job in "E2 0" "E1R 0" "E2 1" "E1R 1" "E2noSV 0" "E2 2" "E1R 2"; do
  set -- $job
  echo "$(date '+%F %T') INICIO $1 semilla $2" >> runs/r1/logs/cola.txt
  $PY train_r1.py --exp "$1" --seed "$2" > "runs/r1/logs/$1_s$2.log" 2>&1
  echo "$(date '+%F %T') FIN $1 semilla $2 (exit $?)" >> runs/r1/logs/cola.txt
done
echo "$(date '+%F %T') COLA COMPLETA" >> runs/r1/logs/cola.txt
