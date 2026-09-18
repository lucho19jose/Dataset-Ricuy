#!/usr/bin/env bash
# Después de la cola R1: verificación controlada del semáforo rojo y latencia (GPU libre).
cd "$(dirname "$0")"
PY=./venv/Scripts/python.exe
until grep -q "COLA COMPLETA" runs/r1/logs/cola.txt 2>/dev/null; do sleep 60; done
$PY r1_check_semaforo.py > runs/r1/logs/check_semaforo.log 2>&1
$PY r1_latency.py --weights runs/r1/E2_s0/weights/last.pt --video dataset_v2/splits/test_sinSV.txt --n 400 > runs/r1/logs/latencia.log 2>&1
echo "$(date '+%F %T') POST COMPLETO" >> runs/r1/logs/cola.txt
