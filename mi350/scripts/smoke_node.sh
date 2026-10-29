#!/bin/bash
# 新節點的冒煙測試：MPI、GEMM bench、取樣器、功耗上限設定／讀回／還原
cd "$(dirname "${BASH_SOURCE[0]}")/.." && source env.sh
mpirun -np 8 --bind-to none hostname | sort | uniq -c
.venv/bin/python tools/smi_sampler.py /tmp/smoke_smi.csv 0.1 & S=$!
tools/pcap.sh 700 && echo "cap700: $(tools/pcap.sh show | awk '{print $2}' | sort | uniq -c | tr '\n' ' ')"
bin/gemm_mant_bench --dtype fp64 --warmup 1 --seconds 2 | cut -c1-140
tools/pcap.sh 1000 && echo "cap1000: $(tools/pcap.sh show | awk '{print $2}' | sort | uniq -c | tr '\n' ' ')"
kill $S; sleep 0.5; wc -l /tmp/smoke_smi.csv; rm -f /tmp/smoke_smi.csv
