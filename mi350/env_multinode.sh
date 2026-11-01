# 兩節點（n01+n02）的 MPI／RCCL 設定，source mi350/env.sh 之後再 source 本檔；以 srun --mpi=pmix 啟動（n02 不接受 ssh，不能用 mpirun）。
#   MPI（UCX）：只有前端 RoCE 網卡 rocep51s0（ens512np0，192.168.254.x）有 rc_verbs，約 35 GB/s、5 µs；後端 8 張網卡在 UCX 只有 ud。
#   RCCL：用 8 張後端 RoCE 網卡（be*-gpuN，Broadcom bnxt），RoCEv2 的 IPv4 GID 在 index 1（前端網卡是 3），GPUDirect RDMA；
#   16 GPU all-reduce 的 bus bandwidth 約 260–280 GB/s（mi350/tools/rccl_allreduce.hip）。
#   兩節點都有 br-* 橋接網卡（172.18.0.1，同一網段），不能讓 OOB 走它，所以 socket 介面固定為 ens512np0。
export OMPI_MCA_opal_common_ucx_tls=any OMPI_MCA_opal_common_ucx_devices=any
export UCX_TLS=rc_verbs,ud_verbs,self,sm UCX_NET_DEVICES=rocep51s0:1
export NCCL_SOCKET_IFNAME=ens512np0 NCCL_IB_GID_INDEX=1
export NCCL_IB_HCA=rocep118s0,rocep6s0,rocep102s0,rocep22s0,rocep246s0,rocep134s0,rocep230s0,rocep150s0
