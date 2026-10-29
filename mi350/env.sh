# MI350X 節點（p01-r01-n01）共用環境：ROCm 7.2.3、OpenMPI 5.0.7 + UCX（複製自管理者在
# /home/mctadmin/works/rocHPL/tpl 的建置，放在 ~/koying/sw，以 OPAL_PREFIX 重新定位）。
#   source mi350/env.sh
export ROCM_PATH=/opt/rocm-7.2.3 HIP_PATH=/opt/rocm-7.2.3
export SW=$HOME/koying/sw
export MPI_HOME=$SW/openmpi OPAL_PREFIX=$SW/openmpi
export PATH=$MPI_HOME/bin:$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$SW/ucx/lib:$MPI_HOME/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}
# /etc/profile.d/env.sh 設了 HIP_VISIBLE_DEVICES=1,3,2,0,5,7,6,4，把 HIP 序重排成 BDF 序（原生的 ROCR/HIP 序是 KFD 序：
# 0→79 1→09 2→69 3→19 4→f9 5→89 6→e9 7→99）。它會和每個 rank 只開一張卡的 ROCR_VISIBLE_DEVICES 衝突（HPL-MxP 的
# device map 用 ROCR 序），所以取消。之後 bench 的 --device 是 KFD 序；分析一律用 bench 印出的 BDF 對齊 amd-smi 序。
unset HIP_VISIBLE_DEVICES
export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx UCX_TLS=sm,self,rocm
