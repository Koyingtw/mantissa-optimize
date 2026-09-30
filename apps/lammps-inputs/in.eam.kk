# bulk Cu lattice, EAM（改自 bench/in.eam：可調大小/步數、thermo 15 位有效數字）
variable        x index 8
variable        y index 8
variable        z index 8
variable        nsteps index 1000
variable        nthermo index 100

variable        xx equal 20*$x
variable        yy equal 20*$y
variable        zz equal 20*$z

units           metal
atom_style      atomic

lattice         fcc 3.615
region          box block 0 ${xx} 0 ${yy} 0 ${zz}
create_box      1 box
create_atoms    1 box

pair_style      eam
pair_coeff      1 1 Cu_u3.eam

velocity        all create 1600.0 376847 loop geom

neighbor        1.0 bin
neigh_modify    every 1 delay 5 check yes

fix             1 all nve

timestep        0.005
thermo_style    custom step temp epair etotal press
thermo_modify   format float %22.15e norm yes
thermo          ${nthermo}

run             ${nsteps}

# 可選：逐位比對用，run 結束後把每個原子的力以 %.17g 依 id 排序寫出（-var dodump 1 -var dumpf <file>）
variable        dodump index 0
variable        dumpf index none
if "${dodump} == 1" then "write_dump all custom ${dumpf} id fx fy fz modify format float %.17g sort id"
