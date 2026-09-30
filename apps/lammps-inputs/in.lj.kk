# 3d Lennard-Jones melt（改自 bench/in.lj：可調大小/步數、thermo 15 位有效數字）
variable        x index 8
variable        y index 8
variable        z index 8
variable        nsteps index 1000
variable        nthermo index 100
variable        eps index 1.0
variable        sig index 1.0

variable        xx equal 20*$x
variable        yy equal 20*$y
variable        zz equal 20*$z

units           lj
atom_style      atomic

lattice         fcc 0.8442
region          box block 0 ${xx} 0 ${yy} 0 ${zz}
create_box      1 box
create_atoms    1 box
mass            1 1.0

velocity        all create 1.44 87287 loop geom

pair_style      lj/cut 2.5
pair_coeff      1 1 ${eps} ${sig} 2.5

neighbor        0.3 bin
neigh_modify    delay 0 every 20 check no

fix             1 all nve

thermo_style    custom step temp epair emol etotal press
thermo_modify   format float %22.15e norm yes
thermo          ${nthermo}
run             ${nsteps}

# 可選：逐位比對用，run 結束後把每個原子的力以 %.17g 依 id 排序寫出（-var dodump 1 -var dumpf <file>）
variable        dodump index 0
variable        dumpf index none
if "${dodump} == 1" then "write_dump all custom ${dumpf} id fx fy fz modify format float %.17g sort id"
