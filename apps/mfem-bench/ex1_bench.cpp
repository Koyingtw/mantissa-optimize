// ex1_bench: MFEM ex1 改寫的 GPU 基準（尾數截斷研究用）
//
// 3D Poisson/擴散問題  -div(kappa grad u) + sigma u = f，單位立方體、齊次 Dirichlet 邊界，
// 製造解 u = sin(pi x) sin(pi y) sin(pi z)，kappa = 1 + 0.5 (x+y+z)（在物理座標上線性）。
// 網格：n^3 個六面體，SetCurvature(p) 後把內部節點用 a*sin(pi x)sin(pi y)sin(pi z)
// 擾動（邊界面上擾動為 0，所以區域仍是單位立方體），使 Jacobian 非對角且逐點不同
// → PA 的求積點資料 pa_data（6 個對稱分量 × Q1D^3 × NE）全部是滿尾數的 FP64。
// 係數 kappa 與右手邊 f 都用 order-p 節點內插的 GridFunctionCoefficient
// （在 GPU 上由 QuadratureInterpolator 求值，避免 host 端數千萬次函數求值）。
//
// 求解：Partial Assembly (-pa) + Jacobi 前處理 CG，裝置 -d cuda。可重複求解 -rep 次
// 以拉長執行時間；-fixed N 則固定跑 N 次 CG 迭代（rtol=atol=0）。
// 另外可 -matvec M 單獨計時 M 次 A*x。
//
// 正確性量（%.17g）：(B, A B)、每次求解的迭代數與最終殘差、||X||_2、(X,B)、
// 節點上對製造解的 max / RMS 誤差。
#include "mfem.hpp"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <chrono>

using namespace std;
using namespace mfem;

static double amp = 0.05;

static inline double wallclock()
{
   using namespace std::chrono;
   return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char *argv[])
{
   int n = 64;
   int order = 3;
   const char *device_config = "cuda";
   bool pa = true;
   bool use_mass = false;
   bool var_kappa = true;
   bool jacobi = true;
   double rtol = 1e-10;
   int maxit = 20000;
   int rep = 1;
   int fixed = 0;
   int matvec = 0;
   int print_level = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&n, "-n", "--nx", "每個方向的六面體元素數（n^3 個元素）");
   args.AddOption(&order, "-o", "--order", "多項式階數 p");
   args.AddOption(&device_config, "-d", "--device", "Device 設定（cuda / cpu）");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa", "--no-partial-assembly",
                  "Partial assembly");
   args.AddOption(&use_mass, "-mass", "--mass", "-no-mass", "--no-mass",
                  "加上 MassIntegrator（sigma=1）");
   args.AddOption(&var_kappa, "-vk", "--var-kappa", "-ck", "--const-kappa",
                  "kappa = 1+0.5(x+y+z)（預設）或常數 1");
   args.AddOption(&amp, "-a", "--amp", "內部節點擾動振幅");
   args.AddOption(&jacobi, "-jac", "--jacobi", "-no-jac", "--no-jacobi",
                  "Jacobi 前處理");
   args.AddOption(&rtol, "-rtol", "--rtol", "CG 相對容許誤差");
   args.AddOption(&maxit, "-maxit", "--maxit", "CG 最大迭代數");
   args.AddOption(&rep, "-rep", "--repeat", "重複求解次數");
   args.AddOption(&fixed, "-fixed", "--fixed-its", "固定 CG 迭代數（>0 時 rtol=atol=0）");
   args.AddOption(&matvec, "-matvec", "--matvec", "額外計時 M 次 A*x");
   args.AddOption(&print_level, "-pl", "--print-level", "CG print level");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   Device device(device_config);
   device.Print();

   double t0 = wallclock();
   // 1. 網格：n^3 六面體，p 階曲面幾何，內部節點擾動
   Mesh mesh = Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   mesh.SetCurvature(order, false, 3, Ordering::byNODES);
   GridFunction &nodes = *mesh.GetNodes();
   const FiniteElementSpace &nfes = *nodes.FESpace();
   const int nnd = nfes.GetNDofs();
   {
      double *X = nodes.HostReadWrite();
      for (int i = 0; i < nnd; i++)
      {
         const int ix = nfes.DofToVDof(i, 0), iy = nfes.DofToVDof(i, 1),
                   iz = nfes.DofToVDof(i, 2);
         const double x = X[ix], y = X[iy], z = X[iz];
         const double s = sin(M_PI*x)*sin(M_PI*y)*sin(M_PI*z);
         X[ix] = x + amp*s;
         X[iy] = y + 0.7*amp*s*cos(M_PI*x);
         X[iz] = z - 0.5*amp*s*cos(M_PI*y);
      }
   }
   mesh.NodesUpdated();
   const int dim = mesh.Dimension();

   // 2. H1 空間（與網格節點同一 fec 與 dof 編號）
   H1_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fespace(&mesh, &fec);
   MFEM_VERIFY(fespace.GetNDofs() == nnd, "dof numbering mismatch");
   const int ndofs = fespace.GetTrueVSize();
   printf("[SETUP] n=%d p=%d NE=%d ndofs=%d amp=%g kappa=%s mass=%d\n",
          n, order, mesh.GetNE(), ndofs, amp, var_kappa ? "var" : "const",
          (int)use_mass);

   // 3. 節點內插：kappa、f、u_exact
   GridFunction kappa_gf(&fespace), f_gf(&fespace), uex(&fespace);
   {
      const double *X = nodes.HostRead();
      double *K = kappa_gf.HostWrite(), *F = f_gf.HostWrite(), *U = uex.HostWrite();
      const double pi = M_PI;
      for (int i = 0; i < nnd; i++)
      {
         const double x = X[nfes.DofToVDof(i, 0)], y = X[nfes.DofToVDof(i, 1)],
                      z = X[nfes.DofToVDof(i, 2)];
         const double sx = sin(pi*x), sy = sin(pi*y), sz = sin(pi*z);
         const double cx = cos(pi*x), cy = cos(pi*y), cz = cos(pi*z);
         const double u = sx*sy*sz;
         const double k = var_kappa ? 1.0 + 0.5*(x + y + z) : 1.0;
         double f = k*3.0*pi*pi*u;
         if (var_kappa) { f -= 0.5*pi*(cx*sy*sz + sx*cy*sz + sx*sy*cz); }
         if (use_mass) { f += u; }
         K[i] = k; F[i] = f; U[i] = u;
      }
   }
   GridFunctionCoefficient kappa(&kappa_gf), fcoef(&f_gf);
   ConstantCoefficient one(1.0);

   Array<int> ess_tdof_list;
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(fcoef));
   b.UseFastAssembly(true);
   b.Assemble();

   GridFunction x(&fespace);
   x = 0.0;

   BilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   if (var_kappa) { a.AddDomainIntegrator(new DiffusionIntegrator(kappa)); }
   else { a.AddDomainIntegrator(new DiffusionIntegrator(one)); }
   if (use_mass) { a.AddDomainIntegrator(new MassIntegrator(one)); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   printf("[SETUP] size=%d ess=%d setup_time=%.3f s\n", A->Height(),
          ess_tdof_list.Size(), wallclock() - t0);

   // 4. 算子作用檢查：q = (B, A B)
   Vector AB(B.Size());
   AB.UseDevice(true);
   A->Mult(B, AB);
   printf("[CHECK] BnormL2=%.17g  (B,AB)=%.17g  ||AB||=%.17g\n",
          B.Norml2(), InnerProduct(B, AB), AB.Norml2());

   // 5. 前處理器
   Solver *M = nullptr;
   if (jacobi && pa) { M = new OperatorJacobiSmoother(a, ess_tdof_list); }

   // 6. 可選：純 matvec 計時
   if (matvec > 0)
   {
      Vector y(B.Size()); y.UseDevice(true);
      MFEM_DEVICE_SYNC;
      const double tm0 = wallclock();
      for (int i = 0; i < matvec; i++) { A->Mult(B, y); }
      MFEM_DEVICE_SYNC;
      const double tm = wallclock() - tm0;
      printf("[MATVEC] count=%d time=%.3f s  per=%.4f ms  %.3f GDOF/s  ||y||=%.17g\n",
             matvec, tm, 1e3*tm/matvec, 1e-9*(double)ndofs*matvec/tm, y.Norml2());
   }

   // 7. CG 求解（重複 rep 次，每次從 X=0 開始）
   CGSolver cg;
   cg.SetOperator(*A);
   if (M) { cg.SetPreconditioner(*M); }
   if (fixed > 0) { cg.SetRelTol(0.0); cg.SetAbsTol(0.0); cg.SetMaxIter(fixed); }
   else { cg.SetRelTol(rtol); cg.SetAbsTol(0.0); cg.SetMaxIter(maxit); }
   cg.SetPrintLevel(print_level);
   long long total_its = 0;
   double tsolve = 0.0;
   for (int r = 0; r < rep; r++)
   {
      X = 0.0;
      MFEM_DEVICE_SYNC;
      const double ts0 = wallclock();
      cg.Mult(B, X);
      MFEM_DEVICE_SYNC;
      const double ts = wallclock() - ts0;
      tsolve += ts;
      total_its += cg.GetNumIterations();
      printf("[SOLVE] rep=%d its=%d converged=%d final_norm=%.17g time=%.3f s\n",
             r, cg.GetNumIterations(), (int)cg.GetConverged(), cg.GetFinalNorm(), ts);
      fflush(stdout);
   }
   printf("[TIME] solves=%d total_its=%lld solve_time=%.3f s  per_it=%.4f ms  %.3f GDOF/s(CG it)\n",
          rep, total_its, tsolve, 1e3*tsolve/max(1LL, total_its),
          1e-9*(double)ndofs*total_its/tsolve);

   // 8. 結果與誤差
   printf("[RESULT] ||X||=%.17g  (X,B)=%.17g\n", X.Norml2(), InnerProduct(X, B));
   a.RecoverFEMSolution(X, b, x);
   {
      const double *xs = x.HostRead(), *ue = uex.HostRead();
      double emax = 0.0, esum = 0.0;
      for (int i = 0; i < nnd; i++)
      {
         const double e = fabs(xs[i] - ue[i]);
         emax = max(emax, e); esum += e*e;
      }
      printf("[RESULT] nodal_err_max=%.17g nodal_err_rms=%.17g\n", emax,
             sqrt(esum/nnd));
   }
   printf("[TIME] wall_total=%.3f s\n", wallclock() - t0);

   delete M;
   return 0;
}
