'''
Usage       :   A simple test for the fully coupled algorithm
Author		:	Huipeng Gu
Date		:	2023/10/16
'''


# import the necessary packages
from dolfin import *
import time
set_log_level(40)
time_start = time.time()


# provide parameters for Biot's model
alpha  = 1.0
c0     = 1.0
kp     = 1.0
mu     = 1.0
lmbda  = 1.0


# define the time step and the final time , the time step , and the mesh size
T       = 1.0
stepnum = 4
dt      = T/stepnum
nx = ny = 64
mesh    = UnitSquareMesh(nx, ny)


# define the finite element spaces and the mixed space
V  = VectorElement("CG", mesh.ufl_cell(), 3)
W  = FiniteElement("CG", mesh.ufl_cell(), 2)
M  = FiniteElement("CG", mesh.ufl_cell(), 2)
Qc = FunctionSpace(mesh, MixedElement([V, W, M]))


# define exact solution and right hand side
u_true    = Expression(("0.1*(exp(t)*(x[0]+x[1]*x[1]*x[1]))","0.1*(t*t*(x[0]*x[0]*x[0]+x[1]*x[1]*x[1]))"), degree=3, t=0.0)
p_true    = Expression("10.0*exp((x[0]+x[1])/10.0)*(1.0+t*t*t)", degree=2, t=0.0)
ksi_true  = Expression("alpha*10.0*exp((x[0]+x[1])/10.0)*(1+t*t*t)-lmbda*(0.1*exp(t)+0.3*t*t*x[1]*x[1])", degree=2, alpha=alpha, lmbda=lmbda, t=0.0)

f         = Expression(("-2*mu*(0.3*exp(t)*x[1]) + alpha*exp((x[0]+x[1])/10.0)*(1.0+t*t*t)","-2*mu*(0.3*t*t*x[0] + 0.6*t*t*x[1]) - lmbda*(0.6*t*t*x[1]) + alpha*exp((x[0]+x[1])/10.0)*(1+t*t*t)"), mu=mu, alpha=alpha, lmbda=lmbda, degree=2, t=0.0)
g         = Expression("c0*(30*exp((x[0]+x[1])/10.0)*(t)*(t)) + alpha*(0.1*exp(t)+0.6*(t)*x[1]*x[1]) - 0.2*kp*exp((x[0]+x[1])/10.0)*(1.0+t*t*t)", c0=c0, alpha=alpha, kp=kp, degree=2, t=0.0)
  
w_0       = Expression(("0.1*(x[0]+x[1]*x[1]*x[1])","0.0","alpha*10*exp((x[0]+x[1])/10.0)-lmbda*(0.1)","10*exp((x[0]+x[1])/10.0)"), alpha=alpha, lmbda=lmbda, degree=2, t=0.0)


# define Dirichlet boundary
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS
bc1c = DirichletBC(Qc.sub(0), u_true, boundary)
bc2c = DirichletBC(Qc.sub(2), p_true, boundary)
bcc  = [bc1c, bc2c]


# define the initial value, and the weak formulation
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

wc             = TrialFunction(Qc)
uc, ksic, pc   = split(wc)
vc, phic, psic = TestFunctions(Qc)

w_oldc                   = interpolate(w_0,Qc)
u_oldc, ksi_oldc, p_oldc = split(w_oldc)
    
a1c = 2*mu*inner(epsilon(uc), epsilon(vc))*dx - inner(ksic,div(vc))*dx
a2c = inner(div(uc),phic)*dx + 1.0/lmbda*inner(ksic,phic)*dx - alpha/lmbda*inner(pc,phic)*dx
a3c = (c0 + alpha*alpha/lmbda)/dt*inner(pc-p_oldc,psic)*dx - alpha/lmbda/dt*inner(ksic-ksi_oldc,psic)*dx + kp*inner(grad(pc),grad(psic))*dx
 
L1c = inner(f, vc)*dx 
L2c = inner(g, psic)*dx 
    
Fc  = a1c + a2c + a3c - L1c - L2c
ac, Lc = lhs(Fc), rhs(Fc)


# start solving the problem
t = 0.0
wc = Function(Qc)
while t < T - dt/5.0:    
    t += dt
    #print(t)
    print("t = ", "%.3f" % t , " in total T = ", T , end='\r')
    
    # update boundary conditions
    u_true.t = t
    p_true.t = t
    ksi_true.t = t
    f.t = t
    g.t = t
    
    # compute solution
    solve(ac == Lc, wc, bcc, solver_parameters={"linear_solver": "petsc"}) 
    w_oldc.assign(wc)

time_end = time.time()    
print("t = ", "%.3f" % t , " in total T = ", T)
print('time cost',time_end - time_start,'s')


# visualisation
from matplotlib import pyplot

(uc, ksic, pc) = wc.split(True)
print('error_H1_u   =', errornorm(u_true,uc,'H1'))
print('error_L2_xi  =', errornorm(ksi_true,ksic,'L2'))
print('error_L2_p   =', errornorm(p_true,pc,'L2'))
print('error_H1_p   =', errornorm(p_true,pc,'H1'))
pic = plot(pc)
pyplot.colorbar(pic)
