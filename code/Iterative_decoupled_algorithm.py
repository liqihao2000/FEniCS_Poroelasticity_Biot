'''
Usage       :   A simple test for the iterative decoupled algorithm
Author		:	Huipeng Gu
Date		:	2023/10/21
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
stepnum = 64
dt      = T/stepnum
nx = ny = 8
mesh    = UnitSquareMesh(nx, ny)
iternum = 100


# define the finite element spaces and the mixed space
V  = VectorElement("CG", mesh.ufl_cell(), 2)
W  = FiniteElement("CG", mesh.ufl_cell(), 1)
M  = FiniteElement("CG", mesh.ufl_cell(), 1)
Q1 = FunctionSpace(mesh, MixedElement([V, W]))
Q2 = FunctionSpace(mesh, M)


# define exact solution and right hand side
u_true    = Expression(("0.1*(exp(t)*(x[0]+x[1]*x[1]*x[1]))","0.1*(t*t*(x[0]*x[0]*x[0]+x[1]*x[1]*x[1]))"), degree=3, t=0.0)
p_true    = Expression("10.0*exp((x[0]+x[1])/10.0)*(1.0+t*t*t)", degree=2, t=0.0)
ksi_true  = Expression("alpha*10.0*exp((x[0]+x[1])/10.0)*(1+t*t*t)-lmbda*(0.1*exp(t)+0.3*t*t*x[1]*x[1])", degree=2, alpha=alpha, lmbda=lmbda, t=0.0)

f         = Expression(("-2*mu*(0.3*exp(t)*x[1]) + alpha*exp((x[0]+x[1])/10.0)*(1.0+t*t*t)","-2*mu*(0.3*t*t*x[0] + 0.6*t*t*x[1]) - lmbda*(0.6*t*t*x[1]) + alpha*exp((x[0]+x[1])/10.0)*(1+t*t*t)"), mu=mu, alpha=alpha, lmbda=lmbda, degree=2, t=0.0)
g         = Expression("c0*(30*exp((x[0]+x[1])/10.0)*(t)*(t)) + alpha*(0.1*exp(t)+0.6*(t)*x[1]*x[1]) - 0.2*kp*exp((x[0]+x[1])/10.0)*(1.0+t*t*t)", c0=c0, alpha=alpha, kp=kp, degree=2, t=0.0)
  
w1_0      = Expression(("0.1*(x[0]+x[1]*x[1]*x[1])","0.0","alpha*10*exp((x[0]+x[1])/10.0)-lmbda*(0.1)"), alpha=alpha, lmbda=lmbda, degree=2, t=0.0)
p_0       = Expression("10*exp((x[0]+x[1])/10.0)",degree=2, t=0.0)


# define Dirichlet boundary
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS
bc11     = DirichletBC(Q1.sub(0), u_true, boundary)
bc22     = DirichletBC(Q2, p_true, boundary)


# define the initial value, and the weak formulation
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

w1       = TrialFunction(Q1)
u, ksi   = split(w1)
p        = TrialFunction(Q2)
v, phi   = TestFunctions(Q1)
psi      = TestFunction(Q2)

w1_old   = interpolate(w1_0,Q1)
w1_      = interpolate(w1_0,Q1)
p_old    = interpolate(p_0,Q2)
p_       = interpolate(p_0,Q2)
u_, ksi_ = split(w1_)
u_old, ksi_old = split(w1_old)
    
a1 = 2*mu*inner(epsilon(u), epsilon(v))*dx - inner(ksi,div(v))*dx
a2 = inner(div(u),phi)*dx + 1.0/lmbda*inner(ksi,phi)*dx - alpha/lmbda*inner(p_,phi)*dx
a3 = (c0 + alpha*alpha/lmbda)/dt*inner(p-p_old,psi)*dx - alpha/lmbda/dt*inner(ksi_-ksi_old,psi)*dx + kp*inner(grad(p),grad(psi))*dx
 
L1 = inner(f, v)*dx 
L2 = inner(g, psi)*dx 
    
F1 = a1 + a2 - L1
F2 = a3 - L2

a_1, L_1 = lhs(F1), rhs(F1)
a_2, L_2 = lhs(F2), rhs(F2)


# start solving the problem
t = 0.0
w1 = Function(Q1)
p  = Function(Q2)
while t < T - dt/5.0:    
    t += dt
    print("t = ", "%.3f" % t , " in total T = ", T , end='\r')
    
    # update boundary conditions
    u_true.t = t
    p_true.t = t
    ksi_true.t = t
    f.t = t
    g.t = t
    
    # compute solution
    for i in range(iternum):
        solve(a_2 == L_2, p, bc22, solver_parameters={"linear_solver": "petsc"}) 
        p_.assign(p)

        solve(a_1 == L_1, w1, bc11, solver_parameters={"linear_solver": "petsc"})
        w1_.assign(w1)

    w1_old.assign(w1_)
    p_old.assign(p_)

time_end = time.time()    
print("t = ", "%.3f" % t , " in total T = ", T)
print('time cost',time_end - time_start,'s')


# visualisation
from matplotlib import pyplot

(u, ksi) = w1.split(True)
print('error_H1_u   =', errornorm(u_true,u,'H1'))
print('error_L2_xi  =', errornorm(ksi_true,ksi,'L2'))
print('error_L2_p   =', errornorm(p_true,p,'L2'))
print('error_H1_p   =', errornorm(p_true,p,'H1'))
pic = plot(p)
pyplot.colorbar(pic)
