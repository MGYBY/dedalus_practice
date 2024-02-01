"""
Dedalus script simulating the 1D Korteweg-de Vries / Burgers equation.
This script demonstrates solving a 1D initial value problem and produces
a space-time plot of the solution. It should take just a few seconds to
run (serial only).

We use a Fourier basis to solve the IVP:
    dt(u) + u*dx(u) = a*dx(dx(u)) + b*dx(dx(dx(u)))

To run and plot:
    $ python3 kdv_burgers.py
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
import csv
logger = logging.getLogger(__name__)


# Parameters
tan_theta = 0.11217
Lx = 172.5*tan_theta
fro = 0.8501
rey = 3.0*fro*fro/tan_theta
web = 5.430
deb = 0.0
lambda_r = 0.0 # r = lambda_2/lambda_1
gw_c = 1.0+1.0/fro
film_param = tan_theta
disp_amp = 0.1250

Nx = 2048
dealias = 3/2
stop_sim_time = 80.0
output_interval = 0.250
timestepper = d3.RK222
# timestepper = d3.SBDF2
init_timestep = 0.160*0.50*(Lx/Nx)/gw_c/2.0
max_timestep = init_timestep*4.0
dtype = np.float64

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype)
xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)
# xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx))
# xbasis = d3.Chebyshev(xcoord, size=Nx, bounds=(0, Lx))

# Fields
h = dist.Field(name='h', bases=xbasis)
q = dist.Field(name='q', bases=xbasis)
# q = dist.VectorField(xcoord, name='q', bases=xbasis)

# Substitutions
dx = lambda A: d3.Differentiate(A, xcoord)
u = q/h
ve_coeff = 1.0-(5.0*deb*(1.0-lambda_r))/(2.0*rey*h*h)

# Problem
problem = d3.IVP([h,q], namespace=locals())
problem.add_equation("dt(h) =  -dx(q)")
# problem.add_equation("dt(q) + (1/ve_coeff)*dx((9*q*q)/(7*h) + 5/(4*rey*tan_theta)*h*h) = (1/ve_coeff)*(q/(7*h)*dx(q) + 5/(2*film_param*rey)*(h-q/h/h) + 5*film_param*film_param*web/2/rey*h*dx(dx(dx(h))) + (film_param/rey)*(9/2*dx(dx(q)) - 9/(2*h)*dx(q)*dx(h) + (4*q)/(h*h)*(dx(h)*dx(h)) - (6*q)/h*dx(dx(h))) + ((5*deb*(1-lambda_r))/(2*rey*h**4))*(6*h*q*dx(q) - 6*q*q*dx(h)))")

# a simplified version of the governing equations: no visco-elastic effect
# problem.add_equation("dt(q) + dx((9*q*q)/(7*h) + 5/(4*rey*tan_theta)*h*h) = q/(7*h)*dx(q) + 5/(2*film_param*rey)*(h-q/h/h) + 5*film_param*film_param*web/2/rey*h*dx(dx(dx(h))) + (film_param/rey)*(9/2*dx(dx(q)) - 9/(2*h)*dx(q)*dx(h) + (4*q)/(h*h)*(dx(h)*dx(h)) - (6*q)/h*dx(dx(h)))")
problem.add_equation("dt(q) = dx(-(9*q*q)/(7*h) - 5/(4*rey*tan_theta)*h*h) + q/(7*h)*dx(q) + 5/(2*film_param*rey)*(h-q/h/h) + 5*film_param*film_param*web/2/rey*h*dx(dx(dx(h))) + (film_param/rey)*(9/2*dx(dx(q)) - 9/(2*h)*dx(q)*dx(h) + (4*q)/(h*h)*(dx(h)*dx(h)) - (6*q)/h*dx(dx(h)))")

# Initial conditions
x = dist.local_grid(xbasis)
q['g'] = 1.0
h['g'] = 1.0*(1.0+disp_amp*np.sin(2.0*np.pi*x/Lx))

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Analysis
# snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=output_interval, max_writes=50)
# snapshots.add_task(h, name='depth')
# snapshots.add_task(q, name='discharge')

# use constant timestep for now
# CFL
# CFL = d3.CFL(solver, initial_dt=init_timestep, cadence=10, safety=0.2, threshold=0.1,
#              max_change=1.5, min_change=0.5, max_dt=max_timestep)
# CFL.add_velocity(u)

# Main loop
# try:
#     logger.info('Starting main loop')
#     while solver.proceed:
#         timestep = CFL.compute_timestep()
#         solver.step(timestep)
#         if solver.iteration % 10 == 0:
#             logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
# except:
#     logger.error('Exception raised, triggering end of main loop.')
#     raise
# finally:
#     solver.log_stats()

h.change_scales(1)
q.change_scales(1)
# u_list = [np.copy(u['g'])]
# t_list = [solver.sim_time]
while solver.proceed:
    solver.step(init_timestep)
    if solver.iteration % 10 == 0:
        logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, init_timestep))
    if solver.iteration % 100 == 0:
        format_string_time = f"{solver.iteration:d}"
        file_name = 'outXYZ_%s' % format_string_time
        with open(file_name, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(np.transpose(x), np.transpose(h['g']), np.transpose(q['g'])))

# Plot
# plt.figure(figsize=(6, 4))
# plt.pcolormesh(x.ravel(), np.array(t_list), np.array(u_list), cmap='RdBu_r', shading='gouraud', rasterized=True, clim=(-0.8, 0.8))
# plt.xlim(0, Lx)
# plt.ylim(0, stop_sim_time)
# plt.xlabel('x')
# plt.ylabel('t')
# plt.title(f'KdV-Burgers, (a,b)=({a},{b})')
# plt.tight_layout()
# plt.savefig('kdv_burgers.pdf')
# plt.savefig('kdv_burgers.png', dpi=200)

