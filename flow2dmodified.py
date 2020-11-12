import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from scipy.optimize import brentq, newton
plt.close('all')

# model_case 1 = Medio
# model_case 2 = vdP
# model_case 3 = Fitzhugh-Nagumo
model_case = 2

def solve_flow(param,lim = [-3,3,-3,3],max_time=10.0):

    if model_case == 1:
# Medio 2D flow 
        def flow_deriv(x_y, t0, a,b,c,alpha):
        #"""Compute the time-derivative of a Medio system."""
            x, y = x_y
            return [a*y + b*x*(c - y**2),-x+alpha]
        model_title = 'Medio Economics'

    elif model_case == 2:
# van der pol 2D flow 
        def flow_deriv(x_y, t0, alpha,beta):
        #"""Compute the time-derivative of a Medio system."""
            x, y = x_y
            return [y,-alpha*x+beta*(1-x**2)*y]
        model_title = 'van der Pol Oscillator'

    else:
# Fitzhugh-Nagumo
        def flow_deriv(x_y, t0, alpha, beta, gamma):
        #"""Compute the time-derivative of a Medio system."""
            x, y = x_y
            return [y-alpha,-gamma*x+beta*(1-y**2)*y]
        model_title = 'Fitzhugh-Nagumo Neuron'



    plt.figure()
    xmin = lim[0]
    xmax = lim[1]
    ymin = lim[2]
    ymax = lim[3]
    plt.axis([xmin, xmax, ymin, ymax])


    N=144
    colors = plt.cm.prism(np.linspace(0, 1, N))
    
    x0 = np.zeros(shape=(N,2))
    ind = -1
    for i in range(0,12):
        for j in range(0,12):
            ind = ind + 1;
            x0[ind,0] = ymin-1 + (ymax-ymin+2)*i/11
            x0[ind,1] = xmin-1 + (xmax-xmin+2)*j/11
             
    # Solve for the trajectories
    t = np.linspace(0, max_time, int(250*max_time))
    x_t = np.asarray([integrate.odeint(flow_deriv, x0i, t, param)
                      for x0i in x0])

    for i in range(N):
        x, y = x_t[i,:,:].T
        lines = plt.plot(x, y, '-', c=colors[i])
        plt.setp(lines, linewidth=1)

    plt.show()
    plt.title(model_title)
    plt.savefig('Flow2D')
    
    return t, x_t


def solve_flow2(param,max_time=20.0):

    if model_case == 1:
# Medio 2D flow 
        def flow_deriv(x_y, t0, a,b,c,alpha):
        #"""Compute the time-derivative of a Medio system."""
            x, y = x_y
            return [a*y + b*x*(c - y**2),-x+alpha]
        model_title = 'Medio Economics'
        x0 = np.zeros(shape=(2,))
        x0[0] = 1
        x0[1] = 1
    elif model_case == 2:
# van der pol 2D flow 
        def flow_deriv(x_y, t0, alpha,beta):
        #"""Compute the time-derivative of a Medio system."""
            x, y = x_y
            return [y,-alpha*x+beta*(1-x**2)*y]
        model_title = 'van der Pol Oscillator'
        x0 = np.zeros(shape=(2,))
        x0[0] = 0
        x0[1] = 4.5

    else:
# Fitzhugh-Nagumo
        def flow_deriv(x_y, t0, alpha, beta, gamma):
        #"""Compute the time-derivative of a Medio system."""
            x, y = x_y
            return [y-alpha,-gamma*x+beta*(1-y**2)*y]
        model_title = 'Fitzhugh-Nagumo Neuron'
        x0 = np.zeros(shape=(2,))#[0.,0.]
        x0[0] = 1
        x0[1] = 1

             
    # Solve for the trajectories
    t = np.linspace(0, max_time, int(250*max_time))
    x_t = integrate.odeint(flow_deriv, x0, t, param)
    
def poin():
    time=[]
    for i,x in enumerate(x_t):
        if x[0]>=-0.01 and x[0]<=0.01 and x[1]<0:
            time.append(i/250)
    return t, x_t, time
print (poin())
if 

if model_case == 1:
    param = (0.9,0.7,0.5,0.6)    # Medio
    lim = (-7,7,-5,5)
elif model_case == 2:
    param = (5, 2.5)             # van der Pol
    lim = (-7,7,-10,10)
else:
    param = (0.02,0.5,0.2)        # Fitzhugh-Nagumo
    lim = (-7,7,-4,4)

t, x_t = solve_flow(param,lim)


t, x_t, time = solve_flow2(param)
plt.figure(2)
lines = plt.plot(t,x_t[:,1],'-')
lines = plt.plot(t,x_t[:,0],'-')
plt.axis(([0, 20, -7, 7]))

plt.figure(1)
lines = plt.plot(x_t[:,0],x_t[:,1],'ko',ms=1)
plt.setp(lines, linewidth=0.5)
plt.show()


