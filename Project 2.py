
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.integrate
import scipy.optimize

#hair angles and force magnitudes
theta_0_list = np.linspace(0,np.pi,20)
f_g = 0.1
f_x = 0
def rhs_func(s,y):
    """
    Define the ODEs for the hair model

    Parameters
    ----------
    s : scalar
        The independent variable.
    y : vector
        The dependent variables.
    
    Returns
    -------
    dyds : vector
        The derivatives of the dependent variables.
    """
    #describe coupled ODEs
    #dtheta/ds = psi
    theta = y[0]
    psi = y[1]

    dyds = np.zeros_like(y)

    dyds[0] = psi
    dyds[1] = (s*f_g*np.cos(theta) + s*f_x*np.sin(theta))

    return dyds


#Define the boundary conditions for theta
R=10
L=4
# We are trying to find the value of psi_0 that will give us the correct value of theta_0


def shot(theta_0,psi_0):
    y0 = np.array([theta_0,psi_0])
    interval = [0,L]
    s, ds = np.linspace(interval[0],interval[1],100,retstep=True)
    
    sol = scipy.integrate.solve_ivp(rhs_func,interval,y0,max_step=ds,dense_output=True)
    
    x=s
    y = sol.sol(x)
    

    
    return x,y


def shot_error(psi_0_guess):
    x,y = shot(theta_0,psi_0_guess)
    phi = y[1,-1] 
    return phi


psi_0_list = []
theta_list=[]
s_list=[]
x=[]
z=[]
#Plot the solution for theta for different values of theta_0
for i in range(len(theta_0_list)):

    theta_0 = theta_0_list[i]
    psi_0 = scipy.optimize.brentq(shot_error,-20,20)
    psi_0_list.append(psi_0)
    s,y = shot(theta_0,psi_0)
    ds = s[1]-s[0]
    theta = y[0,:]
    theta_list.append(theta)
    s_list.append(s)

    #empty arrays to be filled with coords for individual hair
    x_each_hair=np.zeros_like(theta)
    z_each_hair=np.zeros_like(theta)

    #initial x and z coords
    x_0 = R*np.cos(theta_0)
    z_0 = R*np.sin(theta_0)
    x_each_hair[0] = x_0
    z_each_hair[0] = z_0
    #Use Euler method to solve for x and z
    for j in range(1,len(theta)):
        x_each_hair[j] = x_each_hair[j-1] + ds*np.cos(theta[j])
        z_each_hair[j] = z_each_hair[j-1] + ds*np.sin(theta[j])


    
        
    #append to array that contains values for all hairs
    x.append(x_each_hair)
    z.append(z_each_hair)

#Plot each hair
def plot_hair(x,z):
    for i in range(len(x)):
        plt.plot(x[i],z[i],label=f'hair {i}')
        circle=patches.Circle((0,0),R,color='black',fill=False)
        eye1 = patches.Circle((-R/2, R/4), R/10, color='red',fill=False)
        eye2 = patches.Circle((R/2, R/4), R/10, color='red')
        eyebrow1 = patches.Ellipse((-R/2, R/2), R/2, R/8, angle=30, color='black')
        eyebrow2 = patches.Ellipse((R/2, R/2), R/2, R/8, angle=-30, color='black')
        mouth = patches.Wedge((0, -R/2), R/2, 30, 150, width=R/8, color='black')
        
        plt.gca().add_patch(eyebrow1)
        plt.gca().add_patch(eyebrow2)

        plt.gca().add_patch(mouth)
        

        plt.gca().add_patch(eye1)
        plt.gca().add_patch(eye2)
        plt.gca().add_patch(circle)
    
        
        # Draw the smiley face
  
        
    plt.show()

plot_hair(x,z)


        

    



#Calculate the value of psi_0 that gives us the correct value of theta_0
#psi_0 = scipy.optimize.brentq(shot_error,-20,20)
#plt.plot(psi_values,phi)
#print(psi_0)

#Now solve for psi and theta
#x,y = shot(theta_0,psi_0)
#theta = y[0,:]
#psi = y[1,:]
#plt.plot(x,theta)   
#plt.show()
