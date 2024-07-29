import numpy as np
import matplotlib.pyplot as plt


def rk3(A, bvector, y0, interval, N):
    """
    Solve the IVP y' = A y + b, y(0) = y0, in the interval,
    using N steps of RK3.

    Parameters
    ----------
    A : matrix
        Partially defines ODE according to (4)
    bvector : function name returning vector
        Completes definition of ODE according to (4)
    y0 : vector
        Initial data
    interval : vector
        Interval on which solution is required
    N : int
        Number of steps

    Returns
    -------
    x : array of float
        Coordinate locations of the approximate solution
    y : array of float
        Values of approximate solution at locations x
    """


    # Add RK3 algorithm implementation here according to Task 1

    #define space the function is evaluated over
    x, dx = np.linspace(interval[0],interval[-1],N+1,retstep=True)

    #Assert basic conditions for inputs
    assert callable(bvector), "bvector must be a function"
    assert isinstance(y0, (list,np.ndarray)), "y0 must be a list numpy array"
    assert isinstance(interval, (list,np.ndarray)), "interval must be a list numpy array"
    assert isinstance(N, int), "N must be an integer"
    assert interval[0] < interval[-1], "interval must be increasing"
    assert A.shape[1] == np.shape(bvector(x[0]))[0], "The number of columns in A must be equal to the number of rows in bvector"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"


    #Create+Initialise solution matrix define space func evaluated over
    y = np.zeros((N+1,len(y0)))
    y[0,:] = y0
    
    
    
    #Loop procedure over N additional steps
    for i in range(N):
        #RK3 algorithm
            y_1 = y[i,:] + dx*(A@y[i,:] + bvector(x[i]))
            y_2 = 0.75*y[i,:] + 0.25*y_1 + 0.25*dx*(A@y_1 + bvector(x[i]+dx)) 
            y[i+1,:] = (1/3)*y[i,:] + (2/3)*y_2 + (2/3)*dx*(A@y_2 + bvector(x[i]+dx))
            
    # The first output argument x is the locations x_j at which the solution is evaluated;
    # this should be a real vector of length N + 1 covering the required interval.
    # The second output argument y should be the numerical solution approximated at the
    # locations x_j, which will be an array of size n × (N + 1).
    return x, y


def dirk3(A, bvector, y0, interval, N):
    """
    Solve the IVP y' = A y + b, y(0) = y0, in the interval,
    using N steps of DIRK3.

    Parameters
    ----------
    A : matrix
        Partially defines ODE according to (4)
    bvector : function name returning vector
        Completes definition of ODE according to (4)
    y0 : vector
        Initial data
    interval : vector
        Interval on which solution is required
    N : int
        Number of steps

    Returns
    -------
    x : array of float
        Coordinate locations of the approximate solution
    y : array of float
        Values of approximate solution at locations x
    """

    # Add DIRK3 algorithm implementation here according to Task 2

    #Create space function evaluated over
    x, h = np.linspace(interval[0],interval[-1],N+1,retstep=True)

    #Assert basic conditions for inputs
    assert callable(bvector), "bvector must be a function"
    assert isinstance(y0, (list,np.ndarray)), "y0 must be a list numpy array"
    assert isinstance(interval, (list,np.ndarray)), "interval must be a list numpy array"
    assert isinstance(N, int), "N must be an integer"
    assert interval[0] < interval[-1], "interval must be increasing"
    assert A.shape[1] == np.shape(bvector(x[0]))[0], "The number of columns in A must be equal to the number of rows in bvector"
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    
    #defining constants
    mu = 0.5*(1-1/np.sqrt(3))
    v = 0.5*(np.sqrt(3)-1)
    gamma = 3/(2*(3+np.sqrt(3)))
    lamda = 3*(1+np.sqrt(3))/(2*(3+np.sqrt(3)))  
    
    #Create identity matrix
    I = np.zeros_like(A)
    I = np.identity(A.shape[0])
    
    
    
    #Create and initialise solution matrix
    y = np.zeros((N+1,len(y0)))
    y[0,:] = y0
    

    
    #Loop procedure over N additional steps
    for i in range(N):
        y_1 = np.linalg.inv((I-h*mu*A))@(y[i,:]+ h*mu*bvector(x[i]+h*mu))
        
        y_2 = np.linalg.inv((I-h*mu*A))@(y_1 + h*v*(A@y_1+bvector(x[i]+h*mu))+h*mu*bvector(x[i]+h*v+2*h*mu))
        
        y[i+1,:] = ((1-lamda)*y[i,:])+(lamda*y_2)+h*gamma*((A@y_2)+bvector(x[i]+(h*v)+(2*h*mu)))
    
        
        
        
    
    
    


    # The first output argument x is the locations x_j at which the solution is evaluated;
    # this should be a real vector of length N + 1 covering the required interval.
    # The second output argument y should be the numerical solution approximated at the
    # locations x_j, which will be an array of size n × (N + 1).
    return x, y



def bvector_system_1(x):
    """
    b vector definition in Task 3.

    Parameters
    ----------
    x : float
        Coordinate

    Returns
    -------
    b : array of float
        Just zeros in this case
    """
    #Assert basic conditions for inputs
    assert isinstance(x, (float,np.ndarray)), "x must be a float or numpy array"
    # Define vector b according to equation (5)
    b=np.zeros(2)

    # Return vector b
    return b


def bvector_system_2(x):
    """
    b vector definition in Task 4.

    Parameters
    ----------
    x : float
        Coordinate

    Returns
    -------
    b : array of float
        b as given by equation (13)
    """
    #Assert basic conditions for inputs
    assert isinstance(x, (float,np.ndarray)), "x must be a float or numpy array"
    # Define vector b according to equation (13)
    b1 = np.cos(10*x)-10*np.sin(10*x)
    b2 = 199*np.cos(10*x)-10*np.sin(10*x)
    b3 = 208*np.cos(10*x)+10000*np.sin(10*x)
    b = np.array([b1,b2,b3])

    # Return vector b
    return b



###Q1
"""Question 1 already completed as function for RK3 defined"""

###Q2
'''Question 2 already completed as function for DIRK3  defined'''

###Q3

#Define values a1 and a2 for matrix A
a1 = 1000
a2 = 1
A = np.array([[-a1,0],[a1,-a2]])

#Define initial conditions
y0 = np.array([1,0])

#Define exact solution for y
def y_exact(x):
    """Computes the exact solution of the system of ODEs in Task 3.
        Parameters
        ----------
        x : float
            Coordinate
        Returns
        -------
        y_exact : array of float
            Exact solution of the system of ODEs in Task 3"""
    #Assert basic conditions for inputs
    assert isinstance(x, (np.ndarray)), "x must be a float or numpy array"

    #Define exact solution for y
    y_exact_comp_1 = np.exp(-a1*x)
    y_exact_comp_2 = (a1/(a1-a2))*(np.exp(-a2*x)-np.exp(-a1*x))
    y_exact = np.array([y_exact_comp_1,y_exact_comp_2])
    return y_exact

#Define function for norm error
def norm_error(y_true,y_approx,h,N):
    """Computes the norm error of the system of ODEs in Task 3.
        Parameters
        ----------
        y_true : array of float
            Exact solution of the system of ODEs 
        y_approx : array of float
            Approximate solution of the system of ODEs 
        h : float
            Step size
        N : int
            Number of steps
        Returns
        -------
        error : float
            Norm error of the system of ODEs"""
    
    #Assert basic conditions for inputs
    assert isinstance(y_true, (list,np.ndarray)), "y_true must be a list or numpy array"
    assert isinstance(y_approx, (list,np.ndarray)), "y_approx must be a list or numpy array"
    assert isinstance(h, float), "h must be a float"
    assert isinstance(N, int), "N must be an integer"
    assert np.shape(y_true.T)==np.shape(y_approx), "y_true and y_approx must be the same shape"
    
    summation_list = []
    for n in range(1,N+1):
        mod=np.abs((y_true[-1][n]-y_approx[n][-1])/y_true[-1][n])
        summation_list.append(mod)
    summation = np.sum(summation_list)
    error = h*summation
    return error




#define interval for solution to be evaluated over
interval = np.array([0,0.1])

#Create empty list for solution matrices and intervals rk3 & dirk3
x_space_rk3 = []
x_space_dirk3 = []

rk3_sol_matrices = []
dirk3_sol_matrices=[]


h_list = []
error_list_rk3 = []
error_list_dirk3 = []

#Loop over k values
#Exclude k=1 as too few steps to give reasonable error
for k in range(2,11):
    
    #Solution matrices 
    xrk3, rk3_sol_matrix = rk3(A,bvector_system_1,y0,interval,40*k)
    xdirk3, dirk3_sol_matrix = dirk3(A,bvector_system_1,y0,interval,40*k)
    yexact = y_exact(xrk3)
    
    #Append N value for later plotting
    
    #Calculate h and append to list
    h = (interval[1] - interval[0]) / (40*k)
    h_list.append(h)
    
    #Calculate error and append to list
    error_rk3 = norm_error(yexact, rk3_sol_matrix, h, 40*k)
    error_list_rk3.append(error_rk3)

    error_dirk3 = norm_error(yexact, dirk3_sol_matrix, h, 40*k)
    error_list_dirk3.append(error_dirk3)

    
    #Append to list to access later
    x_space_rk3.append(xrk3)
    x_space_dirk3.append(xdirk3)
    
    rk3_sol_matrices.append(rk3_sol_matrix)
    dirk3_sol_matrices.append(dirk3_sol_matrix)
    
   
    


def Q3_plots_1_and_2():
    """Plots the error against h for RK3 and DIRK3
        Parameters
        ----------
        None
        Returns
        -------
        None"""
    #Fit a polynomial of degree 3 to the error
    coefficients_rk3 = np.polyfit((h_list), error_list_rk3, 3)
    coefficients_dirk3 = np.polyfit((h_list), error_list_dirk3, 3)


    
    # Create a polynomial function from the coefficients
    polynomial_rk3 = np.poly1d(coefficients_rk3)
    polynomial_dirk3 = np.poly1d(coefficients_dirk3)
    
    # Plot the polynomial function
    fitted_error_rk3 = polynomial_rk3(h_list)
    fitted_error_dirk3 = polynomial_dirk3(h_list)

    #Calculate gradient for rk3
    m=(((np.log(error_list_rk3[0])-np.log(error_list_rk3[-1]))/(np.log(h_list[0])-np.log(h_list[-1]))))

    #Plot the fitted polynomial against the error
    fig1, axs1 = plt.subplots()
    #Logarithmic to show 3rd order convergence of RK3 through gradient
    axs1.loglog(h_list, error_list_rk3, 'o', label='RK3')
    axs1.loglog(h_list, fitted_error_rk3, label=f'Fitted RK3 with m={np.round(m,2)}')
    axs1.set_title('Logarithmic RK3 with 3rd order fitted polynomial (Q3)')
    axs1.set_xlabel('log(h)')
    axs1.set_ylabel('log(Error)')
    axs1.legend()

    #Calculate m for dirk3
    m=(((np.log(error_list_dirk3[0])-np.log(error_list_dirk3[-1]))/(np.log(h_list[0])-np.log(h_list[-1]))))

    fig1, axs1 = plt.subplots()
    axs1.loglog(h_list, error_list_dirk3, 'o', label='DIRK3')
    axs1.loglog(h_list, fitted_error_dirk3, label=f'Fitted DIRK3 with m={np.round(m,2)}')
    axs1.set_title('Lograthmic DIRK3 with 3rd order fitted polynomial (Q3)')
    axs1.set_xlabel('log(h)')
    axs1.set_ylabel('log(Error)')
    axs1.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def Q3_plots_3():
    """Plots the solutions for N=400 for RK3
        Parameters
        ----------
        None
        Returns
        -------
        None"""
    #Plot the solutions for N=400 for RK3
    fig2, axs2 = plt.subplots(1,2)
    #Use semilog to show fast drop off of y1
    axs2[0].semilogy(x_space_rk3[-1], rk3_sol_matrices[-1][:,0], label='Approximate solution')
    axs2[0].semilogy(x_space_rk3[-1], yexact[0],color='orange',ls='--', label='Exact solution')
    axs2[0].set_title('RK3 solution of y1 for N=400')
    axs2[0].set_xlabel('x')
    axs2[0].set_ylabel('log(y1)')
    axs2[0].legend()

    axs2[1].plot(x_space_rk3[-1], rk3_sol_matrices[-1][:,1], label='Approximate solution')
    axs2[1].plot(x_space_rk3[-1], yexact[1],ls='--',color='orange', label='Exact solution')
    axs2[1].set_title('RK3 solution of y2 for N=400 (Q3)')
    axs2[1].set_xlabel('x')
    axs2[1].set_ylabel('y2')
    axs2[1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle('RK3 solution of y1 and y2 for N=400 (Q3)')
    plt.show()

def Q3_plots_4():
    """Plots the solutions for N=400 for DIRK3
        Parameters
        ----------
        None
        Returns
        -------
        None"""
    
    #Plot the solutions for N=400 for DIRK3
    fig3, axs3 = plt.subplots(1,2)
    #Use semilog to show fast drop off of y1
    axs3[0].semilogy(x_space_dirk3[-1], dirk3_sol_matrices[-1][:,0], label='Approximate solution')
    axs3[0].semilogy(x_space_dirk3[-1], yexact[0],color='orange',ls='--', label='Exact solution')
    axs3[0].set_title('DIRK3 solution of y1 for N=400 (Q3)')
    axs3[0].set_xlabel('x')
    axs3[0].set_ylabel('log(y1)')
    axs3[0].legend()

    axs3[1].plot(x_space_dirk3[-1], dirk3_sol_matrices[-1][:,1], label='Approximate solution')
    axs3[1].plot(x_space_dirk3[-1], yexact[1],ls='--',color='orange', label='Exact solution')
    axs3[1].set_title('DIRK3 solution of y2 for N=400')
    axs3[1].set_xlabel('x')
    axs3[1].set_ylabel('y2')
    axs3[1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle('DIRK3 solution of y1 and y2 for N=400 (Q3)')
    plt.show()


    

    
###Q4

#Define initial values 
y_initial_Q4 = np.array([0,1,0])

#Define matrix A
A_Q4 = np.array([[-1,0,0],[-99,-100,0],[-10098,9900,-10000]])

#Create function for exact solution
def y_exact_Q4(x):
    """Computes the exact solution of the system of ODEs in Task 4.
        Parameters
        ----------
        x : float
            Coordinate
        Returns
        -------
        y_exact : array of float
            Exact solution of the system of ODEs in Task 4"""
    #Assert basic conditions for inputs
    assert isinstance(x, (np.ndarray)), "x must be a float or numpy array"
    y_exact_comp_1 = np.cos(10*x)-np.exp(-x)
    y_exact_comp_2 = np.cos(10*x)+np.exp(-x)-np.exp(-100*x)
    y_exact_comp_3 = np.sin(10*x)+2*np.exp(-x)-np.exp(-100*x)-np.exp(-10000*x)
    y_exact = np.array([y_exact_comp_1,y_exact_comp_2,y_exact_comp_3])
    return y_exact

#interval for solution to be evaluated over
interval = np.array([0,1])
#Create empty list for solution matrices and intervals rk3 & dirk3
x_space_rk3_Q4 = []
x_space_dirk3_Q4 = []

rk3_sol_matrices_Q4 = []
dirk3_sol_matrices_Q4=[]

#Define empty lists for h and error
h_list_Q4 = []
error_list_dirk3_Q4 = []

#Loop over k values
#Exclude k=4 as too few steps to give reasonable error
for k in range(5,17):

    #Use RK3 and DIRK3 algorithms for Q4
    xrk3_Q4, rk3_sol_matrix_Q4 = rk3(A_Q4,bvector_system_2,y_initial_Q4,interval,200*k)
    xdirk3_Q4, dirk3_sol_matrix_Q4 = dirk3(A_Q4,bvector_system_2,y_initial_Q4,interval,200*k)

    #Append to list to access later
    x_space_rk3_Q4.append(xrk3_Q4)
    x_space_dirk3_Q4.append(xdirk3_Q4)

    rk3_sol_matrices_Q4.append(rk3_sol_matrix_Q4)
    dirk3_sol_matrices_Q4.append(dirk3_sol_matrix_Q4)
    
    #Calculate exact solution based on x values
    yexact_Q4 = y_exact_Q4(xdirk3_Q4)

    #Calculate h and append to list
    h_Q4 = (interval[1] - interval[0]) / (200*k)
    h_list_Q4.append(h_Q4)

    #Calculate error and append to list
    error_dirk3_Q4 = norm_error(yexact_Q4, dirk3_sol_matrix_Q4, h_Q4, 200*k)
    error_list_dirk3_Q4.append(error_dirk3_Q4)
print("The statements printed above state that the RK3 algorithm has failed to converge to the exact solution for N=3200. This is because the algorithm is unstable and the solution is diverging.")

def Q4_plots_5():
    """Plots the error against h for RK3 and DIRK3
        Parameters
        ----------
        None
        Returns
        -------
        None"""
    #Plot the error against h for DIRK3
    #Logarithmic to show 3rd order convergence of RK3 through gradient
    plt.loglog(h_list_Q4, error_list_dirk3_Q4, 'o', label='DIRK3')
    plt.xlabel('log(h)')
    plt.ylabel('log(Error)')

    #Fit a polynomial of degree 1 to the error
    coefficents = np.polyfit(np.log(h_list_Q4), np.log(error_list_dirk3_Q4), 1)
    polynomial = np.poly1d(coefficents)
    fitted_error = np.exp(polynomial(np.log(h_list_Q4)))
    plt.loglog(h_list_Q4, fitted_error, ls='--', color='orange', label=f'Fitted DIRK3 with m={np.round(coefficents[0],2)}')
    plt.legend()
    
    plt.title('DIRK3 error for Q4')
    plt.show()

def Q4_plots_6():
    """Plots the solutions for N=3200 for RK3 and DIRK3
        Parameters
        ----------
        None
        Returns
        -------
        None"""
    #Plot all components of y for DIRK3
    fig, axs = plt.subplots(3,1)
    #y1
    axs[0].plot(x_space_dirk3_Q4[-1], dirk3_sol_matrices_Q4[-1][:,0], label='Approximate solution')
    axs[0].plot(x_space_dirk3_Q4[-1], yexact_Q4[0],ls='--',color='orange', label='Exact solution')
    axs[0].set_title('DIRK3 solution of y1')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y1')
    axs[0].legend(loc='upper right',fontsize='small')
    #y2
    axs[1].plot(x_space_dirk3_Q4[-1], dirk3_sol_matrices_Q4[-1][:,1], label='Approximate solution')
    axs[1].plot(x_space_dirk3_Q4[-1], yexact_Q4[1],ls='--',color='orange', label='Exact solution')
    axs[1].set_title('DIRK3 solution of y2')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y2')
    axs[1].legend(loc='upper right',fontsize='small')
    #y3
    axs[2].plot(x_space_dirk3_Q4[-1], dirk3_sol_matrices_Q4[-1][:,2], label='Approximate solution')
    axs[2].plot(x_space_dirk3_Q4[-1], yexact_Q4[2],ls='--',color='orange', label='Exact solution')
    axs[2].set_title('DIRK3 solution of y3')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y3')
    axs[2].legend(loc='upper right',fontsize='small')
    

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle('DIRK3 solution of y1, y2 and y3 for N=3200 (Q4)')
    plt.show()

def Q4_plots_7():
    """Plots the solutions for N=3200 for RK3 and DIRK3
        Parameters
        ----------
        None
        Returns
        -------
        None"""
    #Plot all components of Y for RK3
   
    #y1
    fig, axs = plt.subplots(3,1)
    axs[0].plot(x_space_rk3_Q4[-1], rk3_sol_matrices_Q4[-1][:,0], label='Approximate solution')
    axs[0].plot(x_space_rk3_Q4[-1], yexact_Q4[0],ls='--',color='orange', label='Exact solution')
    axs[0].set_title('RK3 solution of y1')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y1')
    axs[0].legend(loc='upper right',fontsize='small')

    #y2
    axs[1].plot(x_space_rk3_Q4[-1], rk3_sol_matrices_Q4[-1][:,1], label='Approximate solution')
    axs[1].plot(x_space_rk3_Q4[-1], yexact_Q4[1],ls='--',color='orange', label='Exact solution')
    axs[1].set_title('RK3 solution of y2')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y2')
    axs[1].legend(loc='upper right',fontsize='small')

    #y3
    axs[2].plot(x_space_rk3_Q4[-1], rk3_sol_matrices_Q4[-1][:,2], label='Approximate solution')
    axs[2].plot(x_space_rk3_Q4[-1], yexact_Q4[2],ls='--',color='orange', label='Exact solution')
    axs[2].set_title('RK3 solution of y3')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y3')
    axs[2].legend(loc='upper right',fontsize='small')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle('RK3 solution of y1, y2 and y3 for N=3200 (Q4)')
    plt.show()   


    

    
#Run functions- comment out to run specific functions
Q3_plots_1_and_2()
Q3_plots_3()
Q3_plots_4()
Q4_plots_5()
Q4_plots_6()
Q4_plots_7()



    
    

    











