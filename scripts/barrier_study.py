import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_la
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib import rc
#rc('text', usetex=True)
#rc('font', family='serif')
# for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})


from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d

"""
Barrier fracture case
- discontinuous matrix pressure
- separate communication of the fracture to the lowwer and the upper part

- evaluation of the analytical solution
- parametric study for comparison to the numerical solution via. finite differences
"""
class TripletMatrix:
    def __init__(self, N, M):
        self.shape = (N, M)
        self.i = []
        self.j = []
        self.v = []

    def __iadd__(self, triplet):
        """
        :param triplet: (i,j,v)
        :return:
        """
        self.add(triplet)
        return self


    def add(self, triplet):
        i,j,v = triplet
        self.i.append(i)
        self.j.append(j)
        self.v.append(v)


    def make_csr(self):
        coo = sparse.coo_matrix((self.v, (self.i, self.j)))
        csr = coo.tocsr()
        return csr

class BarrierFracture:

    def __init__(self, k1, k2, sigma, P1, P2, n_terms):
        self.k1 = k1
        self.k2 = k2        # tuple
        self.sigma = sigma   # tuple
        self.P1 = P1
        self.P2 = P2         # tuple
        self.n_terms = n_terms
        ###
        self.precompute_analytical()

    # def sinh(self, x):
    #     return 1 - np.exp( -2.0 * x)
    #     #return np.sinh(x)
    #
    # def cosh(self, x):
    #     return 1 + np.exp( -2.0 * x)
    #     #return np.cosh(x)

    def precompute_analytical(self):
        avg_sigma = self.sigma[0] + self.sigma[1]
        self.avg_k = np.sqrt(self.k1 / avg_sigma)
        self.avg_P2 = (self.sigma[0]*self.P2[0] + self.sigma[1] *self.P2[1]) / avg_sigma


        #n_err_terms = int(0.2*self.n_terms)
        n_err_terms = self.n_terms
        n_terms = self.n_terms + n_err_terms
        nn = self.n_series = np.arange(1, n_terms+1, 1.0)

        # Solve a_n systems
        self.an = np.empty((n_terms, 2))
        self.un = np.empty((n_terms,))
        X_mat = np.empty((2, 2))
        y_vec = np.empty((2,))
        for i in range(n_terms):
            n = i + 1
            sign = int(2* (i % 2) - 1)

            ############### an system and un
            cosh_npi = (1 + np.exp(-2.0 * n * np.pi))
            sinh_npi = (1 - np.exp(-2.0 * n * np.pi))
            sqr_npik_one = (1.0 +  (n * np.pi * self.avg_k)**2.0)
            diagonal_p = n * np.pi * self.k2[0] / self.sigma[0] * cosh_npi + sinh_npi
            diagonal_m = n * np.pi * self.k2[1] / self.sigma[1] * cosh_npi + sinh_npi
            off_diagonal = sinh_npi /avg_sigma / sqr_npik_one

            y_vec[0] = y_vec[1] = -sign * 2 * self.avg_k * np.sinh(1 / self.avg_k) / sqr_npik_one
            X_mat[0, 0] = diagonal_p - self.sigma[0] * off_diagonal
            X_mat[1, 1] = diagonal_m - self.sigma[1] * off_diagonal
            X_mat[0, 1] = -self.sigma[1] * off_diagonal
            X_mat[1, 0] = -self.sigma[0] * off_diagonal

            self.an[i, :] = la.solve(X_mat, y_vec)
            avg_an = (self.sigma[0] * self.an[i, 0] + self.sigma[1] * self.an[i, 1])/ avg_sigma
            self.un[i] = avg_an * sinh_npi / sqr_npik_one


        u_sum = -np.sum(np.abs(self.un))

        ### B0 system
        T = self.avg_k * np.sinh(1/self.avg_k) / ( np.cosh(1/self.avg_k) - u_sum)
        y_vec[0] = y_vec[1] = - self.avg_P2 +(self.avg_P2 - self.P1) * T
        y_vec[0] += self.P2[0]
        y_vec[1] += self.P2[1]
        X_mat[0, 0] = self.k2[0]/self.sigma[0] + 1 + (T-1)*self.sigma[0] / avg_sigma
        X_mat[1, 1] = self.k2[1]/self.sigma[1] + 1 + (T-1)*self.sigma[1] / avg_sigma
        X_mat[0, 1] = (T-1) * self.sigma[1] / avg_sigma
        X_mat[1, 0] = (T-1) * self.sigma[0] / avg_sigma

        self.B0 = la.solve(X_mat, y_vec)
        #print(self.B0)
        self.avg_B0 = (self.sigma[0] * self.B0[0] + self.sigma[1] * self.B0[1])/ avg_sigma
        #print(np.cosh(1/self.avg_k), u_sum, self.P1 - self.avg_P2)
        self.u0 = (self.P1 - self.avg_P2 + self.avg_B0) / (np.cosh(1/self.avg_k) - u_sum)

        # an_err = np.sum(self.an[self.n_terms:] )
        #
        # print("un_err: ", u_sum_err, "an_err:", an_err)
        # print("U:", u_sum)
        # # print("alt_u0:", alt_u0)
        # # assert np.isclose(self.u0, alt_u0)
        #
        self.vec_eval_p2 = np.vectorize(self.eval_p2)
        self.vec_eval_p1 = np.vectorize(self.eval_p1)


    def eval_p2(self, x, y):
        if y > 0:
            idx = 0
        else:
            idx = 1

        y = np.abs(y)

        pi_x = np.pi * x
        series_sum = np.sum(self.an[:,idx] * np.cos(pi_x * self.n_series)
                            * (np.exp( (-y * np.pi) * self.n_series ) - np.exp( ((y-2.0)*np.pi) * self.n_series ) ) )

        p2_sum = self.P2[idx] + self.B0[idx] * (y - 1) - self.u0 * series_sum
        return p2_sum

    def eval_p1(self, x):
        #x = np.abs(x)
        pi_x = np.pi * x
        series_sum = np.sum(self.un * np.cos(pi_x * self.n_series))
        p1_sum = self.avg_P2 - self.avg_B0 + self.u0 * np.cosh(x  / self.avg_k) - self.u0 *  series_sum
        return p1_sum

    def plot_p1(self, ax, x):
        f_p1 = np.vectorize(self.eval_p1)
        p1 = f_p1(x)
        ax.plot(x, p1, label="p1an")
        return p1

    def plot_p2(self, ax, x, y):
        f_p2 = np.vectorize(self.eval_p2)
        p2 = f_p2(x, y)
        ax.plot(x, p2, label="p2an, y={:8.2f}".format(y))
        return p2



    def form_fd_matrix(self, nx, ny):
        """
        :param nx: Number of intervals in X direction
        :param ny: Number of intervals in Y direction
        :return:
        """
        k2 = self.k2
        k1 = self.k1
        sigma = self.sigma
        print(k2,sigma,k1)

        nx = int(nx/2)       # we compute only half of domain
        ny += int(ny % 2)    # Make ir even

        ny_half = int(ny/2)
        iy_dirichlet = [0, ny+2]
        iy_bc_frac = [ny_half, ny_half+2]
        iy_frac = ny_half + 1


        Nx = nx+1
        Ny = ny+1
        dx = 1.0/nx
        dy = 2.0/ny
        dx2 = 2*dx
        dy2 = 2*dy
        ddx = dx*dx
        ddy = dy*dy

        n_dofs = Nx*(Ny+2)
        A_vals = TripletMatrix(n_dofs, n_dofs)
        b = np.zeros((n_dofs,))

        def regular_line_2d(iy, side):
            k = iy * Nx

            # left zero Neumann
            # O(dx^2) formula: (-3*u_0 + 4*u_1 - u_2) / (2*dx)
            A_vals.add((k, k, -3.0 / dx2))
            A_vals.add((k, k + 1, 4 / dx2))
            A_vals.add((k, k + 2, -1 / dx2))
            k += 1

            # regular points
            for j in range(1, nx):
                A_vals.add((k, k, 2 * k2[side] / ddx + 2 * k2[side] / ddy))
                A_vals.add((k, k - 1, -k2[side] / ddx))
                A_vals.add((k, k + 1, -k2[side] / ddx))
                A_vals.add((k, k - Nx, -k2[side] / ddy))
                A_vals.add((k, k + Nx, -k2[side] / ddy))
                k += 1

            # right zero Neumann
            # O(dx^2) formula: (-3*u_0 + 4*u_1 - u_2) / (2*dx)
            A_vals.add((k, k, -3.0 / dx2))
            A_vals.add((k, k - 1, 4 / dx2))
            A_vals.add((k, k - 2, -1 / dx2))
            k += 1

        ###########################

        # Dirichlet top, bottom
        for side  in [0, 1]:   # top, bottom
            k = iy_dirichlet[side]*Nx
            for j in range(0, Nx):
                A_vals.add((k, k, 1.0))
                b[k] = self.P2[side]
                k += 1

        # regular lines top
        for iy in range(iy_dirichlet[0]+1, iy_bc_frac[0]):
            regular_line_2d(iy, side=0)

        # regular lines bottom
        for iy in range(iy_bc_frac[1]+1, iy_dirichlet[1]):
            regular_line_2d(iy, side=1)


        k_frac_shift = [Nx, -Nx]

        # Top and bottom BC with fracture
        for side in [0,1]:
            k = iy_bc_frac[side] * Nx
            ks = k_frac_shift[side]

            # Fracture BC, Neumman:
            # O(dy^2) formula: (-3*u_0 + 4*u_1 - u_2) / (2*dy)
            # A_vals.add((k, k,       sigma * (-3.0) / dx2))
            # A_vals.add((k, k + 1,   sigma * (4.0) / dx2))
            # A_vals.add((k, k + 2,   sigma * (-1.0) / dx2))

                    #k+=1
            for ix in range(0, Nx):
                # Neumann 2. order
                A_vals.add((k, k,           k2[side] * (-3.0) / dy2 - sigma[side] ))
                A_vals.add((k, k + ks,      +sigma[side]))
                A_vals.add((k, k - 1*ks,    k2[side] * 4 / dy2))
                A_vals.add((k, k - 2*ks,    k2[side] * (-1) / dy2))
                #A_vals.add((k, k + 1*Nx,    k2[side] * 4 / dy2))
                #A_vals.add((k, k + 2*Nx,    k2[side] * (-1) / dy2))
                k += 1

        # Fracture equation
        k = iy_frac * Nx
        # Left zero Nemmann
        A_vals.add((k, k, -3.0 / dx2))
        A_vals.add((k, k + 1, 4 / dx2))
        A_vals.add((k, k + 2, -1 / dx2))
        k += 1

        # regular points
        print(ddx)
        for j in range(1,nx):
            A_vals.add((k, k, 2 * k1 / ddx + sigma[0] + sigma[1]))
            A_vals.add((k, k - 1, -k1 / ddx))
            A_vals.add((k, k + 1, -k1 / ddx))
            A_vals.add((k, k - Nx, - sigma[0]))
            A_vals.add((k, k + Nx, - sigma[1]))
            k += 1

        # Right Dirichlet
        A_vals.add((k, k, 1.0))
        b[k] = self.P1

        return (A_vals.make_csr(), b)


    def solve_fd(self, nx, ny):
        self.fd_shape = (nx, ny)
        A, b = self.form_fd_matrix(nx, ny)
        print("A shp: ", A.shape)
        self.dof_values = sp_la.spsolve(A, b)

        # dof_values - have dofs only for right side
        ny_half = int(ny/2)
        Ny = ny + 1
        Nx = int(nx/2) + 1
        x = np.linspace(-1, 1, (nx+1))
        y = np.empty(Ny+1, dtype=float)
        y[0:ny_half + 1] = np.linspace(1, 0, (ny_half + 1))
        y[ny_half + 1:] = np.linspace(0, -1, (ny_half + 1))
        y_epsilon = 1e-16
        y[ny_half] = y_epsilon
        y[ny_half+1] = -y_epsilon

        iy_2d = np.concatenate( (np.arange(0, ny_half+1, 1),
                                 np.arange(ny_half+2, Ny + 2, 1)) )
        iy_1d = ny_half+1

        value_matrix = self.dof_values.reshape((ny+3, Nx))
        right_side_2d = value_matrix[iy_2d, :]
        left_side_2d = right_side_2d[:, -1: 0: -1]
        p2_mat = np.concatenate((left_side_2d, right_side_2d), axis = 1)

        r_vec = value_matrix[iy_1d, :]
        l_vec = r_vec[-1: 0: -1]
        p1_vec = np.concatenate( (l_vec, r_vec) )

        return (x,y, p1_vec, p2_mat)


def add_text(ax, data, text, shift):
    y = data[int(len(data)/2)]
    ax.text(0.0, y+shift, text, horizontalalignment='center')




def plot_solution():
    """
    Plot both analytical and FD solution. and errors
    :return:
    """
    ac = BarrierFracture(k1=0.5, k2=(5,2), sigma=(20,10), P1=0, P2=(10, -10), n_terms=1000)
    #sigma = 10**(-0.125)
    #k2 = 10**(-2.625)
    #sigma_values = (2*sigma ** 0.5, sigma)
    #k2_values = (k2, k2**0.5 )

    #ac = BarrierFracture(k1=1, k2=k2_values, sigma=sigma_values, P1=0, P2=(10, -10), n_terms=100)

    nx,  ny = 200, 200
    x, y, p1_fd, p2_fd = ac.solve_fd(nx, ny)

    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(121)
    ax_err = fig.add_subplot(122)




    norm = matplotlib.colors.Normalize(-1.3, 1.3)
    #cmap_an = matplotlib.cm.get_cmap('autumn')
    cmap_fd = matplotlib.cm.get_cmap('winter')

    y_cuts = [1, 0.5, 1e-8, -1e-8, -0.5, -1]
    for y in reversed(y_cuts):
        sgn = 1 if y>0 else -1

        iy = int(abs(y)*ny/2)
        y_true = sgn*2.0/ny * iy + sgn*1e-12
        y_color = sgn*2.0/ny * iy + sgn*0.3
        iy = int(ny/2) - sgn*iy
        if sgn < 0:
            iy += 1

        p2_ycut = p2_fd[iy, :]

        p2_plt, = ax.plot(x, p2_ycut,  color=cmap_fd(norm(y_color)), label="p2, fd, y={:4.2f}".format(y_true))
        #fd_plots.append(p2_plt)
        p2_an = ac.vec_eval_p2(x, y_true)
        #p2_plt, = ax.plot(x, p2_an, color="orange", label="p2, an, y={:4.2f}".format(y_true))
        #an_plots.append(p2_plt)
        add_text(ax, p2_ycut, "p2, y={:4.2f}".format(y_true), 0.1)

        ax_err.plot(x, p2_ycut - p2_an, color=cmap_fd(norm(y_color)), label="p2_diff, y={:4.2f}".format(y_true))

    #fd_plots = []
    #an_plots = []
    p1_plt, = ax.plot(x, p1_fd, color='red', label="p1, fd")

    #fd_plots.append(p1_plt)
    p1_an = ac.vec_eval_p1(x)
    #p1_plt, = ax.plot(x, p1_an, color='orange', label="p1, an")
    #an_plots.append(p1_plt)
    add_text(ax, p1_fd, "p1", -0.3)

    ax_err.plot(x, p1_fd - p1_an, color='red', label="p1_diff")

    ax.set_xlabel("x")
    ax.set_ylabel("pressure")
    p_list = [ac.P1, ac.P2[0], ac.P2[1]]
    ax.set_ylim( (min(p_list)-0.5, 1.1*max(p_list)+0.5) )

    #ax_err.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
    ax_err.legend(loc=9)
    ax_err.set_xlabel("x")
    ax_err.set_ylabel("pressure difference")
    ax_err.ticklabel_format(style='sci', scilimits=(0,0))

    #y = (ac.P1 - ac.P2) * np.cosh( x/ac.k1) / np.cosh(1/ac.k1) + ac.P2
    #ax.plot(x, y, label="ref")

    #all_plots = fd_plots + an_plots
    #labels = [ plot.get_label() for plot in all_plots]
    #ax.legend(all_plots, labels, bbox_to_anchor=(-.05, 1), loc=0, borderaxespad=0.)

    #stripes=[0.0001, 0.00033, 0.001, 0.0033, 0.01, 0.033, 0.1, 0.33, 1]
    #ac.plot_analytical()


    # ax_left.plot(x, 2*self.sigma*(f_p2(x,0) - f_p1(x)), label="f")
    # for y in stripes:
    #     p2 = f_p2(x, y)
    #     ax_left.plot(x, p2, label="p2, y={:8.6f}".format(y))
    # ax_left.legend(bbox_to_anchor=(-0.05, 1), loc=0, borderaxespad=0.)
    #
    # ax_left.set_xlabel("X direction")
    # # Y direction plot
    # for xs in stripes:
    #     xs = 1.0 - xs
    #     p2 = f_p2(xs, x)
    #     ax_right.plot(x, p2, label="p2, x={:8.6f}".format(xs))
    #
    # ax_right.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # ax_right.set_xlabel("Y direction")


    plt.savefig("barrier_solution.pdf", bbox_inches='tight')
    plt.show()
    return
    # compute approx of -k2*\Lapl p_2 for analytical sol.

    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)

    grid = np.linspace(0,1,20)
    X, Y = np.meshgrid(grid, grid)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i,j]
            y = Y[i, j]
            d = 1e-7
            p2_11 = ac.vec_eval_p2(x,y)
            p2_21 = ac.vec_eval_p2(x + d, y)
            p2_01 = ac.vec_eval_p2(x - d, y)
            p2_12 = ac.vec_eval_p2(x, y + d)
            p2_10 = ac.vec_eval_p2(x, y - d)

            Z[i,j] = (4* p2_11 - p2_21 - p2_01 - p2_10 - p2_12) / d /d
    #ax.plot_wireframe(X, Y, p2, rstride=10, cstride=10)
    print(Z)
    surf = ax.contourf(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)



    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()



def plot_p2_field(x, y, p2_diff):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(111)
    #ax2 = fig.add_subplot(122)

    #CS = ax1.contourf(x, y, np.abs(p2_diff), 10, locator=matplotlib.ticker.LogLocator())
    grid = np.abs(p2_diff)
    imgplot = ax1.imshow(grid,
                         norm=matplotlib.colors.LogNorm(vmin=np.min(grid), vmax=np.max(grid)),
                         extent= (x[0],x[-1], y[-1], y[0]))
    #ax1.set_xticks(x)
    #ax1.set_yticks(y)

    cbar = plt.colorbar(imgplot)
    cbar.ax.set_ylabel('diff')
    plt.show()

    # Add the contour line levels to the colorbar
    #cbar.add_lines(CS2)

    # nx = len(x)
    # ax2.plot(y, p2_diff[:, int(nx/2)], label='center')
    # ax2.plot(y, p2_diff[:, 1], label='0.01')
    # ax2.plot(y, p2_diff[:, -2], label='0.99')
    # ax2.legend()
    #return (ax1, ax2)




def compute_error(n_terms, fd_n, k2, sigma, y_band=True):
    ac = BarrierFracture(k1=1, k2=k2, sigma=sigma, P1=0, P2=(10, -10), n_terms=n_terms)
    nx,  ny = fd_n, fd_n
    x, y, p1, p2 = ac.solve_fd(nx, ny)

    p1_diff = ac.vec_eval_p1(x) -p1
    p1_l2 = la.norm( p1_diff) * np.sqrt(2.0 / nx)

    if y_band:
        band = np.abs(y) < 0.5
    else:
        band = np.arange(0, len(y), 1)
    an_p2_band = ac.vec_eval_p2(x[None, :], y[band, None])
    p2_diff = an_p2_band - p2[band, :]
    #plot_p2_field(x, y[band], p2_diff)

    p2_l2 = la.norm(p2_diff.ravel()) * np.sqrt(2.0/ nx * 2.0/ ny)

    print("Errors: ", "e1:", p1_l2, "e2:", p2_l2)
    return p1_l2, p2_l2

    fig = plt.figure(figsize=(15, 5))
    #ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(111)

    # p2 BC condtion for Analytical solution on X axis
    # p2_dy = np.empty_like(x)
    # p12_diff = np.empty_like(x)
    # for i, xv in enumerate(x):
    #     d = 1e-6
    #     p20 = ac.vec_eval_p2(xv, 0)
    #     p2d = ac.vec_eval_p2(xv, d)
    #     p2_dy[i] = ac.k2*(p2d - p20) / d
    #     p12_diff[i] = ac.sigma*(p20 - ac.vec_eval_p1(xv))
    # ax1.plot(x, p2_dy, label = "dp2/dy(x,0)")
    # ax1.plot(x, p12_diff, label = "diff")
    # ax1.legend()

    # p2 error countour plot
    #CS = ax1.contourf(x, y, p2_diff)
    #cbar = plt.colorbar(CS)
    #cbar.ax.set_ylabel('diff')

    # Error lines for constant x
    for ix in [int(0.5*nx), int(0.8*nx), int(nx-1)]:
        ax2.plot(y[band], p2_diff[:, ix], label='ix')
    ax2.legend()


    # Analytical and FD solution lines for constant X
    # ax2.plot(y, an_p2_band[:, int(nx/2)], label='an, x=0')
    # ax2.plot(y, an_p2_band[:, 1], label='an, x=0.01')
    # ax2.plot(y, an_p2_band[:, -2], label='an, x=0.99')
    #
    # ax2.plot(y, p2[:, int(nx/2)], label='fd, x=0')
    # ax2.plot(y, p2[:, 1], label='fd, x=0.01')
    # ax2.plot(y, p2[:, -2], label='fd, x=0.99')




    ax2.legend()
    plt.show()


    return (p1_l2, p2_l2, p1_diff, p2_diff)


def plot_decay(table, n_an, n_fd):
    fig = plt.figure(figsize=(12, 8))

    ax_an2 = fig.add_subplot(223)
    ax_an1 = fig.add_subplot(221, sharex=ax_an2)
    ax_fd2 = fig.add_subplot(224, sharey=ax_an2)
    ax_fd1 = fig.add_subplot(222, sharex=ax_fd2, sharey=ax_an1)


    p1_cmap = matplotlib.cm.get_cmap('Blues')
    p2_cmap = matplotlib.cm.get_cmap('Reds')
    fd_norm = matplotlib.colors.LogNorm(vmin=min(n_fd)/10, vmax=max(n_fd))
    an_norm = matplotlib.colors.LogNorm(vmin=min(n_an)/10, vmax=max(n_an))

    for i, nfd in enumerate(n_fd):
        ax_an1.loglog(n_an, table[:, i, 0], color=p1_cmap(fd_norm(nfd)) , label="p1, $N_f$: {}".format(nfd))
        ax_an2.loglog(n_an, table[:, i, 1], color=p2_cmap(fd_norm(nfd)) , label="p2, $N_f$: {}".format(nfd))
    #ax_an1.loglog(n_an, np.array(n_an) ** (-2.0), color = 'orange', label="$N^{-2}$")
    #ax_an2.loglog(n_an, np.array(n_an) ** (-2.0), color='orange', label="$N^{-2}$")
    #ax_an1.set_xticklabels(n_an)
    #ax_an2.set_xticks(n_an)
    #ax_an1.set_xlabel("Analytical solution. \# of terms to sum $[N_a]$")
    plt.setp(ax_an1.get_xticklabels(), visible=False)
    ax_an2.set_xlabel("Analytical solution. \# of terms to sum $[N_a]$")
    ax_an1.set_ylabel("$p_1$, approx. of $L^2$ error.")
    ax_an2.set_ylabel("$p_2$, approx. of $L^2$ error.")
    ax_an1.legend(loc=1)
    ax_an2.legend(loc=1)

    for i, nan in enumerate(n_an):
        ax_fd1.loglog(n_fd, table[i, :, 0], color=p1_cmap(an_norm(nan)), label="p1, $N_a$: {}".format(nan))
        ax_fd2.loglog(n_fd, table[i, :, 1], color=p2_cmap(an_norm(nan)), label="p2, $N_a$: {}".format(nan))
    ax_fd1.loglog(n_fd, 1e2*np.array(n_fd) ** (-2.0), color = 'orange', label="ref. $N^{-2}$")
    ax_fd2.loglog(n_fd, 1e2*np.array(n_fd) ** (-2.0), color='orange', label="ref. $N^{-2}$")
    ax_fd1.set_ylim( 0.8*np.min(table[:, :, 0].ravel()), 1.2*np.max(table[:, :, 0].ravel()) )
    ax_fd2.set_ylim( 0.8*np.min(table[:, :, 1].ravel()), 1.2*np.max(table[:, :, 1].ravel()) )
    #ax_fd1.set_xticks(n_fd)
    #ax_fd2.set_xticks(n_fd)
    #ax_fd1.set_xlabel("Finite diferences. \# of points on one side $[N_f]$.")
    ax_fd2.set_xlabel("Finite diferences. \# of points on one side $[N_f]$.")
    #ax_fd1.set_ylabel("Approx. of $|| p_f - p_a ||_{L_2}$.")
    #ax_fd2.set_ylabel("Approx. of $|| p_f - p_a ||_{L_2}$.")
    plt.setp(ax_fd1.get_xticklabels(), visible=False)
    ax_fd1.legend(loc=3)
    ax_fd2.legend(loc=3)
    plt.savefig("barrier_convergency.pdf")
    plt.show()



def error_decay(plot = False):
    """
    Compute and plot L2 errors of p1 and p2 as functions of
    number of summed terms in the series and number of FD points.
    Using fixed problem parameters.
    :return:
    """
    k1=0.1
    sigma = 1

    # n_terms_list =  [10, 100, 1000, 10000]
    # nx_list = [10, 20, 40, 80, 160, 320]
    #n_terms_list =  [8, 16, 32, 64, 128]
    n_terms_list = [128]
    nx_list = [20, 40, 80, 160, 320]#, 640, 1280]


    err_table = np.empty((len(n_terms_list), len(nx_list), 2))
    for i, n_terms in enumerate(n_terms_list):
        for j, nx in enumerate(nx_list):
            err_table[i,j,:] = compute_error(n_terms, nx, (1000,3000), (100, 4000))

    plot_decay(err_table, n_terms_list, nx_list)



def plot_analytical_eq():
    """
    Plot error when analytical solution is plugged into selected equations.
    :return:
    """
    ac = BarrierFracture(k1=1, k2=(1, 1), sigma=(100, 100), P1=0, P2=(10, 10), n_terms=1000)

    #######################################
    # Error in P1 eq, and BC+ BC-
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X = np.linspace(0,1,50)
    p1_err = np.empty_like(X)
    bc_err = np.zeros( (2, len(X)) )
    d = 1e-4
    for i, x in enumerate(X):
        p1_1 = ac.vec_eval_p1(x)
        p1_0 = ac.vec_eval_p1(x - d)
        p1_2 = ac.vec_eval_p1(x + d)
        p2_0 = ac.vec_eval_p2(x, 1e-16)
        p2_1 = ac.vec_eval_p2(x, -1e-16)
        p1_err[i] = -ac.k1 * (2* p1_1 - p1_0 - p1_2) / d/ d - ac.sigma[0] * (p1_1 - p2_0) - ac.sigma[1] * (p1_1 - p2_1)

        p2_00 = ac.vec_eval_p2(x, d)
        p2_11 = ac.vec_eval_p2(x, -d)
        bc_err[0, i] = ac.k2[0] * ( p2_00 - p2_0) / d  + ac.sigma[0] * (p1_1 - p2_0)
        bc_err[1, i] = ac.k2[1] * ( p2_11 - p2_1) / d  + ac.sigma[1] * (p1_1 - p2_1)


    ax.plot(X, p1_err, label="p1_err")
    ax.plot(X, bc_err[0], label="bc_err_top")
    ax.plot(X, bc_err[1], label="bc_err_top")
    ax.legend()
    plt.show()

    return
    #########################################
    # compute approx of -k2*\Lapl p_2 for analytical sol.
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)

    d = 1e-7
    grid = np.linspace(0,1,20)
    X, Y = np.meshgrid(grid, grid[1:])
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i,j]
            y = Y[i, j]
            p2_11 = ac.vec_eval_p2(x,y)
            p2_21 = ac.vec_eval_p2(x + d, y)
            p2_01 = ac.vec_eval_p2(x - d, y)
            p2_12 = ac.vec_eval_p2(x, y + d)
            p2_10 = ac.vec_eval_p2(x, y - d)

            Z[i,j] = (4* p2_11 - p2_21 - p2_01 - p2_10 - p2_12) / d /d
    #ax.plot_wireframe(X, Y, p2, rstride=10, cstride=10)

    surf = ax.contourf(X, Y, np.abs(Z), locator=matplotlib.ticker.LogLocator(), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)



    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()



def error_decay_parameter_study():
    step = 0.25
    sigma_bounds = np.arange(-3, 3+step, step)
    k2_bounds = np.arange(-3, 3+step, step)
    sigma_list = (sigma_bounds[0:-1] + sigma_bounds[1:])/2
    k2_list = (k2_bounds[0:-1] + k2_bounds[1:]) / 2
    fit_table = np.empty( (len(sigma_list), len(k2_list), 2, 2))
    for isg, sigma_log in enumerate(sigma_list):
        for ik2, k2_log in enumerate(k2_list):
            print("Case: sigma ", sigma_log, " k2 ", k2_log)
            sigma = 10.0**sigma_log
            k2 = 10.0**k2_log
            
            nx_list = [20, 40, 80, 160, 320]
            #nx_list = [20, 40, 80]
            err_table = np.empty( (len(nx_list), 2) )
            for inx, nx in enumerate(nx_list):
                #sigma_values = (3*sigma**0.5, sigma )
                sigma_values = (2*sigma**0.5, sigma )     # Assymetry to avoid trivial case, p1=0
                k2_values = (k2, k2**0.5)
                err_table[inx, :] = compute_error(100, nx, k2_values, sigma_values, y_band=0.05)
            fit_table[isg, ik2,:, 0] = np.polyfit(-np.log10(np.array(nx_list)), np.log10(err_table[:, 0]), deg = 1)
            fit_table[isg, ik2,:, 1] = np.polyfit(-np.log10(np.array(nx_list)), np.log10(err_table[:, 1]), deg = 1)

    fig = plt.figure(figsize=(16, 3))

    ax0 = fig.add_subplot(141)
    ax0.set_ylabel("$k_2$")
    ax0.set_yscale('log')
  
    ax_list =  [
        ax0,
        fig.add_subplot(142, sharey = ax0),
        fig.add_subplot(143, sharey = ax0),
        fig.add_subplot(144, sharey = ax0)
    ]
    im_list =[]
    X,Y = np.meshgrid(10.0**sigma_bounds, 10.0**k2_bounds)

    cm = [ matplotlib.cm.get_cmap('seismic'), matplotlib.cm.get_cmap('PRGn') ]
    divnorm = [mcolors.DivergingNorm(vcenter=2), mcolors.DivergingNorm(vcenter=0)]
    for i_deg in [0, 1]:
        for i_dim in [0, 1]:
            ii = i_dim +  2*i_deg
            ax = ax_list[ii]
            #ax.axis('equal')
            im = ax.pcolormesh(X, Y, fit_table[:,:, i_deg, i_dim], cmap=cm[i_deg], norm=divnorm[i_deg])
            ax.set_xscale('log')
            im_list.append(im)
            
            #ax.set_xticks(np.arange(0, len(sigma_list), 1))
            #ax.set_yticks(np.arange(0, len(k2_list), 1))
            #ax.set_xticks(10.0 ** sigma_list)
            #ax.set_yticks(10.0 ** k2_list)
            #ax.ticklabel_format(style='sci', scilimits=(0, 0))

            ax.yaxis.set_minor_locator(plt.NullLocator())
            ax.xaxis.set_major_locator(plt.LogLocator(numticks=4))
            ax.yaxis.set_major_locator(plt.LogLocator(numticks=7))
            #ax.tick_params(
                #axis='both',  # changes apply to the x-axis
                #which='minor',  # both major and minor ticks are affected
                #left='off',
                #bottom='off')  # labels along the bottom edge are off

            #ax.tick_params(
                #axis='y',  # changes apply to the x-axis
                #which='minor',  # both major and minor ticks are affected
                #left='off',
                #labelleft='off',
                #bottom='off')  # labels along the bottom edge are off

            # avoid ticks labels for vertical axis
            if ii > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
                #ax.yaxis.set_major_formatter(plt.NullFormatter())
                #ax.tick_params(
                    #axis='y',  # changes apply to the y-axis
                    #which='both',  # both major and minor ticks are affected
                    #labelleft='off')  # labels along the left edge are off

    ax_list[0].set_xlabel("$\sigma$, p1")
    ax_list[1].set_xlabel("$\sigma$, p2")
    ax_list[2].set_xlabel("$\sigma$, p1")
    ax_list[3].set_xlabel("$\sigma$, p2")
    
    cbar_order = fig.colorbar(im_list[0], ax=ax_list[:2])
    cbar_order.set_label("order of convergece", rotation=90)
    cbar_magnitude = fig.colorbar(im_list[2], ax=ax_list[2:])
    cbar_magnitude.set_label("error magnitude, log10", rotation=90)

    #plt.tight_layout()
    #plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig("barrier_conv_rate.pdf", bbox_inches='tight')
    plt.show()






#error_decay()

#plot_solution()

#plot_analytical_eq()


error_decay_parameter_study()
