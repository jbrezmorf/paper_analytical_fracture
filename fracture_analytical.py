import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_la
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d


# Analytica solution to continuous case

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

class ContinousFracture:

    def __init__(self, k1, k2, sigma, P1, P2, n_terms):
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.P1 = P1
        self.P2 = P2
        self.n_terms = n_terms
        ###
        self.precompute_analytical()

    def precompute_analytical(self):
        self.k = np.sqrt(self.k1 / 2.0 / self.sigma)

        nn = self.n_series = np.arange(1, self.n_terms, 1.0)
        k_pi = self.k * np.pi
        k_n_pi_sqr = (k_pi * self.n_series) ** 2
        m2_pi = -2.0 * np.pi
        exp_pi_m2_n = np.exp(m2_pi * nn)
        self.sinh_pi_n = 0.5 * (1.0 - exp_pi_m2_n)
        self.cosh_pi_n = 0.5 * (1.0 + exp_pi_m2_n)
        an_denom = self.k2 * np.pi * nn * self.cosh_pi_n \
                   * (1.0 + k_n_pi_sqr) \
                   + self.sigma * k_n_pi_sqr * self.sinh_pi_n

        alternate = np.empty((self.n_terms-1,), float)
        alternate[::2] = -1      # 0 is n=1, thus odd
        alternate[1::2] = +1

        self.an = alternate * self.k2 / an_denom
        self.un = self.an * self.sinh_pi_n / (1.0 + k_n_pi_sqr)

        u_sum = np.sum(alternate * self.un)
        self.B0 = (self.P2 - self.P1) / ( \
                    1.0 + 2 * u_sum \
                    + self.k2 * np.cosh(1.0 / self.k) \
                    / self.sigma / self.k / np.sinh(1.0 / self.k) \
            )
        self.u0 = -self.k2 * self.B0 / (self.sigma * self.k * np.sinh(1.0 / self.k))
        alt_u0 = (self.P1 -self.P2 + self.B0*(1+2*u_sum)) / np.cosh(1.0 / self.k)
        print("u_sum: ", u_sum)
        print("u0:", self.u0)
        print("alt_u0:", alt_u0)
        assert np.isclose(self.u0, alt_u0)

        self.vec_eval_p2 = np.vectorize(self.eval_p2)
        self.vec_eval_p1 = np.vectorize(self.eval_p1)


    def eval_p2(self, x, y):
        y = np.abs(y)
        pi_x = np.pi * x
        pi_1y_m2 = -2 * np.pi * (1 - y)

        series_sum = np.sum(self.an * np.cos(pi_x * self.n_series)
                            * 0.5 * (1.0 - np.exp(pi_1y_m2 * self.n_series)))
        p2_sum = self.P2 + self.B0 * (y - 1) - 2 * self.B0 * series_sum
        return p2_sum

    def eval_p1(self, x):
        #x = np.abs(x)
        pi_x = np.pi * x
        series_sum = np.sum(self.un * np.cos(pi_x * self.n_series))
        p1_sum = self.P2 - self.B0 + self.u0 * np.cosh(x  / self.k) - 2 * self.B0 * series_sum
        return p1_sum

    def plot_p1(self, ax, x):
        f_p1 = np.vectorize(self.eval_p1)
        p1 = f_p1(x)
        ax.plot(x, p1, label="p1")
        return p1

    def plot_p2(self, ax, x, y):
        f_p2 = np.vectorize(self.eval_p2)
        p2 = f_p2(x, y)
        ax.plot(x, p2, label="p2, y={:8.6f}".format(y))
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
        #k1 = self.k1
        #sigma = self.sigma

        nx = int(nx/2)
        ny += int(ny % 2)    # Make ir even


        Nx = nx+1
        Ny = ny+1
        dx = 1.0/nx
        dy = 2.0/ny
        dx2 = 2*dx
        dy2 = 2*dy
        ddx = dx*dx
        ddy = dy*dy

        n_dofs = Nx*(Ny+1)
        A_vals = TripletMatrix(n_dofs, n_dofs)
        b = np.zeros((n_dofs,))
        k_frac = Ny*Nx

        k=0
        i=0
        # Dirichlet top, bottom
        for i  in [0, ny]:
            k = i*Nx
            for j in range(0, Nx):
                A_vals.add((k, k, 1.0))
                b[k] = self.P2
                k += 1

        # regular lines
        for i in range(1, ny):
            k = i * Nx
            if i == int(ny/2):
                k_frac_shift = Nx * int(ny / 2) + Nx

                # Fracture BC, Neumman:
                # O(dy^2) formula: (-3*u_0 + 4*u_1 - u_2) / (2*dy)
                # A_vals.add((k, k,       sigma * (-3.0) / dx2))
                # A_vals.add((k, k + 1,   sigma * (4.0) / dx2))
                # A_vals.add((k, k + 2,   sigma * (-1.0) / dx2))

                #k+=1
                for j in range(0, Nx):


                    # Neumann 2. order
                    A_vals.add((k, k,        2* k2 * (-3.0) / dy2 - 2*sigma ))
                    A_vals.add((k, k + k_frac_shift, 2*sigma))
                    A_vals.add((k, k - 1*Nx,    k2 * 4 / dy2))
                    A_vals.add((k, k - 2*Nx,    k2 * (-1) / dy2))
                    A_vals.add((k, k + 1*Nx,    k2 * 4 / dy2))
                    A_vals.add((k, k + 2*Nx,    k2 * (-1) / dy2))
                    k += 1



                # Fracture equation
                k = k_frac
                # Left zero Nemmann
                A_vals.add((k, k, -3.0 / dx2))
                A_vals.add((k, k + 1, 4 / dx2))
                A_vals.add((k, k + 2, -1 / dx2))
                k += 1

                # regular points
                for j in range(1,nx):
                    A_vals.add((k, k, 2 * k1 / ddx + 2*sigma))
                    A_vals.add((k, k - 1, -k1 / ddx))
                    A_vals.add((k, k + 1, -k1 / ddx))
                    A_vals.add((k, k - k_frac_shift, - 2*sigma))
                    k += 1

                # Right Dirichlet
                A_vals.add((k, k, 1.0))
                b[k] = self.P1
                continue

            # left zero Neumann
            # O(dx^2) formula: (-3*u_0 + 4*u_1 - u_2) / (2*dx)
            A_vals.add((k, k,     -3.0/dx2))
            A_vals.add((k, k + 1,   4/dx2))
            A_vals.add((k, k + 2, -1/dx2))
            k += 1

            # regular points
            for j in range(1,nx):
                A_vals.add((k, k,      2*k2/ddx + 2*k2/ddy))
                A_vals.add((k, k-1,    -k2/ddx))
                A_vals.add((k, k+1,    -k2/ddx))
                A_vals.add((k, k-Nx,   -k2/ddy))
                A_vals.add((k, k+Nx,   -k2/ddy))
                k += 1

            # right zero Neumann
            # O(dx^2) formula: (-3*u_0 + 4*u_1 - u_2) / (2*dx)
            A_vals.add((k, k,     -3.0/dx2))
            A_vals.add((k, k - 1,   4/dx2))
            A_vals.add((k, k - 2, -1/dx2))
            k += 1
        return (A_vals.make_csr(), b)


    def solve_fd(self, nx, ny):
        self.fd_shape = (nx, ny)
        A, b = self.form_fd_matrix(nx, ny)
        print("A shp: ", A.shape)
        self.dof_values = sp_la.spsolve(A, b)

        # dof_values - have dofs only for right side
        x = np.linspace(-1, 1, (nx+1))
        y = np.linspace(-1, 1, (ny+1))
        nx_2d = int(nx/2) + 1
        ny_2d = (ny + 1)
        frac_start = nx_2d * ny_2d
        right_side = self.dof_values[: frac_start ].reshape((ny_2d, nx_2d))
        left_side = right_side[:, -1: 0: -1]
        p2_mat = np.concatenate((left_side, right_side), axis = 1)

        r_vec = self.dof_values[frac_start:]
        l_vec = r_vec[-1: 0: -1]
        p1_vec = np.concatenate( (l_vec, r_vec) )

        return (x,y, p1_vec, p2_mat)


class BurdaFrac(ContinousFracture):
    def precompute_analytical(self):
        self.k = np.sqrt(self.k1 / 2.0 /self.sigma)

        nn = self.n_series = np.arange(1, self.n_terms, 1.0)
        k_pi = self.k * np.pi
        k_n_pi_sqr = (k_pi * self.n_series) ** 2
        m2_pi = -2.0 * np.pi
        exp_pi_m2_n = np.exp(m2_pi * nn)
        self.sinh_pi_n = 0.5 * (1.0 - exp_pi_m2_n)
        self.cosh_pi_n = 0.5 * (1.0 + exp_pi_m2_n)
        an_denom = self.k2 * np.pi * nn * self.cosh_pi_n \
                   * (1.0 + k_n_pi_sqr) \
                   + self.sigma * k_n_pi_sqr * self.sinh_pi_n

        #alternate = np.empty((self.n_terms-1,), float)
        #alternate[::2] = -1      # 0 is n=1, thus odd
        #alternate[1::2] = +1

        self.an =  self.k2 / an_denom
        self.un = self.an * self.sinh_pi_n / (1.0 + k_n_pi_sqr)

        u_sum = np.sum( self.un)
        self.B0 = (self.P2 - self.P1) / ( \
                    1.0 + 2 * u_sum \
                    + self.k2 * np.cosh(1.0 / self.k) \
                    / self.sigma / self.k / np.sinh(1.0 / self.k) \
            )
        self.u0 = self.k2 * self.B0 / (self.sigma * self.k * np.sinh(1.0 / self.k))
        alt_u0 = (self.P1 -self.P2 + self.B0*(1+2*u_sum)) / np.cosh(1.0 / self.k)
        print("u_sum: ", u_sum)
        print("u0:", self.u0)
        print("alt_u0:", alt_u0)

        #assert np.isclose(self.u0, alt_u0)

    def eval_p2(self, x, y):
        x = 1-np.abs(x)

        y = np.abs(y)
        pi_x = np.pi * x
        pi_1y_m2 = -2 * np.pi * (1 - y)

        series_sum = np.sum(self.an * np.cos(pi_x * self.n_series)
                            * 0.5 * (1.0 - np.exp(pi_1y_m2 * self.n_series)))
        p2_sum = self.P2 + self.B0 * (y - 1) - 2 * self.B0 * series_sum
        return p2_sum

    def eval_p1(self, x):
        x = 1 - np.abs(x)
        #x = np.abs(x)
        pi_x = np.pi * x
        series_sum = np.sum(self.un * np.cos(pi_x * self.n_series))
        p1_sum = self.P2 - self.B0 - self.u0 * np.cosh((1-x)  / self.k) - 2 * self.B0 * series_sum
        return p1_sum



def plot_test():
    ac = ContinousFracture(k1=0.01, k2=1, sigma=1, P1=5, P2=10, n_terms=10000)

    #ac = BurdaFrac(k1=10, k2=10, sigma=200, P1=5, P2=10, n_terms=10000)
    nx,  ny = 200, 200
    x, y, p1, p2 = ac.solve_fd(nx, ny)

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(121)
    ax_err = fig.add_subplot(122)

    ax.plot(x, p1, '.', label="p1fd")
    p2_y0 = p2[int(ny/2), :]
    ax.plot(x, p2_y0, '.', label="p2fd, y=0")
    ac_p1 = ac.plot_p1(ax, x)
    ac_p2 = ac.plot_p2(ax, x, 0)
    y = (ac.P1 - ac.P2) * np.cosh( x/ac.k1) / np.cosh(1/ac.k1) + ac.P2
    #ax.plot(x, y, label="ref")

    ax_err.plot(x, p1 - ac_p1, label="p1-a_p1")
    ax_err.plot(x, p2_y0- ac_p2, label="p2-a_p2")
    ax_err.legend()

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
    ax.legend(bbox_to_anchor=(-.05, 1), loc=0, borderaxespad=0.)
    plt.show()



    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # X, Y = np.meshgrid(x, y)
    # #ax.plot_wireframe(X, Y, p2, rstride=10, cstride=10)
    # surf = ax.plot_surface(X, Y, p2, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    #
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    #
    # plt.show()

def plot_p2_field(x, y, p2_diff):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    CS = ax1.contourf(x, y, p2_diff)
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('diff')
    # Add the contour line levels to the colorbar
    #cbar.add_lines(CS2)

    nx = len(x)
    ax2.plot(y, p2_diff[:, int(nx/2)], label='center')
    ax2.plot(y, p2_diff[:, 1], label='0.01')
    ax2.plot(y, p2_diff[:, -2], label='0.99')
    ax2.legend()
    plt.show()


def compute_error(n_terms, fd_n, k1, sigma, y_band=True):
    ac = ContinousFracture(k1=k1, k2=1, sigma=sigma, P1=5, P2=10, n_terms=n_terms)
    nx,  ny = fd_n, fd_n
    x, y, p1, p2 = ac.solve_fd(nx, ny)

    p1_diff = ac.vec_eval_p1(x) -p1
    p1_l2 = la.norm( p1_diff**2 ) * np.sqrt(2.0 / nx)

    if y_band:
        band = np.arange(int(ny/2*0.9), int(ny/2*1.1), 1)
        print(fd_n, len(band), len(band)*len(x)*2.0/ nx * 2.0/ ny)
    else:
        band = np.arange(0, len(y), 1)
    an_p2_band = ac.vec_eval_p2(x[None, :], y[band, None])
    p2_diff = an_p2_band - p2[band, :]

    p2_l2 = la.norm(p2_diff.ravel() ** 2) * np.sqrt(2.0/ nx * 2.0/ ny)
    plot_p2_field(x,y, p2_diff)
    return (p1_l2, p2_l2, p1_diff, p2_diff)


def plot_decay(table, n_an, n_fd):
    fig = plt.figure(figsize=(15, 5))
    ax_an1 = fig.add_subplot(221)
    ax_fd1 = fig.add_subplot(222)
    ax_an2 = fig.add_subplot(223)
    ax_fd2 = fig.add_subplot(224)

    for i, nfd in enumerate(n_fd):
        ax_an1.loglog(n_an, table[:, i, 0], 'r-', label="p1d, nfd: {}".format(nfd))
        ax_an2.loglog(n_an, table[:, i, 1], 'b-', label="p2d, nfd: {}".format(nfd))

    for i, nan in enumerate(n_an):
        ax_fd1.loglog(n_fd, table[i, :, 0], 'r-', label="p1d, nan: {}".format(nan))
        ax_fd2.loglog(n_fd, table[i, :, 1], 'b-', label="p2d, nan: {}".format(nan))


    ax_an1.legend()
    ax_fd1.legend()
    ax_an2.legend()
    ax_fd2.legend()
    plt.show()


def error_decay():
    k1=10
    sigma = 10

    # n_terms_list =  [10, 100, 1000, 10000]
    # nx_list = [10, 20, 40, 80, 160, 320]
    n_terms_list =  [10, 100, 1000]
    nx_list = [20, 40, 80, 160, 320, 640, 1280]


    err_table = np.empty((len(n_terms_list), len(nx_list), 2))
    for i, n_terms in enumerate(n_terms_list):
        for j, nx in enumerate(nx_list):
            err_table[i,j,:] = compute_error(n_terms, nx, k1, sigma)

    plot_decay(err_table, n_terms_list, nx_list)



p1_l2, p2_l2, p1_diff, p2_diff = compute_error(1000, 320, 10, 10, y_band=False)

#error_decay()
