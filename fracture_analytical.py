import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_la
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})


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

    # def sinh(self, x):
    #     return 1 - np.exp( -2.0 * x)
    #     #return np.sinh(x)
    #
    # def cosh(self, x):
    #     return 1 + np.exp( -2.0 * x)
    #     #return np.cosh(x)

    def precompute_analytical(self):
        self.k = np.sqrt(self.k1 / 2.0 / self.sigma)

        #n_err_terms = int(0.2*self.n_terms)
        n_err_terms = self.n_terms
        n_terms = self.n_terms + n_err_terms
        nn = self.n_series = np.arange(1, n_terms, 1.0)
        k_pi = self.k * np.pi
        k_n_pi_sqr = (k_pi * self.n_series) ** 2
        m2_pi = -2.0 * np.pi
        exp_pi_m2_n = np.exp(m2_pi * nn)
        self.sinh_pi_n = (1.0 - exp_pi_m2_n)
        self.cosh_pi_n = (1.0 + exp_pi_m2_n)
        an_denom = self.k2 * np.pi * nn * self.cosh_pi_n \
                    * (1.0 + k_n_pi_sqr) \
                    + self.sigma * k_n_pi_sqr * self.sinh_pi_n
        #an_denom = self.k2 * np.pi * nn * self.cosh(nn *np.pi) \
        #           * (1.0 + k_n_pi_sqr) \
        #           + self.sigma * k_n_pi_sqr * self.sinh(nn*np.pi)

        alternate = np.empty((n_terms-1,), float)
        alternate[::2] = -1      # 0 is n=1, thus odd
        alternate[1::2] = +1

        self.an = alternate * self.k2 / an_denom
        self.un = self.an * self.sinh_pi_n / (1.0 + k_n_pi_sqr)
        #self.un = self.an * self.sinh(np.pi*nn) / (1.0 + k_n_pi_sqr)

        u_sum = np.sum( (alternate * self.un)[:self.n_terms] )
        u_sum_err = np.sum( (alternate * self.un)[:self.n_terms-1:-1] )

        self.B0 = (self.P2 - self.P1) / ( \
                    1.0 + 2 * u_sum \
                    + self.k2 * np.cosh(1.0 / self.k) \
                    / self.sigma / self.k / np.sinh(1.0 / self.k) \
            )
        self.u0 = -self.k2 * self.B0 / (self.sigma * self.k * np.sinh(1.0 / self.k))
        alt_u0 = (self.P1 -self.P2 + self.B0*(1+2*u_sum)) / np.cosh(1.0 / self.k)

        an_err = np.sum(self.an[self.n_terms:] )
        an_abs_err = np.sum(np.abs(self.an[self.n_terms:]))
        un_abs_err = np.sum( np.abs(self.un[:self.n_terms-1:-1]) )
        u_sum_err = np.sum( self.un[:self.n_terms - 1:-1])

        nn = nn[self.n_terms:]
        err = []
        for i in range(1, 10):
            e = np.log(np.abs(np.sum( np.cos(nn*np.pi/i)*self.an[self.n_terms:])))
            err.append(e)
        print(err)
        #an_x2_err = np.sum(np.cos(nn * np.pi) * self.un[self.n_terms:])


        #print("un_err: ", np.log(np.abs(u_sum_err)), "un_abs_err:", np.log(np.abs(un_abs_err)),
        #      "an_err: ", np.log(np.abs(an_err)), "an_abs_err:", np.log(np.abs(an_abs_err)))
        # print("alt_u0:", alt_u0)
        # assert np.isclose(self.u0, alt_u0)

        self.vec_eval_p2 = np.vectorize(self.eval_p2)
        self.vec_eval_p1 = np.vectorize(self.eval_p1)


    def eval_p2(self, x, y):
        y = np.abs(y)
        pi_x = np.pi * x
        pi_1y_m2 = -2 * np.pi * (1 - y)

        series_sum = np.sum(self.an * np.cos(pi_x * self.n_series)
                            * (np.exp( (-y * np.pi) * self.n_series ) - np.exp( ((y-2.0)*np.pi) * self.n_series ) ) )


        #terms = self.an * np.cos(pi_x * self.n_series)*self.sinh(np.pi * (1 - y) * self.n_series)
        #series_sum = np.sum(terms)
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

        nx = int(nx/2)       # we compute only half of domain
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





def plot_test():
    ac = ContinousFracture(k1=0.1, k2=1, sigma=1, P1=5, P2=10, n_terms=3)

    #ac = BurdaFrac(k1=0.1, k2=1, sigma=3, P1=5, P2=10, n_terms=10000)
    nx,  ny = 100, 100
    x, y, p1, p2 = ac.solve_fd(nx, ny)

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(121)
    ax_err = fig.add_subplot(122)

    ax.plot(x, p1, '.', label="p1fd")
    ac_p1 = ac.plot_p1(ax, x)
    ax_err.plot(x, p1 - ac_p1, label="p1-a_p1")

    y_cuts = [0, 0.2, 0.5, 0.8, 1]
    for y in y_cuts:
        iy = int(y*ny/2)
        y_true = 2.0/ny * iy
        iy = int(ny/2) - iy
        p2_ycut = p2[iy, :]
        ax.plot(x, p2_ycut, '.', label="p2fd, y={}".format(y_true))
        ac_p2 = ac.plot_p2(ax, x, y_true)

        ax_err.plot(x, p2_ycut - ac_p2, label="p2-a_p2, y={}".format(y))

    #y = (ac.P1 - ac.P2) * np.cosh( x/ac.k1) / np.cosh(1/ac.k1) + ac.P2
    #ax.plot(x, y, label="ref")

    ax_err.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)

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
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    CS = ax1.contourf(x, y, p2_diff)
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('diff')
    # Add the contour line levels to the colorbar
    #cbar.add_lines(CS2)

    # nx = len(x)
    # ax2.plot(y, p2_diff[:, int(nx/2)], label='center')
    # ax2.plot(y, p2_diff[:, 1], label='0.01')
    # ax2.plot(y, p2_diff[:, -2], label='0.99')
    # ax2.legend()
    return (ax1, ax2)

    plt.show()


def compute_error(n_terms, fd_n, k1, sigma, y_band=True):
    ac = ContinousFracture(k1=k1, k2=1, sigma=sigma, P1=5, P2=10, n_terms=n_terms)
    nx,  ny = fd_n, fd_n
    x, y, p1, p2 = ac.solve_fd(nx, ny)

    p1_diff = ac.vec_eval_p1(x) -p1
    p1_l2 = la.norm( p1_diff) * np.sqrt(2.0 / nx)

    if y_band:
        band = np.arange(int(ny/2*0.9), int(ny/2*1.1), 1)
        print(fd_n, len(band), len(band)*len(x)*2.0/ nx * 2.0/ ny)
    else:
        band = np.arange(0, len(y), 1)
    an_p2_band = ac.vec_eval_p2(x[None, :], y[band, None])
    p2_diff = an_p2_band - p2[band, :]

    p2_l2 = la.norm(p2_diff.ravel()) * np.sqrt(2.0/ nx * 2.0/ ny)

    return p1_l2, p2_l2

    # fig = plt.figure(figsize=(15, 5))
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    #
    # # p2 BC condtion for Analytical solution on X axis
    # # p2_dy = np.empty_like(x)
    # # p12_diff = np.empty_like(x)
    # # for i, xv in enumerate(x):
    # #     d = 1e-6
    # #     p20 = ac.vec_eval_p2(xv, 0)
    # #     p2d = ac.vec_eval_p2(xv, d)
    # #     p2_dy[i] = ac.k2*(p2d - p20) / d
    # #     p12_diff[i] = ac.sigma*(p20 - ac.vec_eval_p1(xv))
    # # ax1.plot(x, p2_dy, label = "dp2/dy(x,0)")
    # # ax1.plot(x, p12_diff, label = "diff")
    # # ax1.legend()
    #
    # # p2 error countour plot
    # CS = ax1.contourf(x, y, p2_diff)
    # cbar = plt.colorbar(CS)
    # cbar.ax.set_ylabel('diff')
    #
    # # Error lines for constant x
    # ax2.plot(y, p2_diff[:, int(nx/2)], label='center')
    # ax2.plot(y, p2_diff[:, 1], label='0.01')
    # ax2.plot(y, p2_diff[:, -2], label='0.99')
    # ax2.legend()
    #
    #
    # # Analytical and FD solution lines for constant X
    # # ax2.plot(y, an_p2_band[:, int(nx/2)], label='an, x=0')
    # # ax2.plot(y, an_p2_band[:, 1], label='an, x=0.01')
    # # ax2.plot(y, an_p2_band[:, -2], label='an, x=0.99')
    # #
    # # ax2.plot(y, p2[:, int(nx/2)], label='fd, x=0')
    # # ax2.plot(y, p2[:, 1], label='fd, x=0.01')
    # # ax2.plot(y, p2[:, -2], label='fd, x=0.99')




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
    plt.savefig("continuous_convergency.pdf")
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
    n_terms_list =  [10, 50, 100, 200, 500]
    nx_list = [20, 40, 80, 160, 320, 640, 1280]


    err_table = np.empty((len(n_terms_list), len(nx_list), 2))
    for i, n_terms in enumerate(n_terms_list):
        for j, nx in enumerate(nx_list):
            err_table[i,j,:] = compute_error(n_terms, nx, k1, sigma)

    plot_decay(err_table, n_terms_list, nx_list)



#p1_l2, p2_l2, p1_diff, p2_diff = compute_error(100, 100, 0.01, 1, y_band=False)

#error_decay()

#plot_test()

ac = ContinousFracture(k1=0.01, k2=1, sigma=1, P1=5, P2=10, n_terms=10)
ac = ContinousFracture(k1=0.01, k2=1, sigma=1, P1=5, P2=10, n_terms=100)
ac = ContinousFracture(k1=0.01, k2=1, sigma=1, P1=5, P2=10, n_terms=1000)
ac = ContinousFracture(k1=0.01, k2=1, sigma=1, P1=5, P2=10, n_terms=10000)
