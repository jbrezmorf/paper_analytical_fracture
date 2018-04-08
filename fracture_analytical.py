import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt


# Analytica solution to continuous case

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

        self.an = self.k2 / an_denom
        self.un = self.an * self.sinh_pi_n / (1.0 + k_n_pi_sqr)
        u_sum = np.sum(self.un)
        self.B0 = (self.P2 - self.P1) / ( \
                    1.0 + 2 * u_sum \
                    + self.k2 * np.cosh(1.0 / self.k) \
                    / self.sigma / self.k / np.sinh(1.0 / self.k) \
            )
        self.u0 = self.k2 * self.B0 / (self.sigma * self.k * np.sinh(1.0 / self.k))
        print("u_sum: ", u_sum)
        print("u0:", self.u0)

    def eval_p2(self, x, y):
        y = np.abs(y)
        x = 1 - np.abs(x)
        pi_x = np.pi * x
        pi_1y_m2 = -2 * np.pi * (1 - y)

        series_sum = np.sum(self.an * np.cos(pi_x * self.n_series)
                            * 0.5 * (1.0 - np.exp(pi_1y_m2 * self.n_series)))
        p2_sum = self.P2 + self.B0 * (y - 1) - 2 * self.B0 * series_sum
        return p2_sum

    def eval_p1(self, x):
        x = 1 - np.abs(x)
        pi_x = -np.pi * x
        series_sum = np.sum(self.un * np.cos(pi_x * self.n_series))
        p1_sum = self.P2 - self.B0 - self.u0 * np.cosh((x - 1.0) / self.k) - 2 * self.B0 * series_sum
        return p1_sum

    def plot_analytical(self, stripes=[0.0001, 0.00033, 0.001, 0.0033, 0.01, 0.033, 0.1, 0.33, 1]):
        fig = plt.figure(figsize=(15, 5))
        # Axis = podgraf, zde jen jeden.
        ax_left = fig.add_subplot(121)
        ax_right = fig.add_subplot(122)

        x = np.linspace(-1, 1, 100)
        f_p1 = np.vectorize(ac.eval_p1)
        f_p2 = np.vectorize(ac.eval_p2)

        # X direction plot
        p1 = f_p1(x)
        ax_left.plot(x, p1, label="p1")
        for y in stripes:
            p2 = f_p2(x, y)
            ax_left.plot(x, p2, label="p2, y={:8.6f}".format(y))
        ax_left.legend(bbox_to_anchor=(-0.05, 1), loc=0, borderaxespad=0.)

        ax_left.set_xlabel("X direction")
        # Y direction plot
        for xs in stripes:
            xs = 1.0 - xs
            p2 = f_p2(xs, x)
            ax_right.plot(x, p2, label="p2, x={:8.6f}".format(xs))

        ax_right.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax_right.set_xlabel("Y direction")
        plt.show()


    def form_fd_matrix(self, nx, ny):
        """
        :param nx: Number of intervals in X direction
        :param ny: Number of intervals in Y direction
        :return:
        """
        ny += ny % 2    # Make ir even


        Nx = nx+1
        Ny = ny+1
        dx = 1.0/nx
        dy = 1.0/ny
        dx2 = 2*dx
        dy2 = 2*dy
        ddx = dx*dx
        ddy = dy*dy

        A_vals = []
        b = np.array(Nx * (Ny + 1))
        k_frac = Ny*Nx

        k=0
        i=0
        # Dirichlet top, bottom
        for i  in [0, ny]:
            k = i*Nx
            for j in range(0, Nx):
                A_vals.append((k, k, 1.0))
                b[k] = self.P2
                k += 1

        # regular lines
        for i in range(1, ny):
            k = i * Nx
            if i == ny/2:
                k_frac_shift = Nx * ny / 2

                # Fracture BC, Neumman:
                # O(dy^2) formula: (-3*u_0 + 4*u_1 - u_2) / (2*dy)
                for j in range(0, Nx):
                    A_vals.append((k, k,        self.k2 * 2 * (-3.0) / dy2 - 2*self.sigma ))
                    A_vals.append((k, k + k_frac_shift, 2*self.sigma))
                    A_vals.append((k, k - 1,    self.k2 * 4 / dy2))
                    A_vals.append((k, k - 2,    self.k2 * (-1) / dy2))
                    A_vals.append((k, k + 1,    self.k2 * 4 / dy2))
                    A_vals.append((k, k + 2,    self.k2 * (-1) / dy2))
                    k += 1

                # Fracture equation
                k = k_frac
                # Left zero Nemmann
                A_vals.append((k, k, -3.0 / dx2))
                A_vals.append((k, k + 1, 4 / dx2))
                A_vals.append((k, k + 2, -1 / dx2))
                k += 1

                # regular points
                for j in range(1,nx):
                    A_vals.append((k, k, 2 * self.k1 / ddx + 2*self.sigma))
                    A_vals.append((k, k - 1, -self.k1 / ddx))
                    A_vals.append((k, k + 1, -self.k1 / ddx))
                    A_vals.append((k, k - k_frac_shift, - 2*self.sigma))
                    k += 1

                # Right Dirichlet
                A_vals.append((k, k, 1.0))
                b[k] = self.P1
                continue

            # left zero Neumann
            # O(dx^2) formula: (-3*u_0 + 4*u_1 - u_2) / (2*dx)
            A_vals.append((k, k,     -3.0/dx2))
            A_vals.append((k, k + 1,   4/dx2))
            A_vals.append((k, k + 2, -1/dx2))
            k += 1

            # regular points
            for j in range(1,nx):
                A_vals.append((k, k,      2*self.k2/ddx))
                A_vals.append((k, k-1,    -self.k2/ddx))
                A_vals.append((k, k+1,    -self.k2/ddx))
                A_vals.append((k, k-Nx,   -self.k2/ddx))
                A_vals.append((k, k+Nx,   -self.k2/ddx))
                k += 1

            # right zero Neumann
            # O(dx^2) formula: (-3*u_0 + 4*u_1 - u_2) / (2*dx)
            A_vals.append((k, k,     -3.0/dx2))
            A_vals.append((k, k - 1,   4/dx2))
            A_vals.append((k, k - 2, -1/dx2))
            k += 1



ac = ContinousFracture(k1=10, k2=1, sigma=100, P1=5, P2=10, n_terms=1000)

ac.plot_analytical()