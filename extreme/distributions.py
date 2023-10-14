import numpy as np
from scipy import stats
from scipy.special import hyp2f1
from scipy.special import gammainc
import scipy.special as spe


"""
\begin{align}
\label{eq:g:bctm}
    \bctm_a(\alpha) = \frac{g_a(\alpha) - g_0(\alpha)}{a\alpha}=\frac{g_a(\alpha)/\alpha -1}{a}, \mbox{ with }
    g_a(\alpha)=\int_{q(\alpha)}^\infty t^af(t)\dt.
\end{align}
"""

class Frechet2OC():
    def __init__(self):
        self.cste_slowvar = None  # L_\infty
        self.evi = None  # extreme value index
        self.rho = None  # J order parameters
        return

    def cdf(self, x):
        raise ("No distribution called")

    def sf(self, x):
        """survival function """
        return 1 - self.cdf(x)

    def ppf(self, u):
        """quantile function"""
        raise ("No distribution called")

    def isf(self, u):
        """inverse survival function"""
        return self.ppf(1 - u)

    def tail_ppf(self, x):
        """tail quantile function U(x)=q(1-1/x)"""
        return self.isf(1/x)

    def norm_ppf(self, u):
        "quantile normalized X>=1"
        return self.isf((1 - u) * self.sf(1))

    def epsilon_func(self, x):
        """RV_rho from Karamata representation"""
        return x**self.sop * self.ell_slowvar(x)

    def L_slowvar(self, x):
        raise ("No distribution called")

    def ell_slowvar(self, x):
        raise ("No distribution called")

    def expected_shortfall(self, u):
        """ES(u)"""
        raise ("No distribution called")

    def conditional_tail_moment(self, u, a):
        """CTM_a(u)"""
        raise ("No distribution called")

    def g_ctm(self, u, a):
        """g_a(u)"""
        raise ("No distribution called")

    def log_transform(self, u):
        """log q(u)"""
        raise ("No distribution called")

    def box_conditional_tail_moment(self, u, a):
        """BCTM(u)"""
        if a>0:
            return (self.conditional_tail_moment(u, a) - 1) / a
        else:
            raise ValueError("Please enter a positive value for a.")



class Pareto(Frechet2OC):
    def __init__(self, evi, xm=1):
        super(Pareto, self).__init__()
        self.evi = evi
        self.rho=0
        self.xm = xm
        return

    def cdf(self, x):
        return 1 - (self.xm/x)**(1/self.evi) * (x>=self.xm)

    def ppf(self, u):
        return self.xm * (1 - u) ** (-self.evi)

    def conditional_tail_moment(self, u, a):
        theta = 1/self.evi
        return (self.xm * theta) / ((theta - a) * (u) ** (a / theta))

    def g_ctm(self, u, a):
        theta = 1/self.evi
        return (self.xm * theta) / ((theta - a) * (u) ** (a/theta-1))

    def log_transform(self, u):
        return




class Burr(Frechet2OC):
    def __init__(self, evi, rho):
        super(Burr, self).__init__()
        self.evi = evi
        self.rho = np.array(rho)

        self.cste_slowvar = 1
        return

    def cdf(self, x):
        return 1 - (1 + x ** (- self.rho / self.evi)) ** (1 / self.rho)

    def ppf(self, u):
        c = - self.rho / self.evi
        k = -1/self.rho
        return (((1 - u) ** self.rho) - 1) ** (- self.evi / self.rho)
        # return (((1 - u) ** (-1/k)) - 1) ** (1/c)

    def L_slowvar(self, x):
        return (1 - x**self.rho) ** (-self.evi / self.rho)

    def ell_slowvar(self, x):
        return self.evi / (1-x**self.rho)

    def conditional_tail_moment(self, u, a):
        c = - self.rho / self.evi  # zeta
        k = -1/self.rho  # theta
        return ((c*k)/(c*k - a)) * self.ppf(1-u)**a * hyp2f1(-a/c, 1, k-(a/c)+1, 1 / (1-(u)**(-1/k)))

    def g_ctm(self, u, a):
        c = - self.rho / self.evi  # zeta
        k = -1/self.rho  # theta
        return ((c*k*u)/(c*k - a)) * self.ppf(1-u)**a * hyp2f1(-a/c, 1, k-(a/c)+1, 1 / (1-(u)**(-1/k)))




class InverseGamma(Frechet2OC):
    def __init__(self, evi):
        super(InverseGamma, self).__init__()
        self.evi = evi
        self.rho = np.array(-self.evi)
        self.law = stats.invgamma(1/self.evi)
        return
    def cdf(self, x):
        return self.law.cdf(x)

    def ppf(self, u):
        return self.law.ppf(u)

    def conditional_tail_moment(self, u, a):
        zeta = 1/self.evi
        return spe.gammainc(zeta - a , 1/self.ppf(1-u)) * spe.gamma(zeta-a) / (spe.gamma(zeta) * (u))

    def g_ctm(self, u, a):
        zeta = 1/self.evi
        return spe.gammainc(zeta - a , 1/self.ppf(1-u)) * spe.gamma(zeta-a) / (spe.gamma(zeta))


class Frechet(Frechet2OC):
    def __init__(self, evi):
        super(Frechet, self).__init__()
        self.evi = evi
        self.rho = np.array([-1.])
        self.law = stats.invweibull(1 / self.evi)
        return

    def cdf(self, x):
        return self.law.cdf(x)

    def ppf(self, u):
        return self.law.ppf(u)

    def conditional_tail_moment(self, u, a):
        theta = 1/self.evi
        return spe.gammainc(1-a/theta, -np.log(1-u)) * spe.gamma(1-a/theta) / u

    def g_ctm(self, u, a):
        theta = 1/self.evi
        return spe.gammainc(1-a/theta, -np.log(1-u)) * spe.gamma(1-a/theta)


class Fisher(Frechet2OC):
    def __init__(self, evi):
        super(Fisher, self).__init__()
        self.evi = evi
        self.rho = np.array([-2./self.evi])
        
        self.nu1 = 1
        self.nu2 = 2/self.evi
        self.law = stats.f(1, self.nu2)
        return
    
    def cdf(self, x):
        return self.law.cdf(x)

    def ppf(self, u):
        return self.law.ppf(u)

    def conditional_tail_moment(self, u, a):
        term1 = ((self.nu1/self.nu2)**(-(self.nu2)/2)) / u
        term2 = 1/(spe.beta(self.nu1/2, self.nu2/2) * (self.nu2/2 - a))
        term3 = self.ppf(1-u)**(a-(self.nu2/2))
        term4 = spe.hyp2f1((self.nu1+self.nu2)/2, self.nu2/2 -a, self.nu2/2 -a + 1, - self.nu2/(self.nu1*self.ppf(1-u)))
        return term1*term2*term3*term4

    def g_ctm(self, u, a):
        term1 = ((self.nu1/self.nu2)**(-(self.nu2)/2))
        term2 = 1/(spe.beta(self.nu1/2, self.nu2/2) * (self.nu2/2 - a))
        term3 = self.ppf(1-u)**(a-(self.nu2/2))
        term4 = spe.hyp2f1((self.nu1+self.nu2)/2, self.nu2/2 -a, self.nu2/2 -a + 1, - self.nu2/(self.nu1*self.ppf(1-u)))
        return term1*term2*term3*term4
    


class GPD(Frechet2OC):
    def __init__(self, evi):
        super(GPD, self).__init__()
        self.evi = evi
        self.rho = np.array([-self.evi])
        self.law = stats.genpareto(self.evi)
        return
    def cdf(self, x):
        return self.law.cdf(x)

    def ppf(self, u):
        return self.law.ppf(u)

    def conditional_tail_moment(self, u, a):
        num= (u**(-self.evi) -1)**(a-(1/self.evi))
        denum = u*self.evi**a*(1-a*self.evi)
        return num/denum * spe.hyp2f1(1+1/self.evi, 1/self.evi-a, 1/self.evi - a + 1, 1/(1-u**(-self.evi)))

    def g_ctm(self, u, a):
        num= (u**(-self.evi) -1)**(a-(1/self.evi))
        denum = self.evi**a*(1-a*self.evi)
        return num/denum * spe.hyp2f1(1+1/self.evi, 1/self.evi-a, 1/self.evi - a + 1, 1/(1-u**(-self.evi)))




class Student(Frechet2OC):
    def __init__(self, evi):
        super(Student, self).__init__()
        self.evi = evi
        self.rho = np.array([-2*self.evi])
        self.law = stats.t(1/self.evi)
        return
    def cdf(self, x):
        return 2 * self.law.cdf(x) - 1

    # def cdf(self, x):
    #     return self.law.cdf(x)

    def ppf(self, u):
        return self.law.ppf((u+1)/2)

    def pdf(self, x):
        return 2*self.law.pdf(x)

    def conditional_tail_moment(self, u, a):
        nu = 1/self.evi
        cste = 1 / (np.sqrt(nu) * spe.beta(nu / 2, 0.5))
        return 2*cste/2 * nu**((a+1)/2) * spe.beta((nu-a)/2, (a+1)/2) * spe.betainc((nu-a)/2, (a+1)/2, 1/(1+(self.ppf(1-u)**2)/nu)) / u


    def g_ctm(self, u, a):
        nu = 1/self.evi
        cste = 1 / (np.sqrt(nu) * spe.beta(nu/2, 0.5))
        return 2*cste/2 * nu**((a+1)/2) * spe.beta((nu-a)/2, (a+1)/2) * spe.betainc((nu-a)/2, (a+1)/2, 1/(1+(self.ppf(1-u)**2)/nu))








