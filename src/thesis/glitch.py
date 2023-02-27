import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from tinygp import kernels, GaussianProcess

from .jaxdaw.distributions import Distribution, JointDistribution, Normal, Uniform, CircularUniform
from .jaxdaw.model import Model

PARAM_NAMES = [
    'delta_nu',
    'epsilon',
    'log_tau_he',
    'log_beta_he',
    'log_alpha_he',
    'log_alpha_cz',
    'log_tau_cz',
    'phi_he',
    'phi_cz',
    'log_sigma'
]

def estimate_asy(n, nu, nu_err=None):
    w = None if nu_err is None else jnp.broadcast_to(1/nu_err, nu.shape)
    popt, pcov = jnp.polyfit(n.astype(float), nu, 1, w=w, cov=True)
    perr = jnp.sqrt(jnp.diag(pcov))
    delta_nu = Normal(popt[0], perr[0])
    epsilon = Normal(popt[1]/delta_nu.mean, perr[1]/delta_nu.mean)
    return delta_nu, epsilon


class GlitchPrior(JointDistribution):

    def __init__(self, delta_nu, epsilon, **distributions):

        diff_keys = set(distributions.keys()) - set(PARAM_NAMES)
        
        if len(diff_keys) > 0:
            raise KeyError(
                f"Keys {diff_keys} are not valid parameter names for {self.__class__.__name__}. " +
                f"Dictionary 'distributions' may contain any of the following keys: {PARAM_NAMES}"
            )
        
        distributions["delta_nu"] = delta_nu
        distributions["epsilon"] = epsilon
        
        log_tau_he = distributions.setdefault("log_tau_he", Normal(
            jnp.log(0.2/2/delta_nu.mean),  # tau ~ 0.2 tau0
            jnp.sqrt(0.04 + delta_nu.variance/delta_nu.mean**2)  # 0.2
        ))
        
        log_beta_he = distributions.setdefault("log_beta_he", Normal(
            2*(log_tau_he.mean + jnp.log(0.08*jnp.pi)) + jnp.log(8.0),  # beta ~ 8 * (np.pi * 0.08 * tau)**2
            # jnp.sqrt(0.25 + 4*log_tau_he.variance)  # 0.5
            2*jnp.sqrt(log_tau_he.variance)
        ))
        # beta = 8 * pi**2 * delta**2
        # what if we introduced lam = 1/beta to discourage large beta?

        distributions.setdefault("log_alpha_he", Normal(
            0.5*(log_beta_he.mean -  jnp.log(jnp.pi)) + jnp.log(0.5*0.1), # gamma ~ 1/2 * 0.1 * sqrt(beta/pi)
            # jnp.sqrt(0.64 + 0.25*log_beta_he.variance)  # 0.8
            0.5*jnp.sqrt(log_beta_he.variance)
        ))
        # alpha_he = (dgamma / gamma)_min = alpha_he / sqrt(2pi) / delta_he
        # a_he = delta_nu * alpha_he
        
        distributions.setdefault("log_alpha_cz", Normal(
            jnp.log(delta_nu.mean*30.0), 
            jnp.sqrt(0.64 + delta_nu.variance/delta_nu.mean**2)  # 0.8
        ))
        
        distributions.setdefault("log_tau_cz", Normal(
            jnp.log(0.6/2/delta_nu.mean),  # tau ~ 0.6 tau0
            jnp.sqrt(0.04 + delta_nu.variance/delta_nu.mean**2)  # 0.2
        ))
        
        distributions.setdefault("phi_he", CircularUniform())
        distributions.setdefault("phi_cz", CircularUniform())
                
        distributions.setdefault("log_sigma", Normal(jnp.log(0.01), 2.0))
        
        super().__init__(distributions)


class GlitchModel(Model):

    symbols = {
        'delta_nu': r"$\Delta\nu$",
        'epsilon': r"$\varepsilon$",
        'log_tau_he': r"$\ln(\tau_\mathrm{He})$",
        'log_beta_he': r"$\ln(\beta_\mathrm{He})$",
        'log_alpha_he': r"$\ln(\alpha_\mathrm{He})$",
        'log_alpha_cz': r"$\ln(\alpha_\mathrm{cz})$",
        'log_tau_cz': r"$\ln(\tau_\mathrm{cz})$",
        'phi_he': r"$\phi_\mathrm{He}$",
        'phi_cz': r"$\phi_\mathrm{cz}$",
        'log_sigma': r"$\ln(\sigma)$",
    }
    
    def __init__(self, prior: JointDistribution, *, 
                 n, nu, nu_err=None, kernel_scale=5.0, kernel_amplitude=0.5):
        super().__init__(prior)
        self.n = n
        self.nu = nu
        self.nu_err = nu_err
        self.kernel_scale = kernel_scale  # In units of radial order
        self.kernel_amplitude = kernel_amplitude  # As a fraction of delta_nu

    @staticmethod
    def _oscillation(nu, tau, phi):
        return jnp.sin(4 * jnp.pi * tau * nu + phi)

    @staticmethod
    def smooth_component(params, n):
        return params["delta_nu"] * (n + params["epsilon"])

    @staticmethod
    def helium_amp(params, nu):
        return params["delta_nu"] * jnp.exp(params["log_alpha_he"]) * nu \
            * jnp.exp(- jnp.exp(params["log_beta_he"]) * nu**2)

    @staticmethod
    def helium_osc(params, nu):
        return GlitchModel._oscillation(nu, jnp.exp(params["log_tau_he"]), params["phi_he"])

    @staticmethod
    def helium_glitch(params, nu):
        return GlitchModel.helium_amp(params, nu) * GlitchModel.helium_osc(params, nu)

    @staticmethod
    def bcz_amp(params, nu):
        return params["delta_nu"] * jnp.exp(params["log_alpha_cz"]) / nu**2

    @staticmethod
    def bcz_osc(params, nu):
        return GlitchModel._oscillation(nu, jnp.exp(params["log_tau_cz"]), params["phi_cz"])

    @staticmethod
    def bcz_glitch(params, nu):
        return GlitchModel.bcz_amp(params, nu) * GlitchModel.bcz_osc(params, nu)

    @staticmethod
    def glitch(params, nu):
        return GlitchModel.helium_glitch(params, nu) + GlitchModel.bcz_glitch(params, nu)

    def build_gp(self, params):
        # kernel = jnp.exp(params["log_amp"]) * kernels.ExpSquared(jnp.exp(params["log_scale"]))
        kernel = self.kernel_amplitude * params["delta_nu"] \
            * kernels.ExpSquared(self.kernel_scale)
        
        def mean(n):
            nu_sm = self.smooth_component(params, n)
            dnu = self.glitch(params, nu_sm)
            return nu_sm + dnu
        
        diag = jnp.exp(2*params["log_sigma"])
#         diag = 1e-4
        if self.nu_err is not None:
            diag += self.nu_err**2
        
        return GaussianProcess(kernel, self.n, mean=mean, diag=diag)

    def predict(self, params, n, **kwargs):
        gp = self.build_gp(params)
        _, cond = gp.condition(self.nu, n, **kwargs)
        return cond.loc, cond.variance

    def sample(self, key, params, n, shape=(), **kwargs):
        # sample from the predictive
        gp = self.build_gp(params)
        _, cond = gp.condition(self.nu, n, **kwargs)
        return cond.sample(key, shape=shape)
        
    def log_likelihood(self, params):
        gp = self.build_gp(params)
        return gp.log_probability(self.nu)

    def prior_transform(self, uparams):
        return self.prior.transform(uparams)
    
    def log_probability(self, params):
        logp = self.prior.log_probability(params)
        return logp + self.log_likelihood(params)

    def plot_glitch(self, samples, kind="both", intervals=None, draws=None, 
                    color=None, alpha=0.33, ax=None):
        if kind == "both":
            func = self.glitch
        elif kind == "he":
            func = self.helium_glitch
        elif kind == "cz":
            func = self.bcz_glitch
        else:
            raise ValueError(f"Kind '{kind}' is not one of ['both', 'he', 'cz'].")
        
        if ax is None:
            ax = plt.gca()
        
        properties = {"color": color}
        if color is None:
            properties.update(next(ax._get_lines.prop_cycler))
            
        nu_pred = jnp.linspace(self.nu.min(), self.nu.max(), 201)
        dnu_pred = jax.vmap(func, in_axes=(0, None))(samples, nu_pred)
        
        if draws is None:
            dnu_med = np.median(dnu_pred, axis=0)

            ax.plot(nu_pred, dnu_med, label=kind, **properties)
            
            if intervals is not None:
                # intervals is an iterable of two-tuples
                for interval in intervals:
                    assert isinstance(interval, tuple)  # raise errors here
                    assert len(interval) == 2
                    dnu_lower, dnu_upper = np.quantile(dnu_pred, interval, axis=0)
                    ax.fill_between(nu_pred, dnu_lower, dnu_upper, alpha=alpha, **properties)
        else:
            thin = samples["delta_nu"].shape[0] // draws
            y = dnu_pred[::thin]
            x = jnp.broadcast_to(nu_pred, y.shape)
            ax.plot(x.T, y.T, alpha=alpha, **properties)
            
        return ax

    def plot_echelle(self, key, samples, kind="full", delta_nu=None,
                     intervals=None, draws=None, max_samples=None, 
                     color=None, alpha=0.33, ax=None):
        
        if kind == "full":
            func = lambda params, n: self.sample(key, params, n)
        elif kind == "asy":
            func = self.smooth_component
        elif kind == "gp":
            func = lambda params, n: self.smooth_component(params, n) \
                + self.sample(key, params, n, include_mean=False) 
        else:
            raise ValueError(f"Kind '{kind}' is not one of ['full', 'asy', 'gp'].")

        if delta_nu is None:
            delta_nu = samples["delta_nu"].mean()

        if ax is None:
            ax = plt.gca()

        if self.nu_err is None:
            ax.plot(self.nu%delta_nu, self.nu, ".")
        else:
            ax.errorbar(self.nu%delta_nu, self.nu, xerr=self.nu_err, fmt=".")
        
        if draws is not None:
            max_samples = draws
        elif max_samples is None:
            max_samples = 1000

        thin = samples["delta_nu"].shape[0] // max_samples
        thinned_samples = jax.tree_map(lambda x: x[::thin], samples)
        n_pred = jnp.linspace(self.n.min(), self.n.max(), 201)
        nu_pred = jax.vmap(func, in_axes=(0, None))(thinned_samples, n_pred)
        x_pred = (nu_pred - n_pred*delta_nu) % delta_nu
            
        properties = {"color": color}
        if color is None:
            properties.update(next(ax._get_lines.prop_cycler))
        
        if draws is None:
            x_med = np.median(x_pred, axis=0)
            nu_med = np.median(nu_pred, axis=0)
            ax.plot(x_med, nu_med, **properties)

            if intervals is not None:
                # intervals is an iterable of two-tuples
                for interval in intervals:
                    assert isinstance(interval, tuple)  # raise errors here
                    assert len(interval) == 2
                    x_lower, x_upper = np.quantile(x_pred, interval, axis=0)
                    ax.fill_betweenx(nu_med, x_lower, x_upper, alpha=alpha, **properties)
        else:
            ax.plot(x_pred.T, nu_pred.T, alpha=alpha, **properties)
        
        return ax
