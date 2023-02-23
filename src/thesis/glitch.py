import jax.numpy as jnp

from tinygp import kernels, GaussianProcess

from .distributions import JointDistribution, Normal, Uniform, CircularUniform

PARAM_NAMES = [
    'delta_nu',
    'epsilon',
    'log_tau_he',
    'log_beta_he',
    'log_gamma_he',
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
            2*(log_tau_he.mean + jnp.log(0.1*jnp.pi)) + jnp.log(8.0),  # beta ~ 8 * (np.pi * 0.1 * tau)**2
            jnp.sqrt(0.25 + 4*log_tau_he.variance)  # 0.5
        ))
        # beta = 8 * pi**2 * delta**2
        
        distributions.setdefault("log_gamma_he", Normal(
            0.5*(log_beta_he.mean -  jnp.log(jnp.pi)) + jnp.log(0.5*0.1), # gamma ~ 1/2 * 0.1 * sqrt(beta/pi)
            jnp.sqrt(0.64 + 0.25*log_beta_he.variance)  # 0.8
        ))
        # alpha_he = (dgamma / gamma)_min = gamma_he / sqrt(2pi) / delta_he
        # a_he = delta_nu * gamma_he
        
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


class GlitchModel:
    def __init__(self, prior, *, n, nu, nu_err=None):
        self.prior = prior
        self.n = n
        self.nu = nu
        self.nu_err = nu_err
    
    @staticmethod
    def _oscillation(nu, tau, phi):
        return jnp.sin(4 * jnp.pi * tau * nu + phi)

    def smooth_component(self, params, n):
        return params["delta_nu"] * (n + params["epsilon"])
    
    def helium_amp(self, params, nu):
        return params["delta_nu"] * jnp.exp(params["log_gamma_he"]) * nu \
            * jnp.exp(- jnp.exp(params["log_beta_he"]) * nu**2)

    def helium_osc(self, params, nu):
        return self._oscillation(nu, jnp.exp(params["log_tau_he"]), params["phi_he"])

    def helium_glitch(self, params, nu):
        return self.helium_amp(params, nu) * self.helium_osc(params, nu)

    def bcz_amp(self, params, nu):
        return params["delta_nu"] * jnp.exp(params["log_alpha_cz"]) / nu**2

    def bcz_osc(self, params, nu):
        return self._oscillation(nu, jnp.exp(params["log_tau_cz"]), params["phi_cz"])

    def bcz_glitch(self, params, nu):
        return self.bcz_amp(params, nu) * self.bcz_osc(params, nu)

    def glitch(self, params, nu):
        return self.helium_glitch(params, nu) + self.bcz_glitch(params, nu)

    def build_gp(self, params):
        # kernel = jnp.exp(params["log_amp"]) * kernels.ExpSquared(jnp.exp(params["log_scale"]))
        kernel = 0.5*params["delta_nu"] * kernels.ExpSquared(5.0)
        
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
