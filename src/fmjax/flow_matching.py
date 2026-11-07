import optax
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import diffrax



class AdaLNResNetBlock(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x, cond):
        scale_shift = nn.Dense(self.dim * 2)(cond)
        gamma, beta = jnp.split(scale_shift, 2, axis=-1)

        h = nn.LayerNorm()(x)
        h = h * (1.0 + gamma) + beta
        h = nn.silu(h)
        h = nn.Dense(self.dim)(h)

        h = nn.LayerNorm()(h)
        h = nn.silu(h)
        h = nn.Dense(self.dim)(h)

        return x + h

class ContextEncoder1D(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(self, x):
        # x shape: (batch_size, wave_steps)
        x = jnp.expand_dims(x, -1) 

        # 1D Convolution layers to extract features from spectrum
        x = nn.Conv(features=32, kernel_size=(5,), strides=(2,))(x)
        x = nn.silu(x)
        x = nn.Conv(features=64, kernel_size=(5,), strides=(2,))(x)
        x = nn.silu(x)
        x = nn.Conv(features=128, kernel_size=(5,), strides=(2,))(x)
        x = nn.silu(x)
        x = nn.Conv(features=256, kernel_size=(5,), strides=(2,))(x)
        x = nn.silu(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.out_dim)(x)
        return x




class ConditionalVectorFieldNetwork(nn.Module):
    hidden_dim: int = 512
    out_dim: int = 8  # (T_phot, T_spot, T_fac, f1, f2, f3, R_s, ln_f_var)
    context_emb_dim: int = 256

    @nn.compact
    def __call__(self, theta, t, context):
        t_features = jnp.concatenate([
            t,
            jnp.sin(2.0 * jnp.pi * t), jnp.cos(2.0 * jnp.pi * t),
            jnp.sin(4.0 * jnp.pi * t), jnp.cos(4.0 * jnp.pi * t),
            jnp.sin(8.0 * jnp.pi * t), jnp.cos(8.0 * jnp.pi * t)
        ], axis=-1)
        t_emb = nn.Dense(128)(t_features)
        t_emb = nn.silu(t_emb)

        c_emb = ContextEncoder1D(self.context_emb_dim)(context)
        c_emb = nn.silu(c_emb)

        cond = jnp.concatenate([t_emb, c_emb], axis=-1)
        cond = nn.Dense(self.hidden_dim)(cond)
        cond = nn.silu(cond)

        x = nn.Dense(self.hidden_dim)(theta)

        x = AdaLNResNetBlock(self.hidden_dim)(x, cond)
        x = AdaLNResNetBlock(self.hidden_dim)(x, cond)
        x = AdaLNResNetBlock(self.hidden_dim)(x, cond)
        x = AdaLNResNetBlock(self.hidden_dim)(x, cond)
        x = AdaLNResNetBlock(self.hidden_dim)(x, cond)

        x = nn.LayerNorm()(x)
        x = nn.silu(x)
        return nn.Dense(self.out_dim)(x)


def ot_flow_matching_loss(params, model_apply, theta1, theta0, t, context):
    t_expand = t[:, None]
    theta_t = (1.0 - t_expand) * theta0 + t_expand * theta1
    ut = theta1 - theta0
    vt = model_apply({'params': params}, theta_t, t_expand, context)
    return jnp.mean((vt - ut) ** 2)

@jax.jit
def train_step(state, theta1, theta0, t, context):
    def loss_fn(params):
        return ot_flow_matching_loss(params, state.apply_fn, theta1, theta0, t, context)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def sample_posterior(state, rng, num_samples, context_single, param_dim=8):
    theta0 = jax.random.normal(rng, (num_samples, param_dim))
    context_batch = jnp.tile(context_single, (num_samples, 1))

    def vector_field(t, theta, args):
        context = args
        t_batch = jnp.full((theta.shape[0], 1), t)
        return state.apply_fn({'params': state.params}, theta, t_batch, context)

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Euler()
    solution = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=1.0, dt0=0.001,
        y0=theta0, args=context_batch
    )
    return solution.ys[-1]

