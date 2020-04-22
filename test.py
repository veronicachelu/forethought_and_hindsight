import jax
import jax.numpy as jnp
import numpy as np

f = lambda x: jnp.vdot(jnp.power(x, 2), jnp.array([2.0, 2.0]))
g = lambda x: 5 * x
h = lambda x: f(g(x))

x = np.array([1.0, 3.0])
# h = f(g(x))

dfdx = jax.grad(f)(g(x))
dhdx = jax.grad(h)(x)


# def d(f, g, x, v):
#     ff = jax.grad(lambda x: v * jax.grad(f)(g(x)))(x)
#     return ff

# def dvec(f, g, x):
#     ff = jax.grad(lambda x: jax.grad(f)(g(x)))(x)
#     return ff
#
# rez = dvec(f, g, x)
#
# print(rez)


def loss1(x):
    # partial = g(x)
    # _, vjp_fun = jax.vjp(f, partial)
    dfdg = jax.grad(f)(g(x))
    # l = vjp_fun(jnp.array(1.0))
    l = jnp.prod(x) * jnp.sum(dfdg)
    # return jnp.sum(l)
    return l

def loss2(x):
    # partial = g(x)
    _, vjp_fun = jax.vjp(f, g(x))
    l = jnp.sum(vjp_fun(jnp.array(jnp.prod(x))))
    return l

# v, grad = jax.value_and_grad(loss, 2)(f, g, x)
v, grad = jax.value_and_grad(loss2)(x)



print(rez2)