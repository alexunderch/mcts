import jax
import jax.numpy as jnp
from functools import partial

from tree import Tree, get_state


def act_randomly(key, obs, mask):
    """Ignore observation and choose randomly from legal actions"""
    del obs
    probs = mask / mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
    return jax.random.categorical(key, logits=logits, axis=-1)


act_randomly = jax.jit(act_randomly)
batch_act_randomly = jax.jit(jax.vmap(act_randomly, in_axes=(0, 0, 0)))


@partial(
    jax.jit,
    static_argnums=(
        2,
        3,
    ),
)
def act_uct(
    tree: Tree,
    node: jax.Array,
    c: jax.Array = jnp.sqrt(2.0),
    eps: jax.Array = jnp.array(1e-8, dtype=jnp.float32),
):
    """Select action using UCT."""
    children = tree.children[node]
    state = get_state(tree, node)

    visited = children != Tree.UNVISITED

    exploit = -tree.values[children]
    explore = jnp.sqrt(jnp.log(tree.visits[node] + 1.0) / (tree.visits[children] + eps))

    uct = exploit + c * explore

    uct = jnp.where(state.legal_action_mask & (~visited), jnp.inf, uct)
    uct = jnp.where(~state.legal_action_mask, -jnp.inf, uct)
    return jnp.argmax(uct)


def act_greedy(tree: Tree, node: jax.Array):
    """Select action using greedy policy."""
    children = tree.children[node]
    visited = children != Tree.UNVISITED

    state = get_state(tree, node)
    scores = jnp.where(
        state.legal_action_mask & visited, tree.visits[children], -jnp.inf
    )
    return jnp.argmax(scores)


batch_act_greedy = jax.jit(jax.vmap(act_greedy, in_axes=(0, 0)))
act_greedy = jax.jit(act_greedy)
