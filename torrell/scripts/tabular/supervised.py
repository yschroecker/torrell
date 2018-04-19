import environments.environment
import numpy as np
import tqdm

from torrell.scripts.tabular import policy


# noinspection PyPep8Naming
def train_supervised(env: environments.environment, X: np.ndarray, y: np.ndarray, num_epochs: int,
                     batch_size: int, learning_rate: float, lr_decay: float) -> policy.TabularPolicy:
    supervised_policy = policy.TabularPolicy(env.num_states, env.num_actions)
    for _ in tqdm.trange(num_epochs):
        learning_rate *= lr_decay
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        for batch in range(int(np.ceil(len(X)/batch_size))):
            batch_indices = indices[batch*batch_size:(batch + 1)*batch_size]
            gradient = supervised_policy.log_gradient(X[batch_indices], y[batch_indices])
            supervised_policy.parameters += learning_rate * gradient
    return supervised_policy
