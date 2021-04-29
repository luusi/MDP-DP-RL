from typing import Sequence, Tuple, List
from mdp_dp_rl.processes.mab_env import MabEnv
from operator import itemgetter
from numpy import ndarray, empty, exp
from numpy.random import choice
from mdp_dp_rl.algorithms.mab.mab_base import MABBase


class GradientBandits(MABBase):

    def __init__(
        self,
        mab: MabEnv,
        time_steps: int,
        num_episodes: int,
        learning_rate: float,
        learning_rate_decay: float
    ) -> None:
        if learning_rate <= 0 or learning_rate_decay <= 0:
            raise ValueError
        super().__init__(
            mab=mab,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.learning_rate: float = learning_rate
        self.learning_rate_decay: float = learning_rate_decay

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        scores: List[float] = [0.] * self.num_arms
        avg_reward: float = 0.

        for i in range(self.time_steps):
            max_score: float = max(scores)
            exp_scores: Sequence[float] = [exp(s - max_score) for s in scores]
            sum_exp_scores = sum(exp_scores)
            probs: Sequence[float] = [s / sum_exp_scores for s in exp_scores]
            action: int = choice(self.num_arms, p=probs)
            reward: float = self.mab_funcs[action]()
            avg_reward += (reward - avg_reward) / (i + 1)
            step_size: float = self.learning_rate *\
                (i / self.learning_rate_decay + 1) ** -0.5
            for j in range(self.num_arms):
                scores[j] += step_size * (reward - avg_reward) *\
                             ((1 if j == action else 0) - probs[j])

            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions


if __name__ == '__main__':
    mean_vars_data = [(9., 5.), (10., 2.), (0., 4.), (6., 10.), (2., 20.), (4., 1.)]
    mu_star = max(mean_vars_data, key=itemgetter(0))[0]
    steps = 200
    episodes = 1000
    lr = 0.1
    lr_decay = 20.0

    me = MabEnv.get_gaussian_mab_env(mean_vars_data)
    ucb1 = GradientBandits(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        learning_rate=lr,
        learning_rate_decay=lr_decay
    )
    exp_cum_regret = ucb1.get_expected_cum_regret(mu_star)
    print(exp_cum_regret)

    exp_act_count = ucb1.get_expected_action_counts()
    print(exp_act_count)

    ucb1.plot_exp_cum_regret_curve(mu_star)
