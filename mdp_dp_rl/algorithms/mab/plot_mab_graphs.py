from typing import NoReturn
from operator import itemgetter
from mdp_dp_rl.processes.mab_env import MabEnv
from mdp_dp_rl.algorithms.mab.epsilon_greedy import EpsilonGreedy
from mdp_dp_rl.algorithms.mab.ucb1 import UCB1
from mdp_dp_rl.algorithms.mab.ts_gaussian import ThompsonSamplingGaussian
from mdp_dp_rl.algorithms.mab.ts_bernoulli import ThompsonSamplingBernoulli
from mdp_dp_rl.algorithms.mab.gradient_bandits import GradientBandits
from numpy import arange
import matplotlib.pyplot as plt


def plot_gaussian_algorithms() -> NoReturn:
    mean_vars_data = [
        (0., 10.),
        (2., 20.),
        (4., 1.),
        (6., 8.),
        (8., 4.),
        (9., 6.),
        (10., 4.)]
    mu_star = max(mean_vars_data, key=itemgetter(0))[0]

    steps = 500
    episodes = 500

    eps = 0.3
    eps_hl = 400

    ci = 5
    mi = mu_star * 3.

    ts_mi = 0.
    ts_si = 10.

    lr = 0.1
    lr_decay = 20.

    me = MabEnv.get_gaussian_mab_env(mean_vars_data)

    greedy_opt_init = EpsilonGreedy(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=0.,
        epsilon_half_life=1e8,
        count_init=ci,
        mean_init=mi
    )
    eps_greedy = EpsilonGreedy(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=1e8,
        count_init=0,
        mean_init=0.
    )
    decay_eps_greedy = EpsilonGreedy(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=eps_hl,
        count_init=0,
        mean_init=0.
    )
    ts = ThompsonSamplingGaussian(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        init_mean=ts_mi,
        init_stdev=ts_si
    )
    grad_bandits = GradientBandits(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        learning_rate=lr,
        learning_rate_decay=lr_decay
    )

    plot_colors = ['r', 'b', 'g', 'k', 'y']
    labels = [
        'Greedy, Optimistic Initialization',
        '$\epsilon$-Greedy',
        'Decaying $\epsilon$-Greedy',
        'Thompson Sampling',
        'Gradient Bandits'
    ]

    exp_cum_regrets = [
        greedy_opt_init.get_expected_cum_regret(mu_star),
        eps_greedy.get_expected_cum_regret(mu_star),
        decay_eps_greedy.get_expected_cum_regret(mu_star),
        ts.get_expected_cum_regret(mu_star),
        grad_bandits.get_expected_cum_regret(mu_star)
    ]

    x_vals = range(1, steps + 1)
    for i in range(len(exp_cum_regrets)):
        plt.plot(exp_cum_regrets[i], color=plot_colors[i], label=labels[i])
    plt.xlabel("Time Steps", fontsize=20)
    plt.ylabel("Expected Cumulative Regret", fontsize=20)
    plt.title("Cumulative Regret Curves", fontsize=25)
    plt.xlim(xmin=x_vals[0], xmax=x_vals[-1])
    plt.ylim(ymin=0.0)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=15)
    plt.show()

    exp_act_counts = [
        greedy_opt_init.get_expected_action_counts(),
        eps_greedy.get_expected_action_counts(),
        decay_eps_greedy.get_expected_action_counts(),
        ts.get_expected_action_counts(),
        grad_bandits.get_expected_action_counts()
    ]
    index = arange(len(me.arms_sampling_funcs))
    spacing = 0.4
    width = (1 - spacing) / len(exp_act_counts)

    for i in range(len(exp_act_counts)):
        plt.bar(
            index - (1 - spacing) / 2 + (i - 1.5) * width,
            exp_act_counts[i],
            width,
            color=plot_colors[i],
            label=labels[i]
        )
    plt.xlabel("Arms", fontsize=20)
    plt.ylabel("Expected Counts of Arms", fontsize=20)
    plt.title("Arms Counts Plot", fontsize=25)
    plt.xticks(
        index - 0.3,
        ["$\mu$=%.1f,$\sigma$=%.1f" % (m, s) for m, s in mean_vars_data]
    )
    plt.legend(loc='upper left', fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_bernoulli_algorithms() -> NoReturn:
    probs_data = [0.1, 0.2, 0.4, 0.5, 0.6, 0.75, 0.8, 0.85, 0.9]
    mu_star = max(probs_data)

    steps = 500
    episodes = 500

    eps = 0.3
    eps_hl = 400

    ci = 5
    mi = mu_star * 3.

    ucb_alpha = 4.0

    lr = 0.5
    lr_decay = 20.

    me = MabEnv.get_bernoulli_mab_env(probs_data)

    greedy_opt_init = EpsilonGreedy(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=0.,
        epsilon_half_life=1e8,
        count_init=ci,
        mean_init=mi
    )
    eps_greedy = EpsilonGreedy(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=1e8,
        count_init=0,
        mean_init=0.
    )
    decay_eps_greedy = EpsilonGreedy(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=eps_hl,
        count_init=0,
        mean_init=0.
    )
    ucb1 = UCB1(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        bounds_range=1.0,
        alpha=ucb_alpha
    )
    ts = ThompsonSamplingBernoulli(
        mab=me,
        time_steps=steps,
        num_episodes=episodes
    )
    grad_bandits = GradientBandits(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        learning_rate=lr,
        learning_rate_decay=lr_decay
    )

    plot_colors = ['r', 'b', 'g', 'y', 'k', 'c']
    labels = [
        'Greedy, Optimistic Initialization',
        '$\epsilon$-Greedy',
        'Decaying $\epsilon$-Greedy',
        'UCB1',
        'Thompson Sampling',
        'Gradient Bandits'
    ]

    exp_cum_regrets = [
        greedy_opt_init.get_expected_cum_regret(mu_star),
        eps_greedy.get_expected_cum_regret(mu_star),
        decay_eps_greedy.get_expected_cum_regret(mu_star),
        ucb1.get_expected_cum_regret(mu_star),
        ts.get_expected_cum_regret(mu_star),
        grad_bandits.get_expected_cum_regret(mu_star)
    ]

    x_vals = range(1, steps + 1)
    for i in range(len(exp_cum_regrets)):
        plt.plot(exp_cum_regrets[i], color=plot_colors[i], label=labels[i])
    plt.xlabel("Time Steps", fontsize=20)
    plt.ylabel("Expected Cumulative Regret", fontsize=20)
    plt.title("Cumulative Regret Curves", fontsize=25)
    plt.xlim(xmin=x_vals[0], xmax=x_vals[-1])
    plt.ylim(ymin=0.0)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=15)
    plt.show()

    exp_act_counts = [
        greedy_opt_init.get_expected_action_counts(),
        eps_greedy.get_expected_action_counts(),
        decay_eps_greedy.get_expected_action_counts(),
        ucb1.get_expected_action_counts(),
        ts.get_expected_action_counts(),
        grad_bandits.get_expected_action_counts()
    ]
    index = arange(len(me.arms_sampling_funcs))
    spacing = 0.4
    width = (1 - spacing) / len(exp_act_counts)

    for i in range(len(exp_act_counts)):
        plt.bar(
            index - (1 - spacing) / 2 + (i - 1.5) * width,
            exp_act_counts[i],
            width,
            color=plot_colors[i],
            label=labels[i]
        )
    plt.xlabel("Arms", fontsize=20)
    plt.ylabel("Expected Counts of Arms", fontsize=20)
    plt.title("Arms Counts Plot", fontsize=25)
    plt.xticks(
        index - 0.2,
        ["$p$=%.2f" % p for p in probs_data]
    )
    plt.legend(loc='upper left', fontsize=15)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_gaussian_algorithms()
    plot_bernoulli_algorithms()
