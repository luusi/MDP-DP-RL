from setuptools import setup, find_packages

setup(name='mdp_dp_rl',
      version='0.1.0',
      description='Markov Decision Processes, Dynamic Programming and Reinforcement Learning .',
      url='http://github.com/luusi/MDP-DP-RL',
      author='Luciana Silo',
      author_email='silo.1586010@studenti.uniroma1.it',
      license='MIT',
      packages=find_packages(include=['mdp_dp_rl*']),
      zip_safe=False,
      install_requires=[
            "numpy",
            "scipy",
            "matplotlib"
      ]
      )
