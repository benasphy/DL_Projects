"""
Advanced NEAT CartPole trainer using neat-python
Inspired by modular structure of your best projects
"""
import neat
import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

class NEATCartpoleTrainer:
    def __init__(self, config_path="config-feedforward"):
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        self.stats = None
        self.best_genome = None
        self.best_fitness = 0
        self.fitness_history = []

    def eval_genomes(self, genomes, config):
        env = gym.make('CartPole-v1')
        for genome_id, genome in genomes:
            observation = env.reset()
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            fitness = 0.0
            done = False
            while not done:
                action = net.activate(observation)
                observation, reward, done, info = env.step(int(action[0] > 0.5))
                fitness += reward
            genome.fitness = fitness
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_genome = genome
        env.close()

    def train(self, generations=20):
        pop = neat.Population(self.config)
        pop.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        pop.add_reporter(self.stats)
        winner = pop.run(self.eval_genomes, generations)
        self.best_genome = winner
        self.fitness_history = self.stats.get_fitness_mean()
        # Save best genome
        with open("best_genome.pkl", "wb") as f:
            pickle.dump(winner, f)
        return winner

    def plot_fitness(self):
        if self.stats is None:
            return None
        gen = range(len(self.stats.most_fit_genomes))
        best_fitness = [g.fitness for g in self.stats.most_fit_genomes]
        mean_fitness = self.stats.get_fitness_mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(gen, best_fitness, label="Best Fitness")
        ax.plot(gen, mean_fitness, label="Mean Fitness")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("NEAT CartPole Training Progress")
        ax.legend()
        plt.tight_layout()
        return fig

    def run_best(self, render=False):
        if self.best_genome is None:
            with open("best_genome.pkl", "rb") as f:
                self.best_genome = pickle.load(f)
        env = gym.make('CartPole-v1', render_mode='human' if render else None)
        observation = env.reset()
        net = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)
        fitness = 0.0
        done = False
        steps = 0
        while not done and steps < 1000:
            action = net.activate(observation[0] if isinstance(observation, tuple) else observation)
            observation, reward, done, info = env.step(int(action[0] > 0.5))
            fitness += reward
            steps += 1
            if render:
                env.render()
        env.close()
        return fitness
