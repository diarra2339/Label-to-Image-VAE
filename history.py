# Class from deeplib

from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class History:

    def __init__(self, logs_list=None):
        self.history = defaultdict(list)

        if logs_list is not None:
            for logs in logs_list:
                self.save(logs)

    def save(self, logs):
        for k, v in logs.items():
            self.history[k].append(v)

    def display_kl(self):
        pass

    def display_recons(self):
        pass

    def display(self, display_gamma=False):
        epoch = len(self.history['recons_loss'])
        epochs = list(range(1, epoch + 1))

        num_plots = 2
        _, axes = plt.subplots(num_plots, 1, sharex=True)
        plt.tight_layout()

        axes[0].set_ylabel('Loss')
        axes[0].plot(epochs, self.history['recons_loss'], label='Reconstruction loss')
        axes[0].plot(epochs, self.history['kl_loss'], label='KL loss')
        axes[0].legend()

        if display_gamma and 'gamma' in self.history:
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('Gamma')
            axes[1].plot(epochs, self.history['gamma'], label='Gamma')
            axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            axes[0].set_xlabel('Epochs')
            axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()

    def display_loss(self):
        epoch = len(self.history['loss'])
        epochs = list(range(1, epoch + 1))

        plt.tight_layout()

        plt.plot(epochs, self.history['loss'], label='Train')
        plt.plot(epochs, self.history['val_loss'], label='Validation')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()
