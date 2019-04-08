import matplotlib
matplotlib.use('Agg')
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import math
bin_dict = {'trk_pt':np.linspace(-0.5,0.5,100),
            'trk_eta':np.linspace(-2.1,2.1,50),
            'trk_phi':np.linspace(-math.pi,math.pi,50),
            'trk_dxyClosestPV':np.linspace(-0.5,0.5,100),
            'trk_dzClosestPV':np.linspace(-0.5,0.5,100),
            'trk_dzErr':np.linspace(-0.1,0.3, 40),
            'trk_dxyErr':np.linspace(-0.1, 0.3, 40),
            'trk_etaErr':np.linspace(-0.1, 0.3, 40),
            'trk_ptErr':np.linspace(-0.1, 0.3, 40),
            'trk_nChi2':np.linspace(-0.5, 2.0, 25)
            }

def plot_losses(G_loss, D_loss):
    epochs = range(len(G_loss))
    plt.plot(epochs, G_loss, label='Generative loss')
    plt.plot(epochs, D_loss, label='Discriminative loss')
    plt.plot(epochs, np.add(D_loss,G_loss), label='Total loss')
#    plt.yscale('log')
    plt.title('Losses at epoch '+str(len(G_loss)))
    plt.legend()
    plt.xlabel('Epochs')
    plt.savefig('Losses.pdf')
    plt.clf()

def plot_distribution(values, binning, epoch, title="", xlabel=""):
    plt.hist(values, bins=binning)
    plt.title(title+" at epoch "+str(epoch))
    plt.xlabel(xlabel)
    plt.savefig('plots/'+title+'_epoch_'+str(epoch)+'.pdf')
    plt.clf()

def plot_real_vs_gen(values_real, values_gen, name, title="", xlabel=""):
    binning = bin_dict[name]
    plt.hist(values_real, bins=binning, alpha=0.7, label='Real samples')
    plt.hist(values_gen, bins=binning, alpha=0.7, label='Generated samples')
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend()
    plt.savefig('plots_testGenerator/'+name+'.pdf')
    plt.clf()