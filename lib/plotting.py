
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'aliceblue'
mpl.rcParams['text.color'] = 'black'


def create_line_chart(unet_dice, gan_dice, classes, filename):
    fig, ax = plt.subplots()
    ax.plot(classes, unet_dice, marker="o", linewidth=2, markersize=8, label="3D UNET", color='firebrick')
    ax.plot(classes, gan_dice, marker="v", linewidth=2, markersize=8, label="3D GAN", color='darkblue')
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid( which='major', color='gray', linewidth=1.5)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right')
    fig.tight_layout()
    plt.savefig(filename)


classes = ['Cortical \n gray matter', 'Basal ganglia', 'White matter', 'White matter \n lesions', 'Cerebrospinal \n fluid', 'Ventricles',
           'Cerebellum', 'Brain stem']
unet_dice = [0.795, 0.32, 0.83, 0.075, 0.745, 0.785, 0.90, 0.86]
gan_dice = [0.84, 0.385, 0.845, 0.085, 0.76, 0.805, 0.91, 0.875]

unet_vol = [0.955, 0.405, 0.965, 0.15, 0.865, 0.87, 0.955, 0.89]
gan_vol = [0.96, 0.475, 0.975, 0.175, 0.86, 0.88, 0.96, 0.915]




create_line_chart(unet_dice, gan_dice, classes, 'dice.png')
create_line_chart(unet_vol, gan_vol, classes, 'vol.png')