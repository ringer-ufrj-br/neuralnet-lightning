__all__ = ['plot_training_curves']

def plot_training_curves( context ):

    import matplotlib.pyplot as plt
    start_epoch=3
    sort = context.getHandler('sort')
    init = context.getHandler('init')
    imodel = context.getHandler('imodel')
    history = context.getHandler('history')
    output = 'plot_training_sort_%d_init_%d_imodel_%d.pdf'%(sort,init,imodel)
    fig, ax = plt.subplots(2,1, figsize=(10,15))
    fig.suptitle(r'Monitoring Train Plot - Sort = %d, Init = %d, Imodel = %d'%(sort,init,imodel), fontsize=15)
    
    best_epoch = history['max_sp_best_epoch_val'][-1] - start_epoch
    # Make the plot here
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].plot(history['loss'][start_epoch::], c='b', label='Train Step')
    ax[0].plot(history['val_loss'][start_epoch::], c='r', label='Validation Step') 
    ax[0].axvline(x=best_epoch, c='k', label='Best epoch')
    ax[0].legend()
    ax[0].grid()
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('SP')
    ax[1].plot(history['max_sp_val'][start_epoch::], c='r', label='Validation Step') 
    ax[1].axvline(x=best_epoch, c='k', label='Best epoch')
    ax[1].legend()
    ax[1].grid()
    plt.savefig(output)
    plt.close(fig)
        

