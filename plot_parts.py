import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_loss_and_accuracy(train_loss,
                           train_acc,
                           val_loss,
                           val_acc,
                           filename=None):
    """
    Plot the train and validation loss and accuracy over the number of epochs, and save the graphs to file.
    """
    # create figure and subplots
    fig, axs = plt.subplots(2, figsize=(8, 8))

    # plot train and validation loss over number of epochs
    axs[0].plot(train_loss, label='train loss')
    axs[0].plot(val_loss, label='validation loss')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    axs[0].legend()

    # plot train and validation accuracy over number of epochs
    axs[1].plot(train_acc, label='train accuracy')
    axs[1].plot(val_acc, label='validation accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('accuracy (%)')
    axs[1].legend()

    if filename is not None:
        plt.savefig(filename + '.png')
    plt.show()

def plot_images_batch(data, label, dnn_output, estimated_viterbi=None,estimated_SFD=None, save_path=None):
    batch_size, _, Nr, Nd = data.size()
    data = data.cpu().detach().numpy()
    dnn_output = dnn_output.view(batch_size, 1, Nr, Nd).cpu().detach().numpy()
    images = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Create colorbars
    cbar1 = None
    cbar2 = None

    acc_vit = 0
    acc_SFD = 0

    for i in range(batch_size):
        axes[0].cla()  # Clear the previous plot
        axes[1].cla()  # Clear the previous plot

        # Plot the data
        im1 = axes[0].imshow(data[i, 0, :, :],cmap='hot')
        axes[0].set_title('Radar Observation')
        r = label[i][0][0].item()
        d = label[i][1][0].item()
        axes[0].plot(d, r, 'co', markerfacecolor='none', markeredgewidth=2)  # Mark the pixel represented by the target label

        # Plot the DNN's output
        im2 = axes[1].imshow(dnn_output[i, 0, :, :],cmap='hot')
        axes[1].set_title("Log Likelihood")
        r_estimated_vit = estimated_viterbi[i][0]
        d_estimated_vit = estimated_viterbi[i][1]
        r_estimated_SFD = estimated_SFD[i][0]
        d_estimated_SFD = estimated_SFD[i][1]



        # Mark the target and estimated pixels on the output
        #axes[1].plot(d, r, 'cx', markerfacecolor='none', markeredgewidth=2, label='Target')
        if (r,d) == estimated_viterbi[i]:
            acc_vit += 1
            vit_label = 'Viterbi (' + str(acc_vit) + '/' + str(batch_size) + ')'
            axes[1].plot(d_estimated_vit, r_estimated_vit, 'co', markerfacecolor='none', markeredgewidth=2, label=vit_label)
        else:
            vit_label = 'Viterbi (' + str(acc_vit) + '/' + str(batch_size) + ')'
            axes[1].plot(d_estimated_vit, r_estimated_vit, 'bo', markerfacecolor='none',
                         markeredgewidth=2, label=vit_label)

        if (r,d) == estimated_SFD[i]:
            acc_SFD += 1
            SFD_label = 'SFD (' + str(acc_SFD) + '/' + str(batch_size) + ')'
            axes[1].plot(d_estimated_SFD, r_estimated_SFD, 'cD', markerfacecolor='none', markeredgewidth=2, label=SFD_label)
        else:
            SFD_label = 'SFD (' + str(acc_SFD) + '/' + str(batch_size) + ')'
            axes[1].plot(d_estimated_SFD, r_estimated_SFD, 'bD', markerfacecolor='none',
                         markeredgewidth=2, label=SFD_label)
        axes[1].legend()

        # Update colorbar values
        if cbar1 is None:
            cbar1 = fig.colorbar(im1, ax=axes[0])
        else:
            cbar1.update_normal(im1)

        if cbar2 is None:
            cbar2 = fig.colorbar(im2, ax=axes[1])
        else:
            cbar2.update_normal(im2)

        plt.pause(1)

        # Save the current plot as an image
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        images.append(image)

    plt.close(fig)

    # Convert the sequence of images to a GIF
    if save_path:
        images = [Image.fromarray(image) for image in images]
        images[0].save(save_path, save_all=True, append_images=images[1:], loop=0, duration=750)






def plot_graphs(num_epochs, y_list,titles, legends, ylabels, xlabel="Epochs", grid=True, filename=None):
    epochs = [i+1 for i in range(num_epochs)]
    num_plots = len(y_list)
    fig, axs = plt.subplots(1, num_plots, figsize=(12,4))
    for i in range(num_plots):
        axs[i].set_title(titles[i])
        for j in range(len(y_list[i])):
            axs[i].plot(epochs, y_list[i][j], label=legends[j])
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabels[i])
        axs[i].grid(grid)
        axs[i].legend(loc="best")
    if filename is not None:
            plt.savefig(filename + '.png')
    plt.show()



