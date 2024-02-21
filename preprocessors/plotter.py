import matplotlib.pyplot as plt


def scatter_plot(x, y, y_pred, options: dict):
    plt.scatter(x, y, color=options['sc_color'])
    plt.plot(x, y_pred, color=options['plt_color'])
    plt.title(options['title'])
    plt.xlabel(options['x_label'])
    plt.ylabel(options['y_label'])
    plt.show()
