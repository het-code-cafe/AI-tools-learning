"""
main.py

Run all examples from the CCPlots library to make my life a little easier.
"""
import CCPlots


def all_plots():
    classes = [cls for name, cls in CCPlots.__dict__.items()
               if isinstance(cls, type) and cls.__module__.startswith('CCPlots')]

    # Create or save plots
    for cls in classes:
        instance = cls()
        instance.main()


def plot_scratch():
    """ Scratchpad for whatever needs re-generation I suppose """
    from CCPlots.LinearRegressionExample import LinearRegressionExample
    LinearRegressionExample().main()


if __name__ == "__main__":
    # This takes a while, comment out at your own risk
    #all_plots()

    # Plot only what is needed
    plot_scratch()
