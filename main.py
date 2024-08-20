"""
main.py

Run all examples from the CCPlots library to make my life a little easier.

Please note that the entire configuration for the plots is in CCPlots (config.py),
including the output folder.
"""
import CCPlots


def all_plots():
    """
    This function will, as you probably suspected, run all the plots.
    This is a slow process and will give you lots of warnings that I don't have time
    to fix, since we just use dummy data as an illustration.
    :return:
    """
    classes = [cls for name, cls in CCPlots.__dict__.items()
               if isinstance(cls, type) and cls.__module__.startswith('CCPlots')]

    # Create or save plots
    for cls in classes:
        instance = cls()
        instance.main()


def plot_scratch():
    """ Scratchpad for whatever needs re-generation. """
#    from CCPlots.LinearRegressionExample import LinearRegressionExample
#    LinearRegressionExample().main()

#    from CCPlots.MultivariateRegressionExample import MultivariateRegressionExample
#    MultivariateRegressionExample().main()

#    from CCPlots.MSEExample import MSEExample
#    from CCPlots.MSEZoomExample import MSEZoomExample
#    MSEExample().main()
#    MSEZoomExample().main()

    from CCPlots.KNearestExample import KNearestExample
    KNearestExample().main()

#    from CCPlots.KMeansExample import KMeansExample
#    KMeansExample().main()

#    from CCPlots.LogisticRegressionExample import LogisticRegressionExample
#    LogisticRegressionExample().main()

#    from CCPlots.NeuralNetworkGrowthExample import NeuralNetworkGrowthExample
#    NeuralNetworkGrowthExample().main()

#    from CCPlots.NeuralNetworkActivationFunctionsExample import NeuralNetworkActivationFunctionsExample
#    NeuralNetworkActivationFunctionsExample().main()

#    from CCPlots.DecisionTreeExample import DecisionTreeExample
#    DecisionTreeExample().main()

#    from CCPlots.OverfittingUnderfittingExample import OverfittingUnderfittingExample
#    OverfittingUnderfittingExample().main()

#    from CCPlots.KFoldExample import KFoldExample
#    KFoldExample().main()


if __name__ == "__main__":
    # This takes a while, comment out at your own risk
    #all_plots()

    # Plot only what is needed
    plot_scratch()
