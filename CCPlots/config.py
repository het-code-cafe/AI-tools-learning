import matplotlib.colors as mcolors

# Where to store the plots
OUTPUT_PATH = "../../plots/"

# matplotlib theme
THEME = "ocean"

# Or custom colours
COLOR_PALETTE = {
    "base_colors": {
        "dark_green": "#113428",
        "medium_green": "#459578",
        "bright_teal": "#37EDAB",
        "bright_yellow": "#FED53C"
    },
    "complementary_colors": {
        "deep_burgundy": "#341128",
        "warm_brown": "#8A5722",
        "rusty_red": "#7A2D2A",
        "soft_coral": "#D9746E"
    },
    "analogous_colors": {
        "deep_teal": "#165544",
        "soft_green": "#5EBA93",
        "light_yellow": "#FFF066",
        "golden_yellow": "#D9B23F"
    },
    "neutral_colors": {
        "white": "#FFFFFF",
        "light_gray": "#EFEFEF",
        "medium_gray": "#8C8C8C",
        "dark_gray": "#505050",
        "charcoal": "#333333"
    },
    "accent_colors": {
        "coral_pink": "#FF6F61",
        "periwinkle_blue": "#6C7DFF",
        "mint_green": "#ACD1AF",
        "light_beige": "#FFE4B5"
    }
}

# Define a list of colors for a colour map (white to green)
CMAP_WHITE = mcolors.LinearSegmentedColormap.from_list(name="custom_cmap",
                                                       colors=[COLOR_PALETTE['neutral_colors']['white'], COLOR_PALETTE['base_colors']['medium_green']],
                                                       N=256)

# Colourful custom colour map
CMAP_BRAND = mcolors.LinearSegmentedColormap.from_list(name="custom_cmap",
                                                       colors=[COLOR_PALETTE['base_colors']['bright_yellow'], COLOR_PALETTE['base_colors']['medium_green'], COLOR_PALETTE['base_colors']['bright_teal']],
                                                       N=256)