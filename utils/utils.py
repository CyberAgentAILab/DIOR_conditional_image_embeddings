import colorsys

import webcolors


# Convert color values to color names
def hsv_to_color_name(row, hue_column="color_hue", sat_column="color_sat", val_column="color_val"):
    # colorsys expects colors in the range [0, 1]
    rgb_hsv = colorsys.hsv_to_rgb((row[hue_column] + 0.5) % 1, 0.5 * row[sat_column], 0.5 * row[val_column])
    rgb = tuple(int(255 * x) for x in rgb_hsv)

    try:
        color_name = webcolors.rgb_to_name(rgb)
    except ValueError:
        min_colors = {}
        for key, name in webcolors.CSS21_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - rgb[0]) ** 2
            gd = (g_c - rgb[1]) ** 2
            bd = (b_c - rgb[2]) ** 2
            min_colors[(rd + gd + bd)] = name
        color_name = min_colors[min(min_colors.keys())]

    return color_name


def reshape_score(score):
    map_r = round(score["mean_average_precision_at_r"] * 100, 1)
    prec_1 = round(score["precision_at_1"] * 100, 1)
    ami = score["AMI"]
    nmi = score["NMI"]
    return {"map_r": map_r, "prec_1": prec_1, "ami": ami, "nmi": nmi}
