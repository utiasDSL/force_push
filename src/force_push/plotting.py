def rcparams(fontsize=6):
    return {
        "pgf.texsystem": "pdflatex",
        "font.size": fontsize,
        "font.family": "serif",
        "font.sans-serif": "DejaVu Sans",
        "font.weight": "normal",
        "text.usetex": True,
        "legend.fontsize": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "figure.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[utf8]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{siunitx}",
                r"\usepackage{bm}",
            ]
        ),
    }
