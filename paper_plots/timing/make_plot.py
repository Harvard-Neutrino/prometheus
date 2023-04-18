import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
plt.style.use(os.path.abspath("../paper.mplstyle"))

MAXNREADLINES = 3

def initialize_args():
    import argparse
    parser =  argparse.ArgumentParser()
    # Meta
    parser.add_argument(
        "--timing_dir",
        dest="timing_dir",
        type=str,
        default="./muon/",
        help="Where to look for the timing summary files"
    )
    parser.add_argument(
        "--figfile",
        type=str,
        default="",
        help="Where to save the output figure"
    )
    parser.add_argument(
        "--color",
        type=str,
        default="Wistia"
    )
    return parser.parse_args()

def parse_line(line, outarr):
    splitline = line.split()
    if len(splitline)!=6:
        return
    elif 'ppc_photon_propagator.py' in splitline[-1] and "propagate" in splitline[-1]:
        idx = 0
    elif 'prometheus.py' in splitline[-1] and "inject" in splitline[-1]:
        idx = 1
    elif 'prometheus.py' in splitline[-1] and "construct_output" in splitline[-1]:
        idx = 2
    elif "lepton_propagator.py" in splitline[-1] and "energy_losses" in splitline[-1]:
        idx = 3
    elif "lepton_propagator.py" in splitline[-1] and "getitem" in splitline[-1]:
        idx = 3
    elif "prometheus.py" in splitline[-1] and "sim" in splitline[-1]:
        idx = 4
    else:
        #print(splitline)
        return
    outarr[idx] += float(splitline[3])

def mims(outarr):
    total = outarr[:, 4]
    photon_prop = outarr[:, 0] - outarr[:, 3]
    lepton_prop = outarr[:, 3]
    injection = outarr[:, 1]
    output = outarr[:, 2]
    misc = total - (output + injection + lepton_prop + photon_prop)
    we = np.zeros((outarr.shape[0], 6))
    we[:, 0] = misc
    we[:, 1] = injection
    we[:, 2] = output
    we[:, 3] = lepton_prop
    we[:, 4] = photon_prop
    we[:, 5] = total
    return we

def prepare_output(timing_dir):

    fs = sorted(glob(f"{timing_dir}/*summary*"))
    es = []
    outarr = np.zeros((len(fs), 5))
    for idx, f in enumerate(fs):

        splitname = f.split("/")[-1].split("_")
        while len(splitname) > 0:
            test_e = splitname.pop()
            try:
                e = float(test_e)
                break
            except ValueError:
                pass
        es.append(e)
        with open(f, "r") as infile:
            for line in infile.readlines():
                parse_line(line, outarr[idx, :])

    sorter = np.argsort(es)
    es = np.array(es)[sorter]
    we = mims(outarr)
    we = we[sorter, :]

    return es, we

def make_plot(es, output, figfile, cmapname="Wistia"):

    from matplotlib.cm import get_cmap
    cmap = get_cmap(cmapname)

    labels = [
        "Miscellaneous",
        "Injection",
        "Saving output",
        "Lepton propagation",
        "Photon propagation",
    ]

    for x in range(len(labels)):
        color = cmap(x / (len(labels) - 1))
        #amin = 0.2
        #alpha = x * (1 - amin) / 5.0 + amin
        label = labels[x]

        if x == 0:
            y1 = np.zeros(len(output[:, x]))
        else:
            y1 = np.sum(output[:, :x], axis=1)
        y2 = np.sum(output[:, :x+1], axis=1)

        plt.fill_between(es, y1, y2, label=label)
        #plt.fill_between(es, y1, y2, label=label, color=color)
        plt.plot(es, y2, color="k", lw=1)

    #plt.semilogx()
    plt.loglog()
    plt.ylim(1, 5e4)
    plt.legend(
        ncol=3,
        fontsize=10
    )
    plt.xlim(1e2, 1e6)
    plt.xlabel(r"$E_{\nu}~\left[\rm{GeV}\right]$")
    plt.ylabel(r"Execution time [s]")
    plt.savefig(figfile)
    plt.close()

def main(timing_dir, figfile, cmapname="Wistia"):
    es, output = prepare_output(timing_dir)
    make_plot(es, output, figfile, cmapname=cmapname)

if __name__=="__main__":
    args = initialize_args()
    if args.figfile:
        figfile = args.figfile
    else:
        figfile = f"{args.timing_dir}/timing.pdf"
    main(args.timing_dir, figfile, cmapname=args.color)
