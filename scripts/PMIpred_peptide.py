#######
# PMIpred was developed by Niek van Hilten, Nino Verwei, Jeroen Methorst, and Andrius Bernatavicius at Leiden University, The Netherlands (29 March 2023)
#
# This script is an offline version of the peptide module in PMIpred at https://pmipred.fkt.physik.tu-dortmund.de/curvature-sensing-peptide/
#
# When using this code, please cite:
# Van Hilten, N.; Verwei, N.; Methorst, J.; Nase, C.; Bernatavicius, A.; Risselada, H.J., biorxiv (2023)
#######

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import sys
import general_functions as gf
import numpy as np
import math
import modlamp.plot

flags = argparse.ArgumentParser()
flags.add_argument("-s", "--seq", help="Input amino acid sequence (one-letter abbreviations).", type=str, required=True)
flags.add_argument("-c", "--charge", help="Negative target membrane; apply charge correction.", action="store_true")
flags.add_argument("-o", "--output", help="Output directory.", type=str)
args = flags.parse_args()

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams['savefig.transparent'] = True

def check_sequence(seq):
    seq = seq.strip()
    seq = "".join(seq.split())
    seq = seq.upper()

    ALPHABET = sorted(['A', 'R', 'N', 'D', 'C', 'F', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'P', 'S', 'T', 'W', 'Y', 'V'])
    if len(seq) > 24:
        return False, "Sequence too long (should be <=24 residues)."
    elif len(seq) < 7:
        return False, "Sequence too short (should be >=7 residues)."

    for AA in seq:
        if AA not in ALPHABET:
            return False, "Character \"" + AA + "\" not allowed. Please use single-letter abbreviations for natural amino acids only."

    return True, seq

def calc_descriptors(seq):
    hydro_dict = {'I': [1.80], 'F': [1.79], 'V': [1.22], 'L': [1.70], 'W': [2.25], 'M': [1.23], 'A': [0.31],
                'G': [0.00], 'C': [1.54], 'Y': [0.96], 'P': [0.72], 'T': [0.26], 'S': [-0.04], 'H': [0.13],
                'E': [-0.64], 'N': [-0.60], 'Q': [-0.22], 'D': [-0.77], 'K': [-0.99], 'R': [-1.01]} # Fauchere & Pliska 1983

    z = seq.count("R")+seq.count("K")-seq.count("D")-seq.count("E")
    
    H = np.mean([hydro_dict[AA][0] for AA in seq])
    
    sum_cos, sum_sin = 0.0, 0.0
    angle = 100.0
    for i, AA in enumerate(seq):
        h = hydro_dict[AA][0]
        rad_inc = ((i*angle)*math.pi)/180.0
        sum_cos += h * math.cos(rad_inc)
        sum_sin += h * math.sin(rad_inc)
    uH = math.sqrt(sum_cos**2 + sum_sin**2) / len(seq)

    return z, H, uH

def calc_dF_sm(ddF): # Calculate dF_sm_R50: the membrane-binding free energy to a typical liposome (R=50)
    a = 3.83 # calibration ddF_ vs dF_sm_infty
    b = 12.27 # calibration ddF vs dF_sm_infty
    e = 0.165 # relative strain used in ddF calculation
    R = 50 # typical liposome radius
    dF_sm_R50 = a*ddF+b + (ddF/e)*( (1/R**2) + (2/R) )
    return dF_sm_R50

def calc_Pm(ddF, R):
    N_A = 6.022E23 # avogadro constant in mol-1
    V = 1E24 # volume in nm3 (1 liter)
    A_lip = 0.64 # area per lipid in nm2
    conc = 0.001 # [0.0001, 0.005] # concentrations in M
    Vp = 5*1*1 # peptide volume in nm3
    Ap = 5*1 # peptide area in nm2
    kT = 2.479 # kJ/mol
    A = 1/2 * conc*N_A*A_lip
    Ns = V/Vp
    Nm = A/Ap

    a = 3.83
    b = 12.27
    e = 0.165

    Pm = 1/(1+(Ns/Nm)*np.exp((a*ddF+b+(ddF/e)*( (1/R**2) + (2/R) ))/kT))
    return Pm

def create_outdir(out, seq):
    if not out:
        outdir = os.path.join(os.getcwd(), "PMIpred_peptide_"+seq)
    else:
        outdir = os.path.join(out, "PMIpred_peptide_"+seq)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def plot_helical_wheel(seq, name):
    modlamp.plot.helical_wheel(seq, filename=name)
    plt.savefig(name)

def write_output(output, name):
    outfile = open(name, "w")
    outfile.write(output)
    outfile.close()

def plot_probability(ddF, name, charge):
    ddF_lower = -6.4
    ddF_upper = -10.0
    dF_sm_R50_lower = calc_dF_sm(ddF_lower)
    dF_sm_R50_upper = calc_dF_sm(ddF_upper)

    dF_sm_R50 = calc_dF_sm(ddF)
    Pm_R50 = calc_Pm(ddF, 50)

    # PLOTTING
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))

    # thermo model
    ddF_range = np.arange(-20, 0, 0.01)
    dF_sm_R50_range = []
    Pm_R50_range = []
    for F in ddF_range:
        dF_sm_R50_range.append(calc_dF_sm(F))
        Pm_R50_range.append(calc_Pm(F, 50))

    ax[0].axvline(x=dF_sm_R50_lower, color="black", linewidth=1, alpha=0.5, zorder = 0)
    ax[0].axvline(x=dF_sm_R50_upper, color="black", linewidth=1, alpha=0.5, zorder = 0)
    ax[0].axvspan(max(dF_sm_R50_range), dF_sm_R50_lower, color="purple", alpha=0.25, zorder = 0)
    ax[0].axvspan(dF_sm_R50_upper, dF_sm_R50_lower, color="orange", alpha=0.25, zorder = 0)
    ax[0].axvspan(dF_sm_R50_upper, min(dF_sm_R50_range), color="red", alpha=0.25, zorder = 0)

    if dF_sm_R50_lower >= dF_sm_R50 >= dF_sm_R50_upper: # sensor
        color = "orange"
    if dF_sm_R50 > dF_sm_R50_lower: # non-binder
        color = "purple"
    if dF_sm_R50 < dF_sm_R50_upper: # binder
        color = "red"

    ax[0].plot(dF_sm_R50_range, Pm_R50_range, color="black", linewidth=3, zorder = 1)
    if dF_sm_R50 < min(dF_sm_R50_range):
        ax[0].scatter(min(dF_sm_R50_range), Pm_R50, color=color, s=200, zorder = 2)
    elif dF_sm_R50 > max(dF_sm_R50_range):
        ax[0].scatter(max(dF_sm_R50_range), Pm_R50, color=color, s=200, zorder = 2)
    else:
        ax[0].scatter(dF_sm_R50, Pm_R50, color=color, s=200, zorder = 2)

    t1 = ax[0].text(6.5, 1.1, "Non-binder", fontsize=10, color="purple", zorder=3)
    t1.set_bbox(dict(facecolor="white", alpha=0.5, edgecolor='purple'))
    t2 = ax[0].text(-16.7, 1.1, "Sensor", fontsize=10, color="orange", zorder=3)
    t2.set_bbox(dict(facecolor="white", alpha=0.5, edgecolor='orange'))
    t3 = ax[0].text(-45, 1.1, "Binder", fontsize=10, color="red", zorder=3)
    t3.set_bbox(dict(facecolor="white", alpha=0.5, edgecolor='red'))

    ax[0].set_xlim([math.ceil(min(dF_sm_R50_range)), math.floor(max(dF_sm_R50_range))])
    ax[0].invert_xaxis()
    ax[0].set_xlabel("Curved-membrane-binding free energy $\Delta F_\mathrm{sm}(R=50)$")

    ax[0].set_yticks(np.arange(0, 1.3, 0.1))
    ax[0].set_ylabel("Membrane-binding probability $P_\mathrm{m}$")


    # Pm_vs_R
    ddF_list = [ddF_lower, ddF_upper, ddF]
    R_range = np.arange(0.01, 100.01, 0.01)
    for i, F in enumerate(ddF_list):
        Pm_list = []
        for R in R_range:
            Pm_list.append(calc_Pm(F, R))
        if ddF_lower >= F >= ddF_upper: # sensor
            color = "orange"
            LW = 3
            if F == ddF_lower: # edge
                color = "black"
                Pm_sensor_low = Pm_list
                LW = 1
            if F == ddF_upper: # edge
                color= "black"
                Pm_sensor_high = Pm_list
                LW = 1
        if F > ddF_lower: # non-binder
            color = "purple"
            LW = 3
        if F < ddF_upper: # binder
            color = "red"
            LW = 3

        if i ==2:
            if charge:
                label = "$\Delta \Delta$F_adj = "+str(round(F,3))+ " kJ/mol"
            else:
                label = "$\Delta \Delta$F_L24 = "+str(round(F,3))+ " kJ/mol"
            ax[1].plot(R_range, Pm_list, label=label, color=color, linewidth=LW, zorder = 2)
        else:
            ax[1].plot(R_range, Pm_list, color=color, linewidth=LW, zorder = 2)


    ax[1].fill_between(R_range, 0, Pm_sensor_low, color="purple", alpha=0.25, zorder = 0)
    t1 = plt.text(2, 0.03, "Non-binder", fontsize=10, color="purple", zorder=3)
    t1.set_bbox(dict(facecolor="white", alpha=0.5, edgecolor='purple'))

    ax[1].fill_between(R_range, Pm_sensor_low, Pm_sensor_high, color="orange", alpha=0.25, zorder = 0)
    t2 = plt.text(40, 0.45, "Sensor", fontsize=10, color="orange", zorder=3)
    t2.set_bbox(dict(facecolor="white", alpha=0.5, edgecolor='orange'))

    ax[1].fill_between(R_range, Pm_sensor_high, 1, color="red", alpha=0.25, zorder = 0)
    t3 = plt.text(85, 0.96, "Binder", fontsize=10, color="red", zorder=3)
    t3.set_bbox(dict(facecolor="white", alpha=0.5, edgecolor='red'))
    
    ax[1].set_xlabel("Vesicle radius R (nm)")
    ax[1].set_xlim([0,100])
    ax[1].set_ylabel("Membrane-binding probability $P_\mathrm{m}$")
    ax[1].set_yticks(np.arange(0, 1.1, 0.1))

    handles, labels = ax[1].get_legend_handles_labels()
    legend = fig.legend(loc="upper left", bbox_to_anchor=(0.65,0.97))
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_facecolor('none')

    fig.savefig(name)


# check sequence
status, message = check_sequence(args.seq)
if not status:
    print("ERROR: " + message)
    sys.exit()

# calculate physicochemical descriptors
z, H, uH = calc_descriptors(args.seq)

# load transformer model
model, tokenizer = gf.load_model("./final_model", "./final_model/tokenizer.pickle")

# predict ddF
ddF = gf.predict_ddF(model, tokenizer, args.seq)

# corrections
ddF_L24 = gf.length_correction(args.seq, ddF)
if args.charge:
    ddF_adj = gf.charge_correction(args.seq, ddF_L24)
    dF_sm = calc_dF_sm(ddF_adj)
else:
    dF_sm = calc_dF_sm(ddF_L24)

# create output directory
outdir = create_outdir(args.output, args.seq)

# helical wheel
helical_wheel_filename = os.path.join(outdir, "helical_wheel.pdf")
plot_helical_wheel(args.seq, helical_wheel_filename)

# probability plots
probability_plots_filename = os.path.join(outdir, "probability_plots.pdf")
if args.charge:
    plot_probability(ddF_adj, probability_plots_filename, args.charge)
else:
    plot_probability(ddF_L24, probability_plots_filename, args.charge)

# printing output
output = ""
output += args.seq + "\n\n"
output += "ΔΔF =\t\t\t" + str(round(ddF, 3)) + " kJ/mol\n"
output += "ΔΔF_L24 =\t\t" + str(round(ddF_L24, 3)) + " kJ/mol\n"
if args.charge:
    output += "ΔΔF_adj =\t\t" + str(round(ddF_adj, 3)) + " kJ/mol\n"
output += "----------\n"
if args.charge:
    output += "Negatively charged membrane\nCalculated from ΔΔF_adj:\nΔF_sm(R=50) =\t\t" + str(round(dF_sm, 3)) + " kJ/mol\n"
else:
    output += "Neutral membrane\nCalculated from ΔΔF_L24:\nΔF_sm(R=50) =\t\t" + str(round(dF_sm, 3)) + " kJ/mol\n"
output += "----------\n"
output += "Length =\t\t\t" + str(len(args.seq)) + "\n"
output += "Charge =\t\t\t" + str(z) + "\n"
output += "Hydrophobicity =\t\t" + str(round(H, 3)) + "\n"
output += "Hydrophobic moment =\t" + str(round(uH, 3)) + "\n"
print(output)

output_filename = os.path.join(outdir, "output.txt")
write_output(output, output_filename)

print("\n\nOutput written to:\t\t" + output_filename)
print("Helical wheel diagram plotted:\t" + helical_wheel_filename)
print("Probabilities plotted:\t\t" + probability_plots_filename)




