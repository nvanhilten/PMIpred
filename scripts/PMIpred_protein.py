#######
# PMIpred was developed by Niek van Hilten, Nino Verwei, Jeroen Methorst, and Andrius Bernatavicius at Leiden University, The Netherlands (29 March 2023)
#
# This script is an offline version of the protein module in PMIpred at https://pmipred.fkt.physik.tu-dortmund.de/curvature-sensing-protein/
#
# When using this code, please cite:
# Van Hilten, N.; Verwei, N.; Methorst, J.; Nase, C.; Bernatavicius, A.; Risselada, H.J., Bioinformatics, 2024, 40(2) DOI: 0.1093/bioinformatics/btae069 
#######

import argparse
import os
import sys
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import general_functions as gf
import numpy as np
from math import isnan

flags = argparse.ArgumentParser()
flags.add_argument("-p", "--pdb", help="Input PDB-file", type=str, required=True)
flags.add_argument("-c", "--charge", help="Negative target membrane; apply charge correction.", action="store_true")
flags.add_argument("-w", "--window", help="Sliding window size.", type=int, default=15)
flags.add_argument("-s", "--sasa_threshold", help="SASA threshold.", type=float, default=0.8)
flags.add_argument("-o", "--output", help="Output directory.", type=str)
args = flags.parse_args()

sensing_regime = [-6.4, -10.0] # lower and upper bound of sensing regime (kJ/mol)

def check_filename(name):
    if "." in name and name.split(".")[-1].lower() == "pdb":
        return True
    else:
        return False

def create_outdir(out, name):
    path, filename = os.path.split(name)

    if not out:
        outdir = os.path.join(os.getcwd(), "PMIpred_protein_"+filename.split(".pdb")[0])
    else:
        outdir = os.path.join(out, "PMIpred_protein_"+filename.split(".pdb")[0])
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # save input.pdb
    outfile = open(os.path.join(outdir, "input.pdb"), "w")
    for line in open(name, "r"):
        outfile.write(line)
    outfile.close()

    return outdir, filename

def process_pdb(outdir, name):
    def clean_pdb(outdir, name): # removes non-protein molecules from pdb file
        error = False
        structure = ""
        file_path = os.path.join(outdir, name)
        clean_path = os.path.join(outdir, "clean.pdb")
        protein_dict = {} # {id: AA}
        chain_dict = {}
        chain_list = []
        chain_id = ""
        error_message = ""
        prev_res_nr = ""
        with open(file_path, "rt") as f:
            with open(clean_path, "wt") as b:
                try:
                    f.readline()
                except:
                    error = True
                    error_message = "ERROR: PDB is not readable."
                    return 0, 0, error, error_message, 0
                else:
                    for line in f:
                        if line.startswith("MODEL"):
                            b.write(line)
                        elif line.startswith("ATOM"):
                            b.write(line)
                            if prev_res_nr != int(line[22:26]) and int(line[22:26])> 0: 
                                if chain_id != str(line[21:22]) and chain_id != "":
                                    chain_list.append(chain_id)
                                    chain_dict[chain_id] = [protein_dict]
                                    protein_dict = {}
                                AA = str(line[17:20])
                                res_nr = int(line[22:26])
                                chain_id = str(line[21:22])
                                if chain_id == " ":
                                    error = True
                                    error_message = "ERROR: Your PDB does not contain a chain_id."
                                    break
                                try:
                                    protein_dict[res_nr] = d3to1[AA]
                                except KeyError:
                                    error = True
                                    error_message = f"ERROR: Invalid residue at number {res_nr} of model {chain_id}. Only natural amino acids are allowed."
                                    break
                                else:
                                    prev_res_nr = res_nr                  
                        elif line.startswith("END"):
                            b.write("\nENDMDL")
                            break
        chain_list.append(chain_id)
        chain_dict[chain_id] = [protein_dict]

        parser = PDBParser(QUIET=False)
        try:
            structure = parser.get_structure("struct", clean_path)

        except:
            error = True
            error_message ="ERROR: Invalid PDB."

        return structure, chain_dict, error, error_message, chain_list
    
    def fix_gaps(chain_dict, chain_list):
        for chain in chain_list:
            protein_dict = chain_dict[chain][0]
            seq = ""
            for n in range(min(protein_dict.keys()), max(protein_dict.keys())+1):
                if n in protein_dict:
                    seq += protein_dict[n]
                elif n not in protein_dict:
                    protein_dict[n] = "."
                    seq += "."
            chain_dict[chain].append(seq)
        return chain_dict

    def calc_sasa(structure, chain_dict, chain_list):
        for chain in chain_list:
            protein_dict = chain_dict[chain][0]
            sasa = {}
            sr = ShrakeRupley() # cite Shrake, Rupley (1973). J Mol Biol.
            sr.compute(structure, level="R")
            for i in protein_dict:
                try:
                    sasa[i] = round(structure[0][chain][i].sasa / 100, 3)
                except KeyError:
                    pass
            chain_dict[chain].append(sasa)
        return chain_dict

    # amino acids
    d3to1 = {"ALA":"A", "CYS":"C", "ASP":"D", "GLU":"E", "PHE":"F", "GLY":"G", "HIS":"H", "ILE":"I", "LYS":"K", "LEU":"L", "MET":"M", "ASN":"N", "PRO":"P", "GLN":"Q", "ARG":"R", "SER":"S", "THR":"T", "VAL":"V", "TRP":"W", "TYR":"Y"}

    try:
        structure, chain_dict, error, error_message, chain_list = clean_pdb(outdir, name)
    except:
        error = True
        error_message ="ERROR: Invalid PDB."
    
    if error:
        return 0, 0, error, error_message

    chain_dict = fix_gaps(chain_dict, chain_list)

    chain_dict = calc_sasa(structure, chain_dict, chain_list)

    return chain_dict, chain_list, error, error_message 

def screening(outdir, charge, window, chain_dict, chain_list, sasa_threshold, model, tokenizer, filename):
    def segmentize(chain_dict, chain_list, window):
        for chain in chain_list:
            protein_dict = chain_dict[chain][0]
            seq = chain_dict[chain][1]
            segments = {}
            for i, n in enumerate(range(min(protein_dict.keys()), max(protein_dict.keys())+1)):
                segment = seq[i:i+window]
                if len(segment) == window and not "." in segment:
                    if not segment in segments:
                        segments.update({segment:[n]})
                    else:
                        segments[segment].append(n)
            chain_dict[chain].append(segments)
        return chain_dict

    def ddF_per_seg(chain_dict, chain_list, charge, model, tokenizer):
        for chain in chain_list:
            segments = chain_dict[chain][3]
            sequences = []
            n_lists = []
            ddFs = {}
            if len(segments) == 0:
                chain_dict[chain].append(ddFs)
            else:
                for i, s in enumerate(segments):
                    sequences.append(s)
                    n_lists.append(segments[s])
                    ddF = gf.predict_ddF(model, tokenizer, s)
                    ddF_L24 = gf.length_correction(s, ddF)
                    if charge:
                        ddF_choice = gf.charge_correction(s, ddF_L24)
                    else:
                        ddF_choice = ddF_L24
                    for n in n_lists[i]:
                        ddFs.update({n:[s, ddF, ddF_choice]})
                chain_dict[chain].append(ddFs)
        return chain_dict

    def ddF_per_res(chain_dict, chain_list):
        for chain in chain_list:
            scores_per_res = {}
            avg_score_per_res = {}
            scores = chain_dict[chain][4]
            if len(scores) == 0:
                chain_dict[chain].append(avg_score_per_res)
            else:    
                for n in scores:
                    seq = scores[n][0]
                    score_adj = scores[n][2]
                    for i in range(len(seq)):
                        if not n+i in scores_per_res:
                            scores_per_res[n+i] = [score_adj]
                        else:
                            scores_per_res[n+i].append(score_adj)
                
                
                for n in scores_per_res:
                    avg_score = np.mean(scores_per_res[n])
                    avg_score_per_res.update({n:avg_score})
                chain_dict[chain].append(avg_score_per_res)
        return chain_dict

    def write_per_seg(chain_dict, chain_list, outdir, lower, upper, charge):
        with open(os.path.join(outdir, "segments.txt"), "w") as seg_txt:
            if charge: 
                seg_txt.write("\t".join(["chain", "#n", "sequence", "ΔΔF_adj", "-/S/B"]) + "\n")
            else:
                seg_txt.write("\t".join(["chain", "#n", "sequence", "ΔΔF_L24", "-/S/B"]) + "\n")
            segments_data = []
            for chain in chain_list:
                seg_scores = chain_dict[chain][4]
                for n in seg_scores:
                    seq = seg_scores[n][0]
                    score = seg_scores[n][2] # score_adj (depending on charge choice)
                    if score > lower: # Non-binder
                        c = "-"
                    elif score < upper: # Binder
                        c = "B"
                    else: # Sensor
                        c = "S"
                    seg_txt.write("\t".join([chain, str(n), seq, "{:6.2f}".format(score), c]) + "\n")
                    segments_data.append([chain, n, seq, "{:6.2f}".format(score), c])
        return segments_data

    def write_output(chain_dict, chain_list, lower, upper, outdir, sasa_threshold, window, charge):
        def fill_gaps(d):
            new_d = {}
            for n in range(min(d.keys()), max(d.keys())+1):
                if n in d:
                    new_d[n] = d[n]
                else:
                    new_d[n] = np.nan
            return new_d

        def avg_sasa_section(scores, sasa, l):
            avg_sasa = {}
            for n in scores:
                if isnan(scores[n]) == True:
                    avg_sasa[n] = np.nan
                else:
                    stretch = []
                    for m in sasa:
                        if n-(l-1)/2 <= m <= n+(l-1)/2:
                            stretch.append(sasa[m])
                        else:
                            stretch.append(np.nan)
                    stretch_nonan = [x for x in stretch if isnan(x) == False]
                    avg_sasa[n] = np.mean(stretch_nonan)
            return avg_sasa

        def fill_end(seq, dict_to_fill):
            max_ind = int(max(dict_to_fill.keys()))
            diff = abs((len(seq)-1) - (max_ind - int(min(dict_to_fill.keys()))))
            if diff != 0:
                for i in range(diff):
                    dict_to_fill[max_ind + i+1] = np.nan
            return(dict_to_fill)
        
        def fill_begin(sasa, dict_to_fill):
            min_key_sasa = int(min(sasa.keys()))
            min_key_dict_to_fill = int(min(dict_to_fill.keys()))
            diff = (min_key_dict_to_fill - min_key_sasa)

            if diff > 0:
                for i in range(diff):
                    dict_to_fill[i+1] = np.nan
            return(dict_to_fill)

        for chain in chain_list:

            scores = chain_dict[chain][5]
            sasa = chain_dict[chain][2]
            seq = chain_dict[chain][1]

            l = 9 # length of sasa section

            sasa = fill_gaps(sasa)
            if len(scores) == 0:
                for i in sasa.keys():
                    scores[i] = np.nan
            else:
                scores = fill_gaps(scores)
            avg_sasa = avg_sasa_section(scores, sasa, l) # calculates averaged sasa for n-(l-1)/2 to n+(l-1)/2
          
            scores = fill_begin(sasa, scores)
            avg_sasa = fill_begin(sasa, avg_sasa)

            sasa  = fill_end(seq, sasa)
            scores  = fill_end(seq, scores)
            avg_sasa  = fill_end(seq, avg_sasa)

            # ASSIGN B-FACTORS
            Bfactors = {}
            classification = {}
            accessibility = {}
            class_if_accessible = {} # classifiaction if accessible

            for n in scores:
                if isnan(scores[n]) == True: # gap
                    Bfactor = 0.0
                    c = "-"
                elif scores[n] > lower: # non-binder
                    Bfactor = 0.0
                    c = "-"
                elif scores[n] < upper: # binder
                    if avg_sasa[n] > sasa_threshold:
                        Bfactor = 1.0
                    c = "B"
                else: # sensor
                    if avg_sasa[n] > sasa_threshold:
                        Bfactor = 0.5
                    c = "S"
                if avg_sasa[n] > sasa_threshold:
                    a = "A"
                    c_if_a = c
                else:
                    a = "."
                    c_if_a = "-"
                    Bfactor = 0.0
                Bfactors[n] = Bfactor
                classification[n] = c
                accessibility[n] = a
                class_if_accessible[n] = c_if_a
            chain_dict[chain][4] = scores
            chain_dict[chain][2] = sasa
            chain_dict[chain].append(Bfactors)
            chain_dict[chain].append(classification)
            chain_dict[chain].append(accessibility)
            chain_dict[chain].append(class_if_accessible)
            chain_dict[chain].append(avg_sasa)


        # WRITE OUTPUT PDB-FILE
        in_pdb  = os.path.join(outdir, "clean.pdb")
        with open(os.path.join(outdir, "output.pdb"), "w") as out_pdb:
            with open (in_pdb, "r") as c:
                for line in c:
                    if line and line.startswith("ATOM"):
                        line = line.strip()
                        n = int(line[22:26].strip()) # res nr
                        Bfactors = chain_dict[str(line[21:22])][6]
                        if n in Bfactors:
                            Bfactor = Bfactors[n]
                        else:
                            Bfactor = 0.0
                        out_pdb.write(line[:60] + "{:6.2f}".format(Bfactor) + line[66:] + "\n")
                    else:
                        out_pdb.write(line)

        # WRITE OUTPUT HORIZONTAL ALIGNMENT
        output_data = []
        with open (os.path.join(outdir, "output.txt"), "w") as out_txt:
            chainlist = []
            n_list = [] # residue numbers
            s_list = [] # sequence
            c_list = [] # classification
            a_list = [] # accessibility
            c_if_a_list = [] # class if accessible
            for chain in chain_list:
                sasa = chain_dict[chain][2]
                seq = chain_dict[chain][1]
                classification = chain_dict[chain][7]
                accessibility = chain_dict[chain][8]
                class_if_accessible = chain_dict[chain][9]
                for i, n in enumerate(sasa):
                    if n % 10 == 0 or n == min(sasa.keys()):
                        n_list.append(str(n))
                        len_prev = len(str(n))
                    elif len_prev == 1:
                        n_list.append(".")
                    else:
                        len_prev -= 1
                    chainlist.append(chain)
                    s_list.append(seq[i])
                    c_list.append(classification[n])
                    a_list.append(accessibility[n])
                    c_if_a_list.append(class_if_accessible[n])
                length_1 = len("".join(n_list))
                length_2 = len("".join(s_list))
                diff = abs(length_1 - length_2)
                n_list.append(" "*(5 - diff))
                chainlist.append("     ")
                s_list.append("     ")
                c_list.append("     ")
                a_list.append("     ")
                c_if_a_list.append("     ")
            out_txt.write("".join(chainlist) + "\n"+"".join(n_list) + "\n" + "".join(s_list) + "\n" + "".join(c_list) + "\n" + "".join(a_list) + "\n" + "".join(c_if_a_list))
            output_data.append(["".join(chainlist), "Chain"])
            output_data.append(["".join(n_list), "#n"])
            output_data.append(["".join(s_list), "Sequence"])
            output_data.append(["".join(c_if_a_list),"Classification (if accessible)"])

        # WRITE OUTPUT PER RESIDUE
        residues_data = []
        with open(os.path.join(outdir, "residues.txt"), "w") as res_txt:
            res_txt.write("\t".join(["chain", "#n", "AA", "SASA", "ΔΔF", "-/S/B", "access.", "class_if_accessible"]) + "\n")

            for chain in chain_list:
                sasa = chain_dict[chain][2]
                avg_sasa = chain_dict[chain][10]
                seq = chain_dict[chain][1]
                scores = chain_dict[chain][4]
                classification = chain_dict[chain][7]
                accessibility = chain_dict[chain][8]
                class_if_accessible = chain_dict[chain][9]
                    
                for n, i in enumerate(sasa.keys()):
                    AA = seq[n]
                    sasa_n = avg_sasa[i]
                    score = scores[i]
                    c = classification[i]
                    a = accessibility[i]
                    c_if_a = class_if_accessible[i]
                    res_txt.write("\t".join([chain, str(i), AA, "{:6.2f}".format(sasa_n), "{:6.2f}".format(score), c, a, c_if_a]) + "\n")
                    residues_data.append([chain, str(i), AA, "{:6.2f}".format(sasa_n), "{:6.2f}".format(score), c, a, c_if_a])

        if charge:
            membrane = "Negatively charged membrane (e.g. POPC/POPG)"
        else:
            membrane = "Neutral membrane (e.g. POPC)"

        # WRITE SETTINGS
        with open(os.path.join(outdir, "settings.txt"), "w") as set_txt:
            set_txt.write(f"File name: {filename}\nWindow size: {window}\nSASA Threshold: {sasa_threshold}\nMembrane: {membrane}")

        return residues_data, output_data


    chain_dict = segmentize(chain_dict, chain_list, window) # {seq: [start_n, ...]}
    chain_dict = ddF_per_seg(chain_dict, chain_list, charge, model, tokenizer) # {start_n: [seq, score, score_adj]}
    segments_data = write_per_seg(chain_dict, chain_list, outdir, sensing_regime[0], sensing_regime[1], charge)

    chain_dict = ddF_per_res(chain_dict, chain_list) # {n: avg_score_per_res}
    print("ΔΔF prediction completed...")
    residues_data, output_data = write_output(chain_dict, chain_list, sensing_regime[0], sensing_regime[1], outdir, sasa_threshold, window, charge)

    return segments_data, residues_data, output_data



# process input
status = check_filename(args.pdb)
if not status:
    print("ERROR: Invalid PDB format.")
    sys.exit()

outdir, filename = create_outdir(args.output, args.pdb)
print("\n"+filename+" loaded...")

chain_dict, chain_list, error, error_message = process_pdb(outdir, args.pdb)
if error:
    print(error_message)
    sys.exit()
print("SASA calculated...")

len_list =[]
for chain in chain_list:
    len_list.append(len(chain_dict[chain][1]))
len_list.sort()
if len_list[-1] < args.window:
    print("ERROR: Protein chain length is smaller than window size.")
    sys.exit()

# screening
model, tokenizer = gf.load_model("./final_model", "./final_model/tokenizer.pickle")
print("Transformer model initiated...")
segments_data, residues_data, output_data = screening(outdir, args.charge, args.window, chain_dict, chain_list, args.sasa_threshold, model, tokenizer, filename)   
print("\nOutput written to:\t"+outdir)
