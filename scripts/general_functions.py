#######
# PMIpred was developed by Niek van Hilten, Nino Verwei, Jeroen Methorst, and Andrius Bernatavicius at Leiden University, The Netherlands (29 March 2023)
#
# When using this code, please cite:
# Van Hilten, N.; Verwei, N.; Methorst, J.; Nase, C.; Bernatavicius, A.; Risselada, H.J., Bioinformatics, 2024, 40(2) DOI: 0.1093/bioinformatics/btae069 
#######


import keras as k
import pickle


def load_model(name_model, name_tokenizer):
    model = k.models.load_model(name_model)
    with open(name_tokenizer, "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

def predict_ddF(model, tokenizer, seq):
    tokens = tokenizer.texts_to_sequences([seq])
    X = k.preprocessing.sequence.pad_sequences(tokens, maxlen=24, padding="post")
    y = model.predict(X, verbose=0)
    return float(y[0][0])

def length_correction(seq, ddF):
    a = -1.03
    b = 3.28
    ddF_L24 = ( (a*24+b) / (a*len(seq)+b) ) * ddF 
    return ddF_L24

def charge_correction(seq, ddF):
    c_z = -0.93
    z = seq.count("R")+seq.count("K")-seq.count("D")-seq.count("E")
    ddF_z = ddF + c_z*z
    return ddF_z
