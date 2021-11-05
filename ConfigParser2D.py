import configparser
from numpy import sqrt

def ImportConfig(FileName):
    cfg = configparser.ConfigParser()
    cfg.read(FileName)

    #The settings dictionary reads from the .ini file (cfg)
    setd = {
      "name": cfg['settings']['name'],
      "timestamp": cfg.getboolean('settings','timestamp'),
      "symbolic": cfg.getboolean('settings','symbolic'),
      "plots": cfg.getboolean('settings','plots'),
      "log": cfg.getboolean('settings','log'),
      "simp": cfg.getboolean('settings','simplification')
    }

    #prints the settings dictionary to the user
    print('\nValues printed are in SI units unless specified otherwise.\n')
    print(f'Configuration for {setd["name"]}:\n')

    print(f'\t ---Settings---\nName\t\t=\t{setd["name"]}\nTimestamp\t=\t{setd["timestamp"]}\nSymbolic\t=\t{setd["symbolic"]}\nPlot\t\t=\t{setd["plots"]}\nLog\t\t=\t{setd["log"]}\nSimplify\t=\t{setd["simp"]}\n')

    #The QM values dictionary
    qmv = {
      "a0": complex(cfg['QMvalues']['a0']),
      "a1": complex(cfg['QMvalues']['a1']),
      "a2": complex(cfg['QMvalues']['a2']),
      "a3": complex(cfg['QMvalues']['a3']),
      "A": float(cfg['QMvalues']['A']),
      "B": float(cfg['QMvalues']['B']),
      "h": float(cfg['QMvalues']['h'])
    }

    #Since A^2 + B^2 = 1 must be true:
    #tempA = qmv["A"]
    #tempB = qmv["B"]
    #qmv["A"] = qmv["A"]/sqrt(tempA**2 + tempB**2)
    #qmv["B"] = qmv["B"]/sqrt(tempA**2 + tempB**2)

    print(f'\t ---QM Values---\na0\t\t=\t{qmv["a0"]}\na1\t\t=\t{qmv["a1"]}\na2\t\t=\t{qmv["a2"]}\na3\t\t=\t{qmv["a3"]}\nA\t\t=\t{qmv["A"]}\nB\t\t=\t{qmv["B"]}\nhbar\t\t=\t{qmv["h"]}\n')

    #The statistical mechanics values dictionary
    smv = {
      "N": float(cfg['SMvalues']['N']),
      "V": float(cfg['SMvalues']['V']),
      "T": float(cfg['SMvalues']['T']),
      "kB": float(cfg['SMvalues']['kB']),
      "r": float(cfg['SMvalues']['r'])
    }

    print(f'\t ---SM Values---\nNumber\t\t=\t{smv["N"]}\nVolume\t\t=\t{smv["V"]}\nTemperature (K)\t=\t{smv["T"]}\nBoltzmann Const.=\t{smv["kB"]}\nNumber Density=\t{smv["r"]}')

    #The plot values dictionary
    ptv = {
      "Nmin": float(cfg['plotvalues']['Nmin']),
      "Nmax": float(cfg['plotvalues']['Nmax']),
      "Tmin": float(cfg['plotvalues']['Tmin']),
      "Tmax": float(cfg['plotvalues']['Tmax']),
      "a0min": float(cfg['plotvalues']['a0min']),
      "a0max": float(cfg['plotvalues']['a0max']),
      "a3": float(cfg['plotvalues']['a3plot']),
      "tp": int(cfg['plotvalues']['tpoints']),
      "time": float(cfg['plotvalues']['time'])
    }

    print(f'\t ---Plot Settings---\nTime Points\t=\t{ptv["tp"]}\nTime Length\t=\t{ptv["time"]}\nMax Number\t=\t{ptv["Nmax"]}\nMin Number\t=\t{ptv["Nmin"]}\nMax Temp. (K)\t=\t{ptv["Tmax"]}\nMin Temp. (K)\t=\t{ptv["Tmin"]}\nMax a0\t\t=\t{ptv["a0max"]}\nMin a0\t\t=\t{ptv["a0min"]}\na3\t\t=\t{ptv["a3"]}\n')
    
    return setd,qmv,smv,ptv