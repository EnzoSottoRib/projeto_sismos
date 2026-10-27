import pandas as pd

n = pd.read_csv("saida/noaa_sem_match.csv")
print(n[(n.mag_noaa >= 5.5) | (n.mortes_max >= 100)][["Year", "Mo", "Dy", "local", "mag_noaa", "mortes_max"]])

p = pd.read_csv("saida/countries.csv")
print(p.tem_evento.value_counts())