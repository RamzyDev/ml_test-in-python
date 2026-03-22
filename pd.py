import pandas as pd                  
import matplotlib.pyplot as plt     

df = pd.read_csv('patients.csv')

print("Aperçu du fichier CSV :")
print(df.head(), '\n')

print("Infos générales :")
print(df.info(), '\n')

print("Statistiques descriptives :")
print(df.describe(), '\n')

print("Patients atteints de diabète :")
print(df[df['diagnostic'] == 'Diabète'], '\n')

moy_age_femmes = df[df['sexe'] == 'F']['age'].mean()
print(f"Âge moyen des femmes : {moy_age_femmes:.2f} ans\n")

#histogramme 
plt.figure(figsize=(8, 4))                           
df['age'].hist(bins=10, color='skyblue', edgecolor='black')  # Histogramme des âges
plt.title("Distribution des âges des patients")      
plt.xlabel("Âge")                                     
plt.ylabel("Nombre de patients")                      
plt.grid(True)                                        
plt.tight_layout()                                    
plt.show()                                            

