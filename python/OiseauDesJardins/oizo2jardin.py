# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 15:29:52 2021

@author: romain
"""
import requests
from requests_oauthlib import OAuth1Session
from requests_oauthlib import OAuth1

urlbase = "https://www.oiseauxdesjardins.fr/api/"
logs = "?user_email=biodiguard@gmail.com&user_pw=mautodontha417"



#%%requests


client_secret = "dcac6f9436513a5bd8545833eb87b758"
client_key = "c603d758919eb12ca3c68a052df02453060b0ed10"
oauth = OAuth1Session(client_key,
                      client_secret=client_secret)

url = "https://www.oiseauxdesjardins.fr/api/observations/search?user_email=biodiguard@gmail.com&user_pw=mautodontha417"

parameters = {
        "only_with_picture":"1",
        "period_choice":"range",
        "date_from":"1.1.2000",
        "date_to":"31.12.2010"
        }
authnorm = OAuth1(client_key,
                   client_secret=client_secret)
#response = oauth.post(url,params= parameters)
#la ligne la plus importante
response = requests.post(url = url, auth=authnorm,json=parameters)

print(response.status_code)

re = response.json()
#%%
"""la fonction recense_oiseau ajoute à la comptabilité des oiseaux 
déja recensés (par le dictionnaire recense_oiseau) ceux du mois et de l'annee mis en paramètre
Cette fonction marche par effet de bord"""

#Je fais tourner cette fonction pour chaque mois entre 2010 et 2021 
# car sinon l'API a une limite de d'observations (4000) et donc si on fait
#une requete trop large en date elle ne renvoie que les 4000 premiers résultats
liste_mois = list(range(1,13))
liste_annee = list(range(2010,2012))#dernière année + 1
for i in range(len(liste_mois)):
    liste_mois[i] = str(liste_mois[i])
for i in range(len(liste_annee)):
    liste_annee[i] = str(liste_annee[i])



def recense_oiseaux(mois,annee): 
    
    parameters['date_from'] = '1.' + mois + '.' + annee
    parameters['date_to'] = '31.' + mois + '.' + annee
    
    url = "https://www.oiseauxdesjardins.fr/api/observations/search?user_email=biodiguard@gmail.com&user_pw=mautodontha417"
    
    response = requests.post(url = url, auth=authnorm,json=parameters)
    
    re = response.json()
    if 'forms' not in re['data'].keys():
        pass
    else:
        recense_mois = re['data']['forms']
        
        for x in recense_mois:
            observation = x['sightings']
        #print(len(observation))
        
            for j in range(len(observation)):
                try:
                    medias = observation[j]['observers'][0]['medias']
                    
                    for media in medias:
                        compte_espece[observation[j]['species']['name']] += 1
                        web_path = media['path'] + '/' + media['filename']
                except:
                    pass
#%% get list species
species = requests.get(urlbase + "species"+logs +"&is_used=1",auth = authnorm)
species = species.json()
#print(species)
compte_espece = {}
L = []

for info_esp in species['data']:
    L.append(info_esp['french_name'])
    
#print(L)
liste_mangeoire = ["Épervier d'Europe", 'Faucon crécerelle', 'Pigeon biset domestique', 'Pigeon colombin', 'Pigeon ramier', 'Tourterelle turque', 'Pic vert', 'Pic épeiche', 'Pic mar', 'Pic épeichette', 'Corneille noire', 'Corbeau freux', 'Choucas des tours', 'Pie bavarde', 'Geai des chênes', 'Mésange charbonnière', 'Mésange bleue', 'Mésange noire', 'Mésange huppée', 'Mésange nonnette', 'Mésange à longue queue', 'Sittelle torchepot',  'Grimpereau des jardins', 'Troglodyte mignon', 'Rougegorge familier', 'Rougequeue noir', 'Rougequeue à front blanc','Merle noir', 'Grive litorne', 'Grive mauvis', 'Grive musicienne', 'Grive draine', 'Fauvette à tête noire', 'Fauvette des jardins',  'Roitelet huppé', 'Roitelet à triple bandeau', 'Accenteur mouchet', 'Bergeronnette grise', 'Étourneau sansonnet', 'Moineau domestique', 'Moineau friquet', 'Moineau soulcie', 'Grosbec casse-noyaux', "Verdier d'Europe", 'Chardonneret élégant', 'Tarin des aulnes', 'Linotte mélodieuse', 'Serin cini',  'Bouvreuil pivoine',  'Pinson des arbres', 'Pinson du Nord', 'Perruche à collier']
for x in liste_mangeoire:
    compte_espece[x] = 0
    
#%%
parameters = {
        "only_with_picture":"1",
        "period_choice":"range",
        "date_from":"1.1.2000",
        "date_to":"31.12.2010"
        }    
#les premières observations datent de 2002 
#Il y a moins de 200 observations entre 2000 et 2010
#Pour cela je fais une requête sur l'ensemble de cette période pour ne pas multiplier les requests

response = requests.post(url = url, auth=authnorm,json=parameters)
re = response.json()
recense_mois = re['data']['forms']

for x in recense_mois:
    observation = x['sightings']
#print(len(observation))

    for j in range(len(observation)):
        try:
            medias = observation[j]['observers'][0]['medias']
            
            for media in medias:
                compte_espece[observation[j]['species']['name']] += 1
                web_path = media['path'] + '/' + media['filename']
        except:
            pass

for annee in liste_annee:
    for mois in liste_mois:
        print(mois,annee)
        recense_oiseaux(mois,annee)

#%%

#print(re)
#donnes = re['data']['forms'][-1]['observers'][0]['media'][0]['@id'] # l'observation la plus récente
#a = len(re['data']['forms'][0]['sightings'])*#12332 a un média
donnes = "12332"
urlmedia = "https://www.oiseauxdesjardins.fr/api/media/"+donnes+"?user_email=biodiguard@gmail.com&user_pw=mautodontha417"

r2 = oauth.get(urlmedia)
media = r2.json()
#print(media)
web = media['data'][0]['photo']
print(web)
r3 = oauth.get(web)
print(r3.content)
#%% 
def telecharge_image(lien,dossier):
    response = requests.get(lien)
    file = open(dossier, "wb")

    file.write(response.content)

    file.close()
    return True
doss = '/Users/maximecauchoix/Downloads/'

#%%
urlink = "https://www.oiseauxdesjardins.fr/api/import_files_observations/892853?user_email=biodiguard@gmail.com&user_pw=mautodontha417"

r3 = oauth.get(urlink)
lien = r3.json()
print(lien)



