## Rapport technique le 24 juillet 2025

# Segmentation d'Images pour pour le système embarqué d’une voiture autonome

```
Projet réaliser par Djamel Ferguen dans le cadre de la formation AI Engineer d'OpenClassrooms
Projet 8 - Traitez les images pour le système embarqué d'une voiture autonome
```

## Introduction

Dans le cadre de ses travaux sur la **vision embarquée pour véhicules autonomes**, l’entreprise **Future Vision Transport** mène des recherches avancées pour permettre aux voitures de comprendre leur environnement via l’analyse d’images en temps réel.
En tant que membre de l’équipe spécialisée en **intelligence artificielle**, notre responsabilité est de **concevoir et implémenter un module de segmentation sémantique** (composant 3), chargé de traiter les images issues du module de perception (2) avant leur transmission au système de prise de décision (4).

L'objectif est de mettre en place une solution capable de **segmenter avec précision les scènes urbaines** en identifiant **8 grandes catégories d’objets** à partir d’images captées par caméras embarquées.

👉 Cette note décrit le processus de conception, de formation et de déploiement d’un modèle de segmentation optimisé pour l’analyse urbaine.

## Objectifs

- Développer un **modèle de segmentation performant** avec   XXXXXXXXXXX
- Implémenter une **API REST** en utilisant FastAPI
- Concevoir une **interface utilisateur de démonstration** en Streamlit
- Faire le **déploiement sur le Cloud** avec l'environnements Azure  
- **documenter** l’ensemble du projet

## Code Source

L'ensemble du projet ( modéle,API,interface,scripts) est disponible sur le dépot GitHub :
● **Projet 8 : link to GitHub**


## Auteur

Ce projet a été développé par **Djamel Ferguen** dans le cadre du parcours AI Engineer d'OpenClassrooms : 
**Projet 8 -
Traitez les images pour le système embarqué d'une voiture autonome**.

## livrables projet

```
● Les scripts développés sur un notebook
● Une FastAPI déployée sur le Cloud Azure
● Une application (Streamlit) de présentation des résultats
● Une note technique de 10 pages
● Un support de présentation pour l'équipe de Laura

```

## 1. Développement du Modèle de Segmentation

```
Le notebook complet de conception du modèle de segmentation d'image est disponible sur
Colab.
```
### 1.1 Contexte et Enjeux

La **segmentation sémantique appliquée au domaine automobile** constitue un enjeu technologique majeur.

La segmentation sémantique vise à **attribuer une catégorie précise à chaque pixel**.
Cette fonction est essentielle pour les **véhicules autonomes** : 
il est important de distinguer clairement la route,les véhicules, de détecter les piétons, les objets urbains ou encore la végétation.

On parle ici de **prédiction**, car l’objectif est de produire une image annotée où chaque pixel reflète la nature de l’élément qu’il représente.
Ce niveau de **précision** est indispensable pour une compréhension de **l’environnement routier en conditions réelles**.

### 1.2 Analyse du jeu de données

#### 1.2.1 Le jeu de données Cityscapes

Nous avons choisi le jeu de données Cityscapes, une référence dans le domaine de la segmentation urbaine.
Ce dataset présente plusieurs avantages décisifs pour notre cas d'usage :

- **5 000 images avec annotations fines** réparties en 2 975 images d'entraînement, 500 de validation et
    1 525 de test
- **Diversité géographique** : 50 villes européennes différentes
- **Conditions variées** : Plusieurs saisons, conditions météorologiques bonnes à moyennes
- **Richesse des annotations** : 30 classes détaillées avec des annotations polygonales précises
- **Réalisme** : Images capturées depuis des véhicules en circulation réelle

Le dataset Cityscapes du projet offre de nombreux avantages pour l’entraînement d’un modèle de segmentation :

- **5 000 images annotées avec précision**, réparties en 2 975 pour l'entraînement, 500 pour la validation et 1 525 pour les tests
- **Diversité géographique** : les données proviennent de plusieurs villes européennes
- **Conditions de prise de vue variées** : saisons différentes, météo réaliste
- **Annotations de grande qualité** : 34 classes sémantiques détaillées avec annotations polygonales précises ( JSON)
- **Contexte réel** : les images ont été capturées à bord de véhicules en circulation


Selon la demande de Franck et pour garder une corrélation lors de la prise de décision. On  a utilisé le mapping des catégories suivant en gardant seulement 8 catégories principales : 

class_groups = {
'flat': ['road', 'sidewalk', 'parking', 'rail track'],
'human': ['person', 'rider'],
'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', ...],
'construction': ['building', 'wall', 'fence', 'bridge', ...],
'object': ['pole', 'traffic sign', 'traffic light', ...],
'nature': ['vegetation', 'terrain'],
'sky': ['sky'],
'void': ['unlabeled', 'out of roi', ...]
}

#### 1.2.3 Préparation des Données

Les données dans notre projet intégrent plusieurs étapes :

**Redimensionnement intelligent** : Les images Cityscapes originales (2048×1024) sont redimensionnées à
224×224. Cette taille est optimale pour l'utilisation de VGG16 pré-entraîné.

**Train/Validation** : Pour éviter le "data leakage = surapprentissage", nous avons adopté l'approche suivante :

- **dataset 'train'** 
- **dataset 'val'** Pour garantir que notre modéle est réalisés sur de nouvelles données 
- **Augmentation de données** : Nous avons implémenté des techniques d'augmentation adaptées à la segmentation :

##### Voir code script

- Appliqué de manière synchronisée sur l'image et le masque

##### Voir code script

- Appliquée uniquement sur l'image (pas sur le masque)
- Amplitude : ±0.
##### Voir code script

##### Voir code script


### 1.3 Architecture du Modèle : UNet-mini

```
Architecture U-Net mini pour segmentation sémantique d'images
```

#### 1.3.1 Structure générale en forme de U

Nous avons utilisé une architecture en forme de U pour réaliser une segmentation d'images. 

- **Objectif** : Attribuer une classe à chaque pixel de l'image (segmentation).


#### Architecture U-Net Partie encodeur (descendante)

#### Partie décodeur (ascendante)

##### Construction, Entraînement du Modèle et Optimisation

#### Paramètres

- **Total des paramètres** : 5,418,
    - **Entraînables** : 3,575,624 (décodeur et tête de segmentation)
    - **Non-entraînables** : 1,842,976 (backbone gelé)
- Si base_trainable=True, **tous les paramètres deviennent entraînables** , offrant plus de flexibilité
    au prix d'un entraînement plus long.
	
#### Analyse de l'entraînement du modèle


### 1.4 Architecture du Modèle : VGG16 + UNet-mini

#### Architecture U-Net + VGG16 Partie encodeur (descendante)

#### Partie décodeur (ascendante)

##### Construction, Entraînement du Modèle et Optimisation

#### Avec Data Augmentataion 

- **Total des paramètres** : 5,418,
    - **Entraînables** : 3,575,624 (décodeur et tête de segmentation)
    - **Non-entraînables** : 1,842,976 (backbone gelé)
- Si base_trainable=True, **tous les paramètres deviennent entraînables** , offrant plus de flexibilité
    au prix d'un entraînement plus long.
	
#### Sans Data Augmentataion

- **Total des paramètres** : 5,418,
    - **Entraînables** : 3,575,624 (décodeur et tête de segmentation)
    - **Non-entraînables** : 1,842,976 (backbone gelé)
- Si base_trainable=True, **tous les paramètres deviennent entraînables** , offrant plus de flexibilité
    au prix d'un entraînement plus long. 	

#### Analyse de l'entraînement du modèle
```
Code Python
```

#### 1.3.3 Justification du choix architectural

```
None
```
### 1.5 Évaluation et Métriques

#### 1.5.1 Métriques de Performance

L'évaluation de modèles de segmentation nécessite des métriques spécialisées. Nous utilisons
principalement :

#### Accuracy

Mesure le **chevauchement entre la prédiction et la vérité terrain**.
IoU = Intersection / Union
Cette métrique est particulièrement pertinente car elle pénalise à la fois les faux positifs et les faux négatifs.

#### Dice

```
Moyenne des IoU de toutes les classes , offrant une vue d'ensemble des performances.
```
#### Total_loss ( Dice_loss+Crossentropy_loss)


#### IoU  faut-il mettre cette métrique ?

#### Total_loss ( Dice_loss+Crossentropy_loss)

```
Pourcentage de pixels correctement classifiés, métrique intuitive mais pouvant être biaisée par
les classes majoritaires.
```
#### 1.5.2 Résultats   

Insérer imgaes , Notre modèle atteint des performances ( encourageantes : trouve un autre mot) :

Tableau de performances :  Unet-mini ; Unet+VGG16 ( Avec et sans Data Augmentataion)  


#### Analyse par Classe :

```
flat xxxxxxxx   
human xxxxxxx  
vehicle xxxxxxx  
construction xxxxxxx  
object xxxxxxxxxxx 
nature xxxxxxxx  
sky xxxxxxxx  
void xxxxxxxxxx  
```

#### Matrice de confusion ( faut-il mettre !!!)   OUI il faut inétgrer dans le script 

La matrice de confusion constitue un outil d'évaluation pour les modèles de classification, particulièrement
crucial en segmentation sémantique où chaque pixel doit être correctement assigné à sa classe.
Pour un système de vision automobile, cette analyse est importante car elle révèle **quelles confusions
pourraient compromettre la sécurité** (ex: confondre un piéton avec un objet statique).


#### X.X.X Analyse des Performances de l'architecture retenue ( Unet+VGG16) 

#### Points Forts :

- **Excellente détection des surfaces planes** (routes, ciel)
- **Bonne segmentation des grandes structures** (bâtiments, végétation)

#### Points d'Amélioration :

- **Détection des objets fins** (poteaux, panneaux) perfectible
- **Segmentation des humains variable** (due à leur taille et variabilité)
- **Besoin d'optimisation pour les petits objets**
Voici le paragraphe que vous pouvez ajouter à votre documentation technique :

#### 1.5.4 Impact de l'augmentation des données


#### Résultats comparatifs :

```
Métrique Avec Augmentation
(exp_001)
Sans Augmentation
(exp_002)
Écart
Mean IoU 63.25% 64.22% +0.97%
Accuracy 87.95% 87.98% +0.03%
Perte finale 0.4123 0.4107 -0.
```

#### Observations principales :



#### Interprétation :


## 2. Développement et déploiement de l'application "  " 

### 2.1. Développement de l'application-web "  "

### 2.1.1 Backend (FastAPI)  voir projet 7

L'interface communique avec l' **API FastAPI** de ségmentation sémantique d'images via des **requêtes HTTP** :

- **Endpoint** : POST /predict pour l'analyse d'images
- **Format** : Multipart/form-data pour l'upload de fichiers
- **Réponse** : JSON contenant les images encodées et statistiques
Cette architecture **facilite la maintenance** et permet une **évolution future** vers des fonctionnalités


### 2.1.2 frontend (Streamlit) voir projet 7


- Réultat 
### 2.2. Déploiement de l'application-web

- Suivre le même parcours que blob projet 7
 
### 2.2.1 Dépot GitHub voir projet 7

### 2.2.2 Configuration du portail Azure 

### 2.2.3 résultats


## Conclusion et pistes d'Amélioration 
 
Dépot GitHub voir projet 7













Pour valider les performances de notre modèle en conditions réelles, nous avons développé un **pipeline de
prédiction complet** permettant de traiter aussi bien des **images du jeu de données de test** que des
**images externes**. Ce pipeline intègre toutes les étapes nécessaires depuis le **chargement du modèle
jusqu'à l'analyse statistique des résultats**.
def predict_single_image(model, config, image_path=None, image_array=None, img_size=( 224 , 224 )):
"""Réalise une prédiction complète avec préprocessing et post-analyse"""
# Chargement et préprocessing adaptatif
if image_path is not None:
image_pil = Image.open(image_path).convert('RGB')
image_array = np.array(image_pil)
# Pipeline de normalisation identique à l'entraînement
image_resized = tf.image.resize(image_array, img_size, method='bilinear')
image_normalized = tf.cast(image_resized, tf.float32) / 255. 0
image_batch = tf.expand_dims(image_normalized, axis= 0 )
# Prédiction et conversion en masque de classes
predictions = model.predict(image_batch, verbose= 0 )
pred_mask = tf.argmax(predictions, axis=-1)[ 0 ].numpy()
# Analyse statistique automatique par classe
unique_classes, counts = np.unique(pred_mask, return_counts=True)
total_pixels = pred_mask.size
class_stats = []
for class_id, count in zip(unique_classes, counts):
class_name = config['group_names'][class_id]
percentage = (count / total_pixels) * 100
class_stats.append({
'class_id': int(class_id),
'class_name': class_name,
'pixel_count': int(count),
'percentage': percentage
})
return {
'prediction_mask': pred_mask,
'class_statistics': class_stats,
'predictions_raw': predictions[ 0 ]
}

#### 1.6.1. Résultats de prédiction

Les résultats de prédiction se présentent sous la forme suivante :
**Masque de Segmentation (pixels)** : Le résultat principal est un **masque 224×224** où chaque pixel contient
l' **ID de la classe prédite** (0-7). Cette représentation permet une analyse précise de la scène.
**Distribution Statistique des Classes** : Pour chaque prédiction, nous générons automatiquement un tableau
statistique détaillant la répartition des classes :
📊 Classes détectées:


flat: 19536 pixels ( 38. 9 %) # Routes et surfaces de circulation
construction: 14510 pixels ( 28. 9 %) # Bâtiments et infrastructures
nature: 8285 pixels ( 16. 5 %) # Végétation et terrain naturel
void: 5459 pixels ( 10. 9 %) # Zones non classifiées
vehicle: 1630 pixels ( 3. 2 %) # Véhicules en circulation
human: 594 pixels ( 1. 2 %) # Piétons et cyclistes
sky: 98 pixels ( 0. 2 %) # Ciel visible
object: 64 pixels ( 0. 1 %) # Signalisation et mobilier urbain
**Visualisation Tripartite** : Notre système génère systématiquement trois vues :
● Image originale
● Vérité terrain (quand disponible)
● Prédiction colorisée selon notre palette de couleurs spécifique


### 4.3 Analyse qualitative des performances

L'évaluation sur des images réelles révèle des **performances encourageantes** avec des points forts et des
axes d'amélioration clairement identifiés :
**Excellente Détection des Structures Dominantes** : Notre modèle démontre une capacité remarquable à
identifier les éléments structurants de la scène urbaine. La segmentation des surfaces planes (routes,
trottoirs) atteint 38.9% de couverture dans l'exemple analysé, avec des contours nets et une classification
cohérente. Les bâtiments et constructions (28.9%) sont également très bien délimités, ce qui est crucial pour
la navigation autonome.
**Robustesse sur la végétation et l'environnement** : La détection de la végétation (16.5%) montre une
bonne capacité de généralisation du modèle. Les arbres, buissons et espaces verts sont correctement
identifiés malgré leur variabilité de texture et de forme, témoignant de l' **efficacité du transfer learning
depuis ImageNet**.
**Défis sur les objets de petite taille** : L'analyse révèle des difficultés attendues sur les éléments fins et de
petite taille. Les objets de signalisation ne représentent que 0.1% des pixels détectés, ce qui correspond à la
fois à leur **faible représentation spatiale réelle** et aux **limitations du modèle sur ces éléments critiques**.
**Gestion Intelligente des Zones Ambiguës** : La catégorie "void" (10.9%) capture efficacement les zones
d'incertitude et les éléments non classifiables, évitant les fausses classifications qui pourraient être
dangereuses. Cette approche conservative est préférable dans un contexte automobile.
Cette analyse confirme que **notre architecture MobileNetV2-UNet offre un bon équilibre** pour les
contraintes d'un système de vision embarqué, avec des performances robustes sur les éléments critiques
pour la prise de décision de navigation tout en maintenant une efficacité computationnelle compatible avec
les ressources limitées d'un véhicule autonome.

### 1.7 Innovations et optimisations futures

#### 1.7.1 Pistes d'amélioration identifiées

**Entraînement Progressif** Stratégie en deux phases :

##### 1. Phase 1 : Encoder gelé (configuration actuelle)

##### 2. Phase 2 : Fine-tuning avec encoder dégelé et taux d'apprentissage réduit

**Augmentation de données avancée**

- Simulation de conditions météorologiques (pluie, brouillard)
- Variations d'éclairage plus poussées
- Transformations géométriques adaptées au contexte automobile
**Architecture Hybride** Exploration de techniques comme :
- **Attention mechanisms** pour améliorer la détection des petits objets
- **Ensembling de modèles** pour améliorer la robustesse
- **Multi-scale training** pour robustesse aux variations d'échelle


```
None
```
### 1.8 Conclusion de la partie modélisation

Le développement de notre **modèle de segmentation MobileNetV2-UNet** représente un équilibre réussi
entre performance et efficacité computationnelle. Avec un **Mean IoU de 63.25 %** et une **précision globale de
87.95 %** , notre modèle fournit une base solide pour le système de vision du véhicule autonome. Les pistes
d'amélioration identifiées offrent un **roadmap clair pour les développements futurs**. La mise en place d'un
**pipeline MLOps avec MLflow** et notre organisation modulaire garantissent la reproductibilité et facilitent
l' **itération continue** sur les performances du modèle.

## 🔮 2. Développement du Backend FastAPI - API de

## Segmentation Sémantique

Le backend de l'application est une API REST développée avec FastAPI. Cette API **expose notre modèle de
deep learning** ( **MobileNetV2-UNet** ) entraîné pour la **segmentation sémantique d'images urbaines** du
dataset Cityscapes.

### Architecture technique

```
app/backend/
├── main.py # Point d'entrée FastAPI
├── config.py # Configuration et variables d'environnement
├── models/
│ └── predictor.py # Logique de prédiction et chargement du modèle
├── routers/
│ └── segmentation.py # Endpoints de l'API
├── schemas/
│ └── prediction.py # Modèles Pydantic pour validation
├── utils/
│ └── image_processing.py # Utilitaires de traitement d'image
└── requirements.txt # Dépendances Python
```
### Fonctionnalités principales

#### 1. Chargement du modèle depuis MLflow

Le modèle est hébergé sur **MLflow** et chargé dynamiquement au démarrage de l'API :

- **Modèle** : **MobileNetV2-UNet** (format Keras 3.x)
- **Poids** : ~25MB
- **Classes** : 8 catégories (flat, human, vehicle, construction, object, nature, sky, void)

#### 2. Prédiction avec génération d'artefacts

L'API génère pour chaque prédiction :

- **Masque de segmentation** : Classification pixel par pixel
- **Visualisations** : Images côte à côte et superposition
- **Statistiques** : Distribution des classes détectée


```
JSON
JSON
JSON
```
- **Sauvegarde** : Artefacts de prédiction sauvegardés dans le dossier /predictions

### Endpoints de l'API

#### POST /api/v1/segmentation/predict

Notre endpoint principal pour la segmentation d'images.
● **Entrée** : Image (PNG, JPEG) via multipart/form-data
● **Sortie** : JSON contenant :
{
"class_statistics": [...],
"image_size": [largeur, hauteur],
"dominant_class": "flat",
"dominant_class_percentage": 42. 5 ,
"images": {
"original": "data:image/png;base64,...",
"prediction_mask": "data:image/png;base64,...",
"overlay": "data:image/png;base64,...",
"side_by_side": "data:image/png;base64,..."
}
}

#### GET /api/v1/segmentation/health

Ce endpoint vérifie l'état de santé de l'API.
{
"status": "healthy",
"model_loaded": true
}

#### GET /api/v1/segmentation/model/info

Ce endpoint renvoie des informations détaillées sur le modèle chargé.
{
"model_name": "MobileNetV2-UNet",
"input_shape": [null, 224 , 224 , 3 ],
"num_classes": 8 ,
"class_names": ["flat", "human", "vehicle", ...],
"tensorflow_version": "2.18.0"
}


```
JSON
```
#### GET /docs

**Documentation interactive Swagger UI** générée automatiquement par FastAPI. Ce endpoint est **très utile
pour effectuer des requêtes de tests** vers notre API, en attendant de développer le frontend.

### Technologies utilisées

- **FastAPI** : Framework web asynchrone haute performance
- **TensorFlow 2.18** (CPU) : Inférence du modèle
- **Gunicorn + Uvicorn** : Serveur ASGI pour la production
- **Pillow** : Manipulation d'images
- **MLflow** : Gestion et versioning du modèle

### Optimisations pour la production

##### 1. TensorFlow CPU : Version allégée sans GPU (250MB vs 500MB)

##### 2. Workers limités : 1 worker pour économiser la RAM

##### 3. Timeout ajusté : 120s pour les prédictions complexes

##### 4. Stockage temporaire : Utilisation de /tmp sur Railway

### Déploiement sur Railway

**Notre backend FastAPI est déployée sur Railway** qui offre des ressources suffisantes pour déployer un
modèle de segmentation d'image, **gratuitement**.

#### Configuration Railway

Le déploiement sur Railway utilise un fichier railway.json :
{


```
"$schema": "https://railway.app/railway.schema.json",
"build": {
"builder": "NIXPACKS",
"buildCommand": "pip install -r requirements.txt"
},
"deploy": {
"startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
"restartPolicyType": "ON_FAILURE",
"restartPolicyMaxRetries": 10
}
}
```
#### Variables d'environnement

Les "credentials" sont configurés via le **dashboard Railway** :

- MLFLOW_TRACKING_URI : URL du serveur MLflow
- AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY : Accès S3 pour MLflow
- RUN_ID : Identifiant de l'expérience MLflow
- PORT : Port d'écoute (géré automatiquement)
- FRONTEND_URL : URL de l'interface frontend (pour éviter les erreurs CORS)

#### Processus de déploiement

##### 1. Initialisation : railway init pour créer le projet

##### 2. Configuration : Ajout des variables via le dashboard

##### 3. Déploiement : railway up pour déployer le code

##### 4. Domaine : railway domain pour générer l'URL publique

#### URL de production

**L'API est accessible à : https://future-vision-api-production.up.railway.app/docs**
Cette architecture permet un déploiement rapide et scalable d'un modèle de deep learning, avec une API
REST moderne et bien documentée, idéale pour être consommée par une application frontend.


## ⚡ 3. Développement de l'interface frontend Next.js

### 3.1 Architecture et choix techniques

**L'interface utilisateur a été développée avec Next.js 15.3 en JavaScript** , privilégiant la simplicité et la
performance. Le framework React permet de développer une interface adaptée aux besoins de **visualisation
des résultats de segmentation**.

#### Stack technique

- **Next.js 15.3** : **Framework React** pour le développement frontend
- **Bootstrap 5.3** : **Framework CSS** pour un design responsive et professionnel
- **Fetch API** : Communication avec l' **API FastAPI de prédiction**
- **Base64** : Gestion des **images encodées** retournées par l'API

### 3.2 Fonctionnalités principales

L'application propose une interface intuitive pour tester le modèle de segmentation sémantique :

- **Upload d'images** : Sélection et prévisualisation des images (formats JPEG, PNG, GIF)
- **Validation** : Contrôle de la taille maximale (4096x4096px) et du format
- **Prédiction en temps réel** : Appel à l'API avec feedback visuel de progression
- **Visualisation multi-format** :
    - **Image avec overlay** de segmentation semi-transparent
    - **Comparaison côte à côte** (originale vs prédiction)
    - **Masque de segmentation** isolé
- **Statistiques détaillées** : Distribution des classes, classe dominante, métadonnées


### 3.3 Interface utilisateur

L'interface suit les principes de design UX/UI pour une utilisation professionnelle.

#### État initial

- **Formulaire d'upload central** avec instructions claires
- Validation en temps réel du fichier sélectionné
- Boutons d'action désactivés tant qu'aucune image n'est sélectionnée

#### Résultats de prédiction

**- Affichage principal de l'image avec overlay de segmentation**
- **Panneau latéral avec statistiques** et distribution des classes
- Section de comparaison détaillée avec visualisations multiples
- **Design responsive** adapté aux différentes tailles d'écran


### 3.4 Déploiement et Accessibilité

L'application est déployée sur Vercel :
**URL de production : https://oc-ai-engineer-p08-images-systeme-v.vercel.app/**

- **Déploiement automatique** : Intégration continue depuis le repository Git
- **Configuration** : Variables d'environnement pour l'URL de l'API backend
    - NEXT_PUBLIC_API_URL : URL de l'API REST de prédiction

### 3.5 Intégration API

L'interface communique avec l' **API FastAPI** de ségmentation sémantique d'images via des **requêtes HTTP** :

- **Endpoint** : POST /predict pour l'analyse d'images
- **Format** : Multipart/form-data pour l'upload de fichiers
- **Réponse** : JSON contenant les images encodées et statistiques
Cette architecture **facilite la maintenance** et permet une **évolution future** vers des fonctionnalités
avancées comme la gestion de lots d'images ou l'historique des prédictions.

## 📋 Conclusion

Ce projet de développement d'un **système de segmentation d'images pour véhicule autonome** a permis
de concevoir une **solution complète** , de la **modélisation** au **déploiement en production**.

### Réalisations techniques

**Modélisation IA** : Nous avons développé un modèle **MobileNetV2-UNet** atteignant **63.25% de Mean IoU**
sur 8 catégories d'objets urbains. L'architecture hybride combine l' **efficacité de MobileNetV2** pour les
contraintes embarquées avec la **précision de U-Net** pour la reconstruction spatiale.
**Pipeline MLOps** : L'intégration de **MLflow** a permis un **suivi rigoureux des expériences** et le versioning des
modèles, garantissant la reproductibilité et la traçabilité des développements.
**Architecture API** : Le **backend FastAPI expose le modèle via une API REST moderne** avec **génération
automatique de visualisations** et **statistiques détaillées** , **déployée sur Railway**.
**Interface utilisateur** : L'application **Next.js** offre une **interface intuitive** pour tester le modèle avec
visualisations multi-formats (overlay, comparaison, masques) et facilement **déployée sur Vercel**.

### Axes d'Amélioration

Les principales pistes d'optimisation incluent un **entraînement progressif** (encoder dégelé en phase 2), une
**augmentation de données avancée** simulant conditions météorologiques et variations d'éclairage,
l'intégration d' **attention mechanisms** pour améliorer la détection des petits objets (panneaux, poteaux).
Également, l'ajout d'un **système d'authentification API** , d'un **cache de prédictions** et de **métriques de
monitoring** en production compléterait l'industrialisation du système.


### A propos de ce projet

Ce projet démontre la faisabilité technique d'un **module de segmentation pour Future Vision Transport** ,
avec une base solide pour l'évolution vers un **système de production robuste**.

### Récapitulatif des livrables

```
● GitHub - DavidScanu/oc-ai-engineer-p08-images-systeme-voiture-autonome
● Notebook de modélisation sur Colab
● Backend API REST [FastAPI]
● Interface utilisateur [Next.js]
● Rapport technique
● Article complet disponible sur dev.to
● Support de présentation pour l’équipe de Laura
```

