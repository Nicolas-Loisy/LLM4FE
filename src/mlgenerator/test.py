from mlclassifier import MLClassifier


# Étape 1 : Chargement initial du dataset
dataset_name = "loan_approval_dataset"  # sans le .csv
target_column = "loan_status"  # remplace par le nom réel de ta colonne cible

# Étape 2 : Création de l'objet MLClassifier
prep = MLClassifier(dataset=dataset_name, target_col=target_column)

# Étape 3 : Chargement du dataset
df = prep.get_init_dataset()

# Étape 4 : Mise à jour de l'objet avec le dataset chargé
prep._dataset = df

# Étape 5 : Nettoyage des données manquantes
df_clean = MLClassifier.get_missing_values_free_dataset(df)
prep._dataset = df_clean

# Étape 6 : Suppression des doublons
df_nodup = prep.del_duplicate()
prep._dataset = df_nodup

# Étape 7 : Encodage des colonnes non numériques
df_encoded = prep.encoding_label_col()
prep._dataset = df_encoded

# Étape 8 : Extraction de la colonne cible
target = df_encoded[target_column]
prep._target_col = target


# Suppression des colonnes spécifiques (loan_id et la colonne cible)
df_supp = df_encoded.drop(columns=[target_column, 'loan_id'], axis=1)

# Mettre à jour self._dataset avec le DataFrame modifié
prep._dataset = df_supp

# Vérification des colonnes restantes
# Étape 9 : Validation croisée + entraînement + score
prep.cross_validation(size=0.2)
