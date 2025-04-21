### Guide Git : Travailler avec une Branche à partir de `develop`

Ce guide explique comment créer une nouvelle branche à partir de `develop`, y apporter des modifications, puis soumettre une Pull Request (PR) pour validation.

---

## 1. **Mettre à jour `develop`**

Avant de créer une nouvelle branche, assurez-vous que votre branche `develop` est à jour avec le dépôt distant.

```sh
git checkout develop
git pull origin develop
```

Cela garantit que vous travaillez avec la dernière version du code.

---

## 2. **Créer et basculer sur une nouvelle branche**

Créez une nouvelle branche à partir de `develop`. Nommez-la de manière descriptive en fonction de la fonctionnalité ou du correctif à apporter.

```sh
git checkout -b feature/nom-de-la-fonctionnalité
```

> **Exemple :** `feature/ajout-authentification`

---

## 3. **Développer et commit ses modifications**

Effectuez vos modifications et ajoutez-les à votre commit.

```sh
git add .
git commit -m "Ajout de l'authentification des utilisateurs"
```

Si vous devez modifier plusieurs fichiers, pensez à faire des commits atomiques (petits et compréhensibles).

---

## 4. **Pousser la branche sur le dépôt distant**

Envoyez votre branche sur le dépôt distant pour la partager avec l’équipe.

```sh
git push origin feature/nom-de-la-fonctionnalité
```

---

## 5. **Créer une Pull Request (PR) vers `develop`**

1. Rendez-vous sur votre plateforme de gestion de code (GitHub, GitLab, Bitbucket...).
2. Ouvrez une nouvelle PR en sélectionnant :
   - **Branche source** : `feature/nom-de-la-fonctionnalité`
   - **Branche de destination** : `develop`
3. Ajoutez une description claire du travail effectué.
4. Assignez des relecteurs et mettez des labels si nécessaire.

---

## 6. **Relecture et validation**

- Les relecteurs examinent votre code et laissent des commentaires.
- Apportez les modifications demandées si nécessaire.
- Une fois approuvé, la PR peut être fusionnée dans `develop`.

---

## 7. **Supprimer la branche après fusion**

Une fois votre PR fusionnée, supprimez la branche localement et à distance :

```sh
git branch -d feature/nom-de-la-fonctionnalité
git push origin --delete feature/nom-de-la-fonctionnalité
```

---

### **Bonnes pratiques**

✅ Travaillez avec des branches courtes et spécifiques.  
✅ Faites des commits clairs et explicites.  
✅ Demandez une relecture avant de fusionner.
