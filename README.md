# Kata Refactor Design Patterns

Ce dépôt contient des katas de refactoring conçus pour apprendre et pratiquer différents design patterns. L'objectif est de mieux comprendre comment structurer et améliorer le code en utilisant des patterns classiques et modernes.

## Objectif

- **Apprentissage des design patterns** : Mettre en pratique les patterns classiques du GoF (Gang of Four) ainsi que des patterns plus récents adaptés aux besoins modernes, notamment dans le domaine du Machine Learning et de la Data Science.
- **Amélioration des compétences en refactoring** : Identifier les opportunités d'amélioration dans le code et appliquer les patterns appropriés pour le rendre plus lisible, maintenable et extensible.

## Contenu

- **Strategy** : Exemple d'implémentation avec la normalisation de données dans `strategy/normalization.py`.
- **Pipeline** : Exemple d'implémentation d'un pipeline de traitement d'images dans `pipeline/image_pipeline.py` et `pipeline/image_pipeline_dsl.py`.
- **Pipeline DSL** : Une approche fonctionnelle pour composer des étapes de traitement d'images avec un Domain Specific Language (DSL).

## Structure du projet

- `strategy/` : Contient des exemples liés au pattern Strategy.
- `pipeline/` : Implémentations de pipelines pour le traitement d'images.
- `tests/` : Tests unitaires pour valider les implémentations.
- `.vscode/` : Configuration pour l'environnement de développement (Pytest activé).

## Prérequis

- Python 3.8 ou supérieur
- Bibliothèques nécessaires : `numpy`, `pytest`

## Installation

1. Clonez le dépôt :
   ```bash
   git clone <url-du-repo>
   cd kata-refactor-design-patterns
   ```
2. Créez un environnement virtuel :
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Sous Windows : .venv\Scripts\activate
   ```
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Tests

Pour exécuter les tests :
```bash
pytest
```

## Notes

Ces exercices sont principalement destinés à un usage personnel, mais peuvent être utiles à d'autres développeurs souhaitant pratiquer les design patterns. N'hésitez pas à explorer et adapter les exemples à vos propres besoins.

## Inspirations

- **Design Patterns: Elements of Reusable Object-Oriented Software** (GoF)
- [Machine Learning Design Patterns](https://www.oreilly.com/library/view/machine-learning-design/9781098115777/)
- **[Refactoring Guru](https://refactoring.guru)** : Une ressource utile pour comprendre les design patterns et le refactoring.
- **Clean Code** : Un livre essentiel pour apprendre à écrire un code propre et maintenable.


---

Bon apprentissage et bon refactoring !