.. :changelog:

Changelog
=========

0.8.11 (2020-03-30)
-------------------

- Fix a bug that ``preprocessing.FrequencyEncoder`` fails on unobserved categories by @jeongyoonlee


0.8.10 (2020-03-17)
-------------------

- Add ``preprocessing.FrequencyEncoder`` by @ppstacy

0.8.0 (2019-08-03)
------------------

- Add ``model.BaseAutoML`` and ``model.AutoLGB`` for auto feature selection and hyperparameter tuning
- Add ``.travis.yml``, ``tox.ini`` for test with ``pytest`` and linting with ``flake8``
- Move ``online_model.DecisionTree.ClassificationTree`` to ``online_model.ClassificationTree``
- Fix ``flake8`` warnings and errors
- Fix the macOSX installation issue
