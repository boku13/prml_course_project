# Pattern Recognition and Machine Learning Course Project (PRML 2050)
----------------------------------

### MID TERM REPORT:

https://grave-pyroraptor-4c4.notion.site/PRML-Course-Project-Mid-Term-Report-76c7132de8c049b692f15099b201a3e9

----------------------------------

### DEMONSTRATION:

- Visit the following webpage to use the retriever: http://boku.pythonanywhere.com/
- Or else, 
    -  Set up the project locally using `git clone "project url"`
    -  run 'pip install -r requirements.txt'
    -  run 'python viewer.py'

----------------------------------

### TEAM MEMBERS:
1. SHYAM SATHVIK
2. NEERMITA BHATTACHARYA
3. SIDDESH AYYATHAN
4. SAKSHI SHARMA
5. GUBBALA BHARGAVI


----------------------------------

### File Structure (few omitted)

```
├── README.md
├── cli
│   ├── cli_tool.py
│   └── tool.py
├── data
│   ├── cifar10
│   ├── data_handler.py
│   └── retriever
├── data.py
├── models
│   ├── neermita
│   │   ├── ada_boost_model.pkl
│   │   ├── decision_tree_model.pkl
│   │   ├── knn_model.pkl
│   │   └── random_forest_model.pbz2
│   ├── default
│   │   ├── AdaBoost.pkl
│   │   ├── PCA.pkl
│   │   └── svm.pkl
│   ├── siddesh
│   │   ├── sid_dt_classifier_hog.pkl
│   │   ├── sid_knn_classifier_hog.zip
│   │   └── sid_rf_classifier_hog.zip
│   └── svm_model.pkl
├── notebooks
│   ├── bhargavi
│   │   ├── bhargavi_sift.ipynb
│   │   └── bhargavi_sift.py
│   ├── neermita
│   │   ├── CNN_clustering.ipynb
│   │   ├── Clustering_with_PCA+LDA+tSNE.ipynb
│   │   ├── PRML_PROJECT.ipynb
│   │   └── PRML_PROJECT_save_projection_vector.ipynb
│   ├── sakshi
│   │   └── Sakshi_lda_3_classifiers.ipynb
│   ├── shyam
│   │   ├── cosine_similarity.ipynb
│   │   ├── prml_pre_mid_report_experiment_0.ipynb
│   │   └── simple_fft_transformed_cifar10_for_classification.ipynb
│   └── siddesh
│       ├── SIDDHESH_CLUSTER.ipynb
│       └── SIDDHESH_HOG.ipynb
├── reports
│   ├── Using PCA as a dimensionality reduction technique.docx
│   └── figures
├── requirements.txt
├── retriever.py    (Retriever)
├── retriever_clustering.py
└── viewer.py       (PyQT app)
```
