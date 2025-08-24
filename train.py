# %%
import pandas as pd

df = pd.read_csv("data/dataset.csv")
df.head()
# %%
for i in df.columns:
    print(i)

features = {
    "1.a_idade": "idade",
    "1.b_genero": "genero",
    "1.d_pcd": "pcd",
    "1.i.1_uf_onde_mora": "ufOndeMora",
    "2.f_cargo_atual": "cargoAtual",
    "2.g_nivel": "nivel",
    "2.i_tempo_de_experiencia_em_dados": "tempoDeExperienciaemDados",
    "2.j_tempo_de_experiencia_em_ti": "tempodeExperienciaEmTi",
}

target = "2.h_faixa_salarial"

columns = list(features.keys()) + [target]
df = df[columns].copy()
df.rename(columns=features, inplace=True)

# %%
depara_salario = {
    'Menos de R$ 1.000/mês': '01 - Menos de R$ 1.000/mês',
    'de R$ 1.001/mês a R$2.000/mês': '02 -de R$ 1.001/mês a R$2.000/mês',
    'de R$ 2.001/mês a R$ 3.000/mês': '03 -de R$ 2.001/mês a R$ 3.000/mês',
    'de R$ 3.001/mês a R$ 4.000/mês': '04 -de R$ 3.001/mês a R$ 4.000/mês',
    'de R$ 4.001/mês a R$ 6.000/mês': '05 -de R$ 4.001/mês a R$ 6.000/mês',
    'de R$ 6.001/mês a R$ 8.000/mês': '06 -de R$ 6.001/mês a R$ 8.000/mês',
    'de R$ 8.001/mês a R$ 12.000/mês': '07 -de R$ 8.001/mês a R$ 12.000/mês',
    'de R$ 12.001/mês a R$ 16.000/mês': '08 -de R$ 12.001/mês a R$ 16.000/mês',
    'de R$ 16.001/mês a R$ 20.000/mês': '09 -de R$ 16.001/mês a R$ 20.000/mês',
    'de R$ 20.001/mês a R$ 25.000/mês': '10 -de R$ 20.001/mês a R$ 25.000/mês',
    'de R$ 25.001/mês a R$ 30.000/mês': '11 -de R$ 25.001/mês a R$ 30.000/mês',
    'de R$ 30.001/mês a R$ 40.000/mês': '12 -de R$ 30.001/mês a R$ 40.000/mês',
    'Acima de R$ 40.001/mês': '13 -Acima de R$ 40.001/mês'
}

df[target] = df[target].replace(depara_salario)

df_not_na = df[~df[target].isna()]

# %%
X = df_not_na[features.values()]
Y = df_not_na[target]

from sklearn import model_selection

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)


X_train.to_csv("data/template.csv",index=False)

from feature_engine import imputation, encoding
from sklearn import ensemble, pipeline
from sklearn import metrics

# Imputação de categorias faltantes
input_classe = imputation.CategoricalImputer(
    fill_value="Não informado",
    variables=["ufOndeMora", "cargoAtual", "nivel"]
)

# One-hot encoding das variáveis categóricas
onehot = encoding.OneHotEncoder(
    variables=[
        'genero',
        'pcd',
        'ufOndeMora',
        'cargoAtual',
        'nivel',
        'tempoDeExperienciaemDados',
        'tempodeExperienciaEmTi'
    ]
)

# Random Forest - Modelo Atual
#clf = ensemble.RandomForestClassifier(random_state=42, min_samples_leaf=20)

clf = ensemble.GradientBoostingClassifier(n_estimators=500, learning_rate=0.6)



# Pipeline completo
modelo = pipeline.Pipeline(
    steps=[
        ('inputador', input_classe),
        ('encoder', onehot),
        ('algoritmo', clf)
    ]
)


# %%
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_id="568823065322822667")
# %%

with mlflow.start_run():

    mlflow.sklearn.autolog()

    # ISSO AQUI É OQUE CHAMAMOS DE MACHINE LEARNING
    modelo.fit(X_train,Y_train)
    Y_train_predict = modelo.predict(X_train)
    acc_train = metrics.accuracy_score(Y_train, Y_train_predict)

    Y_test_predict = modelo.predict(X_test)
    acc_test = metrics.accuracy_score(Y_test, Y_test_predict)

