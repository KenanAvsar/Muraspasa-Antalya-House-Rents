
# ## Kütüphaneleri Yükle ##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ## Veri Setini Yükle ##

ev_info=pd.read_csv("/kaggle/input/antalya-muratpaa-daire-kira-cretleri-2024-ocak/antalya_kiralik_ev.csv")


# ## Keşifsel Veri Analizi ##

ev_info.head()


# ## Unnamed: 0 Kolonunu Sil ##

ev_info.drop("Unnamed: 0",axis=1,inplace=True)


ev_info.info()


# ## Veri Görselleştirme ##

ev_info['mahalle'].value_counts(normalize=True)


# ## Fiyat Görselleştirme ##

ev_info['fiyat'].plot.hist(bins=20)


# ## Betimsel İstatistikler ##

ev_info.describe(include="all").T


# ## Eksik Veri Kontrolü ##

ev_info.isnull().sum()


ev_info['isitma_turu'].value_counts().plot.bar()

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, StackingRegressor, VotingRegressor
from catboost import CatBoostRegressor

def all_reg_models(X_train, X_test, y_train, y_test):
    # Tanımlanan Modeller
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor(),
        "LightGBM": LGBMRegressor(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "SVR": SVR(),
        "MLPRegressor": MLPRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "CatBoost": CatBoostRegressor(verbose=0), 
        "Extra Trees": ExtraTreesRegressor(),
        "Bagging": BaggingRegressor()
       
    }

    results = {}
    for name, model in models.items():
        # Modeli eğitme
        model.fit(X_train, y_train)
        # Test seti üzerinde tahmin yapma
        predictions = model.predict(X_test)
        # MSE ve R^2 değerlerini hesaplama
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)  # MSE'nin karekökünü alarak RMSE hesaplama
        r2 = r2_score(y_test, predictions)
        # Sonuçları saklama
        results[name] = (mse, rmse, r2)

    # Sonuçları yazdırma
    for name, (mse, rmse, r2) in results.items():
        print(f"{name}: Average RMSE: {rmse:.2f}")
        print(f"{name}: R2: {r2:.2f}")

    # En iyi modeli bulma (En düşük MSE'ye göre)
    best_model_name = min(results, key=lambda x: results[x][0])
    best_model_mse, best_model_rmse, best_model_r2 = results[best_model_name]
    print(50*"*")
    print(f"\nBest Performing Model: {best_model_name} with Average RMSE: {best_model_rmse:.2f} and R2: {best_model_r2:.2f}")

sns.boxplot(x='fiyat', data=ev_info)


ev_info[ev_info['fiyat'] == 200000]


ev_fiyat_mahalle = ev_info.groupby('mahalle')['fiyat'].mean().sort_values(ascending=False).reset_index(name='Ortalama Fiyat')
ev_fiyat_mahalle


sns.set_style('whitegrid')
plt.figure(figsize=(12,8))
sns.barplot(x='Ortalama Fiyat', y= 'mahalle', data=ev_fiyat_mahalle, palette ='viridis')
plt.title('Mahallelere Göre Ortalama Fiyat')
plt.xlabel('Ortalama Fiyat')
plt.ylabel('Mahalleler')
plt.tight_layout()
plt.show()


top_10_ev_fiyat_mahalle = ev_fiyat_mahalle.head(10)
plt.figure(figsize=(12,8))
sns.barplot(x='Ortalama Fiyat', y= 'mahalle', data=top_10_ev_fiyat_mahalle, palette ='cubehelix')
plt.title('Mahallelere Göre Ortalama Fiyat')
plt.xlabel('Ortalama Fiyat')
plt.ylabel('Mahalleler')
plt.tight_layout()
plt.show()


ev_bina_yas=ev_info['bina_yas'].value_counts().sort_values()
ev_bina_yas


plt.figure(figsize=(12,8))
sns.barplot(x=ev_bina_yas.index,y=ev_bina_yas.values,palette='inferno')
plt.title('Bina Yaşlarına Göre Daire Sayıları')
plt.xlabel('Bina Yaşları')
plt.ylabel('Daire Sayıları')
plt.tight_layout()


ev_info['oda_sayisi'].value_counts()


plt.figure(figsize=(12,6))
sns.histplot(ev_info['oda_sayisi'].sort_values(ascending=True),bins=10,palette='twilight')
plt.title('Oda Sayısına Göre Daire Sayıları')
plt.xlabel('Daire Sayıları')
plt.ylabel('Oda Sayıları')
plt.tight_layout()
plt.xticks(rotation=30);


ev_net = ev_info['net_alan_m2'].value_counts().sort_index(ascending=True)
ev_net


ev_isitma_turu = ev_info['isitma_turu'].value_counts().sort_values(ascending=False)
ev_isitma_turu


plt.figure(figsize=(12,6))
sns.barplot(x=ev_isitma_turu.values,y=ev_isitma_turu.index,palette='viridis')
plt.xlabel('Daire Sayıları')
plt.ylabel('Isıtma Türleri')
plt.tight_layout()
plt.show()


ev_fiyat_isitma = ev_info.groupby('isitma_turu')['fiyat'].mean().sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x=ev_fiyat_isitma.values, y=ev_fiyat_isitma.index, palette='magma')
plt.title('Isıtma Türünün Fiyata Etkisi')
plt.xlabel('Fiyat')
plt.ylabel('Isıtma Türü')
plt.tight_layout()


pd.set_option('display.max_rows',None)
ev_alan_fiyat = ev_info.groupby('net_alan_m2')['fiyat'].mean().sort_values()
ev_alan_fiyat


sns.boxplot(x='net_alan_m2',data=ev_info)


ev_info_kor=ev_info[['fiyat','brut_alan_m2','net_alan_m2','bina_kat_sayisi','banyo_sayisi','aidat','depozito']]
outliers=ev_info_kor.quantile(q=.99)
outliers


ev_info.sort_values('depozito',ascending=False)


ev_info_non_outliers = ev_info[ev_info['fiyat']<outliers['fiyat']]
ev_info_non_outliers = ev_info_non_outliers[ev_info_non_outliers['brut_alan_m2']<outliers['brut_alan_m2']]
ev_info_non_outliers = ev_info_non_outliers[ev_info_non_outliers['net_alan_m2']<outliers['net_alan_m2']]
ev_info_non_outliers = ev_info_non_outliers[ev_info_non_outliers['bina_kat_sayisi']<outliers['bina_kat_sayisi']]
ev_info_non_outliers = ev_info_non_outliers[ev_info_non_outliers['banyo_sayisi']<outliers['banyo_sayisi']]
ev_info_non_outliers = ev_info_non_outliers[ev_info_non_outliers['aidat']<outliers['aidat']]
ev_info_non_outliers = ev_info_non_outliers[ev_info_non_outliers['depozito']<outliers['depozito']]


ev_info_non_outliers.info()


ev_info_non_outliers.isnull().sum()


from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer


encoded_ev_info = ColumnTransformer(
    transformers=[
        ('obj',OrdinalEncoder(),['mahalle','oda_sayisi','bina_yas','dairenin_bulundugu_kat','isitma_turu','otopark'])
                 ],
)


encoded_fitted_data = encoded_ev_info.fit_transform(ev_info_non_outliers)
encoded_fitted_data


encoded_columns = ['mahalle','oda_sayisi','bina_yas','dairenin_bulundugu_kat','isitma_turu','otopark']
encoded_ev_info_df = pd.DataFrame(encoded_fitted_data,columns=encoded_columns)
encoded_ev_info_df.isnull().sum()


ev_info_non_outliers_dropped = ev_info_non_outliers.drop(columns=['mahalle','oda_sayisi','bina_yas','dairenin_bulundugu_kat','isitma_turu','otopark'])
ev_info_non_outliers_dropped


encoded_ev_info_df = encoded_ev_info_df.reset_index(drop=True)
ev_info_non_outliers_dropped = ev_info_non_outliers_dropped.reset_index(drop=True)


ev_info_son = pd.concat([encoded_ev_info_df,ev_info_non_outliers_dropped],axis=1)


ev_info_son.isnull().sum()


ev_info_son.info()


ev_info_son


y = ev_info_son['fiyat']
X = ev_info_son.drop(['fiyat'],axis=1)


sc = StandardScaler()
X_sc = sc.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split (X_sc,y,test_size=.2,random_state=42)


all_reg_models(X_train,X_test,y_train,y_test)


model = XGBRegressor()


model.fit(X_train,y_train)
y_pred = model.predict(X_test)
r2=r2_score(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred)**.5
print('Model in R2 Skoru:' , r2,
      '\nModel in RMSE: ', rmse)

