from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import joblib

# 데이터셋 로드 (여기서는 대신 iris 데이터셋을 사용함)
# 실제  데이터셋으로 변경 필요
iris = datasets.load_iris()
X = iris.data
y = iris.target

#  데이터셋을 학습용과 테스트용으로 분리
# test_size는 필요에 따라 조정 가능
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 데이터 스케일링 (옵션,  데이터셋에 필요한 경우에만 사용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 로지스틱 회귀 모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 모델 평가
# 실제 코드에서는 적절한 평가 지표를 선택해서 사용해야 함
score = model.score(X_test, y_test)
print('Accuracy:', score)


joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'model.pkl')