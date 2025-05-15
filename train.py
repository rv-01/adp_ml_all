import joblib
from preprocess import X
from models import isolation_forest, dbscan, autoencoder, ocsvm

iso_cv = isolation_forest.build_model()
iso_cv.fit(X)
joblib.dump(iso_cv, 'models/iso_cv.pkl')

db = dbscan.build_model()
db.fit(X)
joblib.dump(db, 'models/dbscan.pkl')

ae = autoencoder.build_model(X.shape[1])
ae.fit(X, X, epochs=50, batch_size=32, validation_split=0.1)
ae.save('models/autoencoder.h5')

svm_cv = ocsvm.build_model()
svm_cv.fit(X)
joblib.dump(svm_cv, 'models/ocsvm.pkl')
