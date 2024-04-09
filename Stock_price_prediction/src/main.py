from feature_engineering import *
from data_preprocessing import *
from model_building import *
from model_evaluation import *

df = get_stock_data('NIO','1mo')
print(df.head(5))

df2 = preprocess_stock_data(df)
print(df2.head(5))

df3 = cal_ex_moving_avg(df2,3)
print(df3.head(5))

df4 = calculate_rsi(df3,window_days = 7)
print(df4.head(5))

model,X_train, X_test, y_train, y_test = perform_linear_regression(df4)

predicted = model_prediction(X_test, model)

df_compaison=pd.DataFrame({'Actual_Price': y_test, 'Predicted_Price':predicted})
print(df_compaison.head(10))

print(mean_absolute_percentage_error(y_true = y_test, y_pred =predicted ))
print(evaluate_regression_model(y_true = y_test , y_pred = predicted))

