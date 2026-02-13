from sales_analysis import load_data, aggregate_sales, train_linear_regression, arima_forecast

# Change path if needed
DATA_PATH = "data/walmart_sales.csv"

def test_load_data():
    df = load_data(DATA_PATH)
    assert not df.empty, "DataFrame is empty"
    print("✔ load_data test passed")


def test_aggregate_sales():
    df = load_data(DATA_PATH)
    sales = aggregate_sales(df)
    assert 'Weekly_Sales' in sales.columns, "Weekly_Sales column missing"
    print("✔ aggregate_sales test passed")


def test_linear_regression():
    df = load_data(DATA_PATH)
    sales = aggregate_sales(df)
    model, predictions, mae = train_linear_regression(sales)
    assert mae > 0, "MAE should be positive"
    print("✔ Linear Regression test passed")


def test_arima():
    df = load_data(DATA_PATH)
    sales = aggregate_sales(df)
    forecast = arima_forecast(sales, steps=3)
    assert len(forecast) == 3, "Forecast length incorrect"
    print("✔ ARIMA test passed")


if __name__ == "__main__":
    test_load_data()
    test_aggregate_sales()
    test_linear_regression()
    test_arima()
    print("\n🎉 All tests passed successfully!")
